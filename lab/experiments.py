"""Seven experiments. Each pairs a slow query shape with a faster one, records
the median of 3 runs (after one warm-up run), the metrics Spark reports, and the
plan node that gives the slow shape away. Results go to results/results.json.

Every timed query is rebuilt from scratch on each run. Re-running the same
DataFrame object lets Spark skip stages whose shuffle output already exists,
which makes the second run look impossibly fast.

Run:  python experiments.py            (all)
      python experiments.py joins udf  (a subset)
"""
import json
import os
import sys
import time

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

from common import DATA, measure, plan_text, session

OUT = "../results/results.json"
UA = "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Mobile Safari/537.36"
results = json.load(open(OUT)) if os.path.exists(OUT) else {}


def has(df, names):
    p = plan_text(df)
    return [n for n in names if n in p]


def save():
    os.makedirs("../results", exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=2)


def events(spark, path="events"):
    return spark.read.parquet(f"{DATA}/{path}")


def products(spark):
    return spark.read.parquet(f"{DATA}/products")


# 0. Lazy evaluation: building the plan is cheap, the action pays for all of it.
def lazy():
    spark = session("lazy")

    def build():
        return (events(spark).filter(F.col("event_type") == "purchase")
                .withColumn("price_band", (F.col("price") / 50).cast("int"))
                .groupBy("product_id", "price_band").agg(F.sum("price").alias("revenue")))
    build()  # warm the catalog
    t0 = time.perf_counter()
    q = build()
    plan_s = time.perf_counter() - t0
    m = measure(spark, "lazy-action", lambda: build().count())
    results["lazy"] = {"plan_build_s": round(plan_s, 3), "action": m,
                       "pushed_filter": "EqualTo(event_type,purchase)" in plan_text(q)}
    spark.stop()


# 1. Joins: let the small side broadcast, versus forcing a sort merge join.
def joins():
    out = {}
    for label, conf in [("broadcast_hash", {}),
                        ("sort_merge", {"spark__sql__autoBroadcastJoinThreshold": "-1"})]:
        spark = session(f"join-{label}", **conf)

        def build():
            return (events(spark).join(products(spark), "product_id")
                    .groupBy("category").agg(F.count("*").alias("n")))
        m = measure(spark, label, lambda: build().collect())
        m["plan_nodes"] = has(build(), ["BroadcastHashJoin", "SortMergeJoin"])
        out[label] = m
        spark.stop()
    results["joins"] = out


# 2. OR in a join condition: no equi-join key, so Spark falls back to a nested loop.
def or_join():
    out = {"rows": {"events": 1_000_000, "lookup": 2_000}}
    spark = session("or-join")

    def ev2():
        return (events(spark).limit(1_000_000)
                .withColumn("alt_sku", F.regexp_replace("product_id", "^p-", "sku-")))

    def bad():
        e, lk = ev2(), products(spark).select("product_id", "alt_sku", "brand")
        return (e.join(lk, (e.product_id == lk.product_id) | (e.alt_sku == lk.alt_sku), "left")
                 .agg(F.countDistinct(lk.brand)))

    def good():
        e, lk = ev2(), products(spark)
        a = e.join(lk.select("product_id", F.col("brand").alias("b1")), "product_id", "left")
        return (a.join(lk.select("alt_sku", F.col("brand").alias("b2")), "alt_sku", "left")
                 .agg(F.countDistinct(F.coalesce("b1", "b2"))))
    names = ["BroadcastNestedLoopJoin", "CartesianProduct", "BroadcastHashJoin"]
    out["or_condition"] = {**measure(spark, "or_condition", lambda: bad().collect()),
                           "plan_nodes": has(bad(), names)}
    out["two_equi_joins"] = {**measure(spark, "two_equi_joins", lambda: good().collect()),
                             "plan_nodes": has(good(), names)}
    out["same_answer"] = bad().first()[0] == good().first()[0]
    spark.stop()
    results["or_join"] = out


# 3. Skew: the automatic skew-join fix only fires on partitions above 256 MB.
def skew():
    out = {}
    base = {"spark__sql__autoBroadcastJoinThreshold": "-1"}
    for label, conf in [("aqe_defaults", base),
                        ("aqe_thresholds_lowered", {**base,
                            "spark__sql__adaptive__skewJoin__skewedPartitionThresholdInBytes": "4MB",
                            "spark__sql__adaptive__advisoryPartitionSizeInBytes": "4MB"})]:
        spark = session(f"skew-{label}", **conf)

        def build():
            return (events(spark).join(products(spark), "product_id")
                    .groupBy("category", "event_type").agg(F.sum("price").alias("rev")))
        m = measure(spark, label, lambda: build().collect())
        q = build()
        q.collect()
        m["skew_split_applied"] = "skew=true" in plan_text(q)
        out[label] = m
        spark.stop()
    results["skew"] = out


# 4. UDFs: native expression vs pandas UDF vs row-at-a-time Python UDF.
def udf():
    out = {}
    spark = session("udf")

    # Pull the numeric part out of a product id such as "p-1234".
    @F.pandas_udf(IntegerType())
    def num_pandas(s: pd.Series) -> pd.Series:
        return s.str.split("-").str[1].astype("int32")

    @F.udf(IntegerType())
    def num_python(s):
        return None if s is None else int(s.split("-")[1])

    variants = {
        "native": lambda c: F.regexp_extract(c, r"(\d+)$", 1).cast("int"),
        "pandas_udf": num_pandas,
        "python_udf": num_python,
    }
    for label, fn in variants.items():
        def build(fn=fn):
            return (events(spark).select("product_id")
                    .withColumn("bucket", fn(F.col("product_id")) % 10)
                    .groupBy("bucket").count())
        m = measure(spark, label, lambda: build().collect())
        m["plan_nodes"] = has(build(), ["BatchEvalPython", "ArrowEvalPython"])
        out[label] = m
    spark.stop()
    results["udf"] = out


# 5. Deduplication: hashing every column vs hashing the key.
def dedup():
    out = {}
    spark = session("dedup")

    def wide():
        ev = events(spark)
        return (ev.unionByName(ev.sample(0.1, seed=3))
                  .withColumn("user_agent", F.concat(F.lit(UA + " sid="), F.col("session_id")))
                  .withColumn("utm", F.struct(F.lit("newsletter").alias("source"), F.lit("email").alias("medium"),
                                              F.concat(F.lit("autumn-sale-"), F.col("event_type")).alias("campaign"))))
    for label, fn in [("all_columns", lambda: wide().dropDuplicates()),
                      ("key_only", lambda: wide().dropDuplicates(["event_id"]))]:
        out[label] = measure(spark, label, lambda fn=fn: fn().count())
    out["same_count"] = wide().dropDuplicates().count() == wide().dropDuplicates(["event_id"]).count()
    spark.stop()
    results["dedup"] = out


# 6. Partition pruning: filter the partition column, not an expression that hides it.
def pruning():
    out = {}
    spark = session("pruning")
    day = "2026-08-30"
    variants = {
        "expression_on_timestamp": lambda: events(spark, "events_by_dt")
            .filter(F.to_date(F.from_unixtime("event_ts")) == F.lit(day)).agg(F.sum("price")),
        "partition_column": lambda: events(spark, "events_by_dt")
            .filter(F.col("dt") == F.lit(day)).agg(F.sum("price")),
    }
    for label, fn in variants.items():
        m = measure(spark, label, lambda fn=fn: fn().collect())
        p = plan_text(fn())
        m["partition_filter_used"] = "PartitionFilters: [isnotnull(dt" in p
        out[label] = m
    out["same_answer"] = (round(variants["expression_on_timestamp"]().first()[0], 2)
                          == round(variants["partition_column"]().first()[0], 2))
    out["partitions_total"] = events(spark, "events_by_dt").select("dt").distinct().count()
    spark.stop()
    results["pruning"] = out


# 7. The one config win: AQE off (a fixed 200 shuffle partitions) vs on (the default).
def config():
    out = {}
    for label, conf in [("aqe_off_200_partitions", {"spark__sql__adaptive__enabled": "false"}),
                        ("aqe_on_default", {})]:
        spark = session(f"cfg-{label}", **conf)

        def build():
            return (events(spark).groupBy("product_id", "event_type")
                    .agg(F.countDistinct("session_id").alias("n")))
        out[label] = measure(spark, label, lambda: build().collect())
        spark.stop()
    results["config"] = out


ALL = {"lazy": lazy, "joins": joins, "or_join": or_join, "skew": skew,
       "udf": udf, "dedup": dedup, "pruning": pruning, "config": config}

if __name__ == "__main__":
    for name in (sys.argv[1:] or ALL):
        t = time.time()
        ALL[name]()
        save()
        print(f"{name} done in {time.time() - t:.0f}s", flush=True)
    spark = session("env")
    keys = ["spark.sql.shuffle.partitions", "spark.sql.adaptive.enabled",
            "spark.sql.autoBroadcastJoinThreshold",
            "spark.sql.adaptive.skewJoin.skewedPartitionFactor",
            "spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes",
            "spark.sql.adaptive.advisoryPartitionSizeInBytes"]
    results["env"] = {"spark": spark.version, "python": sys.version.split()[0], "master": "local[2]",
                      "driver_memory": "4g", "events": 5_000_000, "products": 2_000,
                      "defaults": {k: spark.conf.get(k) for k in keys}}
    spark.stop()
    save()
