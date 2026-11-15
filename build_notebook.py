"""Builds pyspark_basics_to_performance.ipynb: PySpark basics, then the article's experiments."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# PySpark from basics to fast: fix the query, not the cluster

Part 1 walks through the PySpark fundamentals: the session, reading data, columns, missing values, filters, aggregations, SQL, pivots, lazy evaluation, caching and a first model. Part 2 runs the article's experiments on 5 million synthetic clickstream events, each comparing a slow query shape with a fast one.

Companion notebook to the article *Spark Is "100x Faster." So Why Is Your PySpark Job Still Slow?*

**Before you start:** run `python lab/gen_data.py` once from the repo root to create the dataset (about 30 seconds).
""")
code("""
import os, sys, time
os.chdir("lab")            # data/ lives here
sys.path.insert(0, ".")
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
UA = "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Mobile Safari/537.36"
import pandas as pd
from common import session, plan_text

spark = session("notebook")
events = lambda: spark.read.parquet("data/events")
products = lambda: spark.read.parquet("data/products")

def timed(label, fn, runs=3):
    fn()                                   # warm-up
    t = []
    for _ in range(runs):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    t.sort(); print(f"{label:<28} {'median' if runs > 1 else '1 run '} {t[len(t)//2]:6.2f}s")

def show_nodes(df, names):
    p = plan_text(df)
    print("plan nodes:", [n for n in names if n in p] or "none of", names if not [n for n in names if n in p] else "")

print("Spark", spark.version, "| events:", events().count(), "| products:", products().count())
""")
import notebook_part1
notebook_part1.add(md, code)
md("""
## 0. Lazy evaluation: the slow line is not the slow line
Transformations only build a plan. The action at the end pays for everything. `explain()` shows the route before you drive it.
""")
code("""
t0 = time.perf_counter()
q = (events().filter(F.col("event_type") == "purchase")
       .withColumn("price_band", (F.col("price") / 50).cast("int"))
       .groupBy("product_id", "price_band").agg(F.sum("price").alias("revenue")))
print(f"building the plan: {time.perf_counter() - t0:.2f}s")
t0 = time.perf_counter(); q.count(); print(f"running it:        {time.perf_counter() - t0:.2f}s")
q.explain(mode="formatted")   # look for PushedFilters on the scan
""")
md("""
## 1. Let the small table travel: broadcast hash join vs sort merge join
""")
code("""
q = lambda: events().join(products(), "product_id").groupBy("category").agg(F.count("*").alias("n"))
show_nodes(q(), ["BroadcastHashJoin", "SortMergeJoin"]); timed("broadcast (default)", lambda: q().collect())
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
show_nodes(q(), ["BroadcastHashJoin", "SortMergeJoin"]); timed("forced sort merge", lambda: q().collect())
spark.conf.unset("spark.sql.autoBroadcastJoinThreshold")
""")
md("""
## 2. The OR that turned a join into a loop
An OR across two columns has no single equality key, so Spark falls back to a nested loop. Rewrite as one equi-join per key.
""")
code("""
def ev2():
    return events().limit(1_000_000).withColumn("alt_sku", F.regexp_replace("product_id", "^p-", "sku-"))

def slow():
    e, lk = ev2(), products()
    return e.join(lk, (e.product_id == lk.product_id) | (e.alt_sku == lk.alt_sku), "left").agg(F.countDistinct(lk.brand))

def fast():
    e, lk = ev2(), products()
    a = e.join(lk.select("product_id", F.col("brand").alias("b1")), "product_id", "left")
    return a.join(lk.select("alt_sku", F.col("brand").alias("b2")), "alt_sku", "left") \\
            .agg(F.countDistinct(F.coalesce("b1", "b2")))

names = ["BroadcastNestedLoopJoin", "BroadcastHashJoin"]
show_nodes(slow(), names); timed("OR condition", lambda: slow().collect())
show_nodes(fast(), names); timed("two equi-joins", lambda: fast().collect())
print("same answer:", slow().first()[0] == fast().first()[0])
""")
md("""
## 3. Python UDFs leave the engine
""")
code("""
# Pull the numeric part out of a product id such as "p-1234"
@F.pandas_udf(IntegerType())
def num_pandas(s: pd.Series) -> pd.Series:
    return s.str.split("-").str[1].astype("int32")

@F.udf(IntegerType())
def num_python(s):
    return None if s is None else int(s.split("-")[1])

native = lambda c: F.regexp_extract(c, r"(\\d+)$", 1).cast("int")
for label, fn in [("native", native), ("pandas_udf", num_pandas), ("python_udf", num_python)]:
    q = lambda fn=fn: events().select("product_id").withColumn("bucket", fn(F.col("product_id")) % 10).groupBy("bucket").count()
    show_nodes(q(), ["BatchEvalPython", "ArrowEvalPython"]); timed(label, lambda: q().collect())
""")
md("""
## 4. Deduplicate on the key, not the whole row
""")
code("""
def wide():
    ev = events()
    return (ev.unionByName(ev.sample(0.1, seed=3))
              .withColumn("user_agent", F.concat(F.lit(UA + " sid="), F.col("session_id")))
              .withColumn("utm", F.struct(F.lit("newsletter").alias("source"), F.lit("email").alias("medium"),
                                          F.concat(F.lit("autumn-sale-"), F.col("event_type")).alias("campaign"))))
timed("dropDuplicates() all columns", lambda: wide().dropDuplicates().count(), runs=1)  # 1 run: slow
timed("dropDuplicates(['event_id'])", lambda: wide().dropDuplicates(["event_id"]).count(), runs=1)
# Compare "Shuffle Write" for the two jobs in the Spark UI (Stages tab).
""")
md("""
## 5. Filter the partition, not a disguise of it
""")
code("""
day = "2026-08-30"
by_dt = lambda: spark.read.parquet("data/events_by_dt")
hidden = lambda: by_dt().filter(F.to_date(F.from_unixtime("event_ts")) == F.lit(day)).agg(F.sum("price"))
direct = lambda: by_dt().filter(F.col("dt") == F.lit(day)).agg(F.sum("price"))
for label, q in [("expression on timestamp", hidden), ("partition column", direct)]:
    print(label, "| partition filter used:", "PartitionFilters: [isnotnull(dt" in plan_text(q()))
    timed(label, lambda q=q: q().collect())
""")
md("""
## 6. The one configuration win: Adaptive Query Execution (on by default)
""")
code("""
q = lambda: events().groupBy("product_id", "event_type").agg(F.countDistinct("session_id").alias("n"))
spark.conf.set("spark.sql.adaptive.enabled", "false"); timed("AQE off, 200 partitions", lambda: q().collect(), runs=1)
spark.conf.set("spark.sql.adaptive.enabled", "true");  timed("AQE on (default)", lambda: q().collect(), runs=1)
""")
md("""
## 7. Skew, and the 256 MB threshold nobody reads
A partition only counts as skewed above 5x the median **and** above 256 MB. On this data, the automatic split never fires until you lower the threshold.
""")
code("""
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
q = lambda: events().join(products(), "product_id").groupBy("category", "event_type").agg(F.sum("price"))
d = q(); d.collect(); print("defaults        -> skew split applied:", "skew=true" in plan_text(d))
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "4MB")
spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "4MB")
d = q(); d.collect(); print("threshold 4 MB  -> skew split applied:", "skew=true" in plan_text(d))
for k in ["spark.sql.autoBroadcastJoinThreshold", "spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes",
          "spark.sql.adaptive.advisoryPartitionSizeInBytes"]:
    spark.conf.unset(k)
""")
md("""
## Read the plan, change the question
Before you change configuration, run `explain(mode="formatted")` and look for:

| Plan node | What it means | Fix |
|---|---|---|
| `BroadcastNestedLoopJoin`, `CartesianProduct` | Join with no usable equality key | Rewrite as equi-joins |
| `SortMergeJoin` against a small table | Size estimate is wrong | Check statistics, measure, then hint `broadcast()` |
| `BatchEvalPython`, `ArrowEvalPython` | Python in the hot path | Native expression first |
| More `Exchange` nodes than expected | Extra shuffles and stages | Ask whether each one is needed |
| No `PartitionFilters` / `PushedFilters` | Reading data you throw away | Filter the raw partition column |
""")
code("spark.stop()")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "pyspark_basics_to_performance.ipynb")
print("written", len(C), "cells")
