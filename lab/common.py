"""Shared helpers: a local SparkSession and a tiny metrics reader.

The metrics come from Spark's own monitoring REST API (the same numbers the
Spark UI shows), so every figure this lab reports can be checked in the UI.
"""
import json
import statistics
import time
import urllib.request

from pyspark.sql import SparkSession

DATA = "data"


def session(app_name, cores=2, **conf):
    b = (SparkSession.builder.appName(app_name)
         .master(f"local[{cores}]")
         .config("spark.driver.memory", "4g")
         .config("spark.ui.showConsoleProgress", "false")
         .config("spark.sql.session.timeZone", "UTC")
         # Arrow (pandas UDFs) needs these on Java 17+.
         .config("spark.driver.extraJavaOptions",
                 "--add-opens=java.base/java.nio=ALL-UNNAMED -Dio.netty.tryReflectionSetAccessible=true"))
    for k, v in conf.items():
        b = b.config(k.replace("__", "."), v)
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def _api(spark, path):
    base = spark.sparkContext.uiWebUrl
    app = spark.sparkContext.applicationId
    with urllib.request.urlopen(f"{base}/api/v1/applications/{app}/{path}") as r:
        return json.load(r)


def plan_text(df):
    return df._jdf.queryExecution().executedPlan().toString()


def measure(spark, label, action, repeats=3, warmup=True):
    """Run `action` once to warm up, then `repeats` times. `action` must build
    a fresh DataFrame each call, otherwise Spark reuses shuffle output from the
    previous run and skips stages. Reports median wall time plus the shuffle,
    input and task-skew metrics of the LAST run, read from the REST API."""
    if warmup:
        action()
    times = []
    for _ in range(repeats):
        group = f"{label}-{time.time_ns()}"
        spark.sparkContext.setJobGroup(group, label)
        t0 = time.perf_counter()
        action()
        times.append(time.perf_counter() - t0)
    jobs = [j for j in _api(spark, "jobs") if j.get("jobGroup") == group]
    stage_ids = sorted({s for j in jobs for s in j["stageIds"]})
    shuffle_w = input_b = 0
    worst_ratio, worst = 0.0, None
    for sid in stage_ids:
        for st in _api(spark, f"stages/{sid}"):
            if st["status"] != "COMPLETE":
                continue
            shuffle_w += st.get("shuffleWriteBytes", 0)
            input_b += st.get("inputBytes", 0)
            tasks = _api(spark, f"stages/{sid}/{st['attemptId']}/taskList?length=100000")
            durs = [t["duration"] for t in tasks if t.get("duration") is not None]
            if len(durs) >= 4:
                med = statistics.median(durs)
                ratio = max(durs) / med if med > 0 else 0
                if ratio > worst_ratio:
                    worst_ratio = ratio
                    worst = {"stage": sid, "tasks": len(durs), "max_ms": max(durs),
                             "median_ms": med}
    return {
        "label": label,
        "median_s": round(statistics.median(times), 2),
        "runs_s": [round(t, 2) for t in times],
        "shuffle_write_mb": round(shuffle_w / 1e6, 1),
        "input_mb": round(input_b / 1e6, 1),
        "skew": worst,
    }
