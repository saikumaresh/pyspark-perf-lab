"""Part 1 cells for the notebook: PySpark fundamentals in the order you use them."""


def add(md, code):
    md('''
# Part 1. The fundamentals, in the order you will use them

Why Spark exists, in one paragraph: a single machine reads data from one disk with a few CPU cores. Hadoop MapReduce split work across many machines but wrote every intermediate result back to disk. Spark keeps intermediate data in memory where it can, and plans the whole job before running it. That is where the famous "up to 100x faster" line came from: iterative work such as machine learning, which rereads the same data many times.

How a Spark application is organised:
- **Driver**: your Python program plus the `SparkSession`. It builds the plan and hands out tasks.
- **Executors**: worker processes that run the tasks and hold cached data.
- **Cluster manager**: YARN, Kubernetes or standalone. It gives the driver its executors.

Here, `local[2]` runs the driver and executors as threads in one process, with 2 cores. The code is identical on a real cluster.
''')
    md('''
## 1.1 The SparkSession is your entry point
`SparkSession.builder...getOrCreate()` returns the running session if one exists, otherwise starts one. The helper in `lab/common.py` does exactly that.
''')
    code('''
spark
''')
    md('''
## 1.2 Reading data: let Spark guess, or tell it the schema
Reading a CSV without a schema makes every column a string. `inferSchema` guesses types but needs an extra pass over the data. On real pipelines, declare the schema.
''')
    code('''
# Write a small CSV sample so we can practise CSV options
(events().limit(50_000).coalesce(1).write.mode("overwrite")
   .option("header", True).csv("data/events_sample_csv"))

raw = spark.read.option("header", True).csv("data/events_sample_csv")
raw.printSchema()          # everything is a string

schema = ("product_id STRING, event_id STRING, session_id STRING, event_type STRING, "
          "event_ts LONG, price DOUBLE")
typed = spark.read.option("header", True).schema(schema).csv("data/events_sample_csv")
typed.printSchema()        # real types, no extra pass
''')
    md('''
## 1.3 Looking at a DataFrame
`show`, `count`, `select` and `describe`. Note that `count` and `show` are **actions**: each one runs a job.
''')
    code('''
print("rows:", typed.count())
typed.select("event_type", "price").show(5)
typed.describe("price").show()
print(typed.dtypes)
''')
    md('''
## 1.4 Changing columns
Every transformation returns a **new** DataFrame. Nothing is changed in place.
''')
    code('''
df = (typed
      .withColumn("event_time", F.from_unixtime("event_ts").cast("timestamp"))
      .withColumn("price_int", F.col("price").cast("int"))
      .withColumnRenamed("event_type", "action")
      .drop("session_id"))
df.show(3, truncate=False)
''')
    md('''
## 1.5 Missing values
Drop rows with nulls, or fill them. Shown on a tiny DataFrame so the effect is visible.
''')
    code('''
messy = spark.createDataFrame(
    [("p-1", 10.0, "view"), ("p-2", None, "purchase"), (None, 5.0, None)],
    "product_id STRING, price DOUBLE, event_type STRING")
messy.na.drop().show()                                   # any null drops the row
messy.na.drop(subset=["price"]).show()                   # only look at price
messy.na.fill({"price": 0.0, "event_type": "unknown"}).show()
''')
    md('''
## 1.6 Filtering
`filter` and `where` are the same. Combine conditions with `&` (and), `|` (or) and `~` (not), each wrapped in brackets.
''')
    code('''
df.filter((F.col("action") == "purchase") & (F.col("price") > 150)).show(3)
df.where(~F.col("action").isin("view", "search")).groupBy("action").count().show()
print(df.filter("price < 5 AND action = 'checkout'").count())   # a SQL string works too
''')
    md('''
## 1.7 Grouping, aggregating and sorting
''')
    code('''
(df.groupBy("action")
   .agg(F.count("*").alias("events"),
        F.round(F.avg("price"), 2).alias("avg_price"),
        F.countDistinct("product_id").alias("products"))
   .orderBy(F.desc("events"))
   .show())
''')
    md('''
## 1.8 SQL on a DataFrame
Register a temporary view and query it. Same engine, same plan, different syntax.
''')
    code('''
df.createOrReplaceTempView("events_sample")
spark.sql("""
    SELECT action, ROUND(SUM(price), 2) AS revenue
    FROM events_sample
    GROUP BY action
    ORDER BY revenue DESC
""").show()
''')
    md('''
## 1.9 Pivot
Pass the pivot values explicitly. Otherwise Spark runs an extra job just to find them.
''')
    code('''
(df.withColumn("band", F.when(F.col("price") < 50, "low").otherwise("high"))
   .groupBy("action").pivot("band", ["low", "high"]).count()
   .show())
''')
    md('''
## 1.10 Transformations, actions and lazy evaluation
- **Transformations** (`select`, `filter`, `withColumn`, `groupBy`, `join`) only add steps to a plan.
- **Actions** (`count`, `show`, `collect`, `write`) run the plan.

Spark builds the full plan first, optimises it, then runs it. Part 2 starts by measuring this.
''')
    md('''
## 1.11 Caching, collect and toPandas: use with care
Each action recomputes the plan from the source. `cache()` keeps the result in memory for reuse, and it only fills on the next action. `collect()` and `toPandas()` pull **everything** to the driver, so use them only on small results.
''')
    code('''
hot = events().filter(F.col("event_type") == "purchase").cache()
t0 = time.perf_counter(); hot.count(); print(f"first count (reads and caches): {time.perf_counter()-t0:.2f}s")
t0 = time.perf_counter(); hot.count(); print(f"second count (from cache):      {time.perf_counter()-t0:.2f}s")
print(hot.groupBy("event_type").count().toPandas())   # fine: the result is one row
hot.unpersist()
''')
    md('''
## 1.12 A machine learning model in a few lines
Spark ML wants all features in **one vector column**, and `VectorAssembler` builds it. The data is synthetic and random, so the model learns nothing useful. The point is the mechanics.
''')
    code('''
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

ml = (typed.withColumn("label", (F.col("event_type") == "purchase").cast("double"))
           .withColumn("hour", F.hour(F.from_unixtime("event_ts")).cast("double")))
features = VectorAssembler(inputCols=["price", "hour"], outputCol="features").transform(ml)
train, test = features.randomSplit([0.8, 0.2], seed=42)
model = LogisticRegression(featuresCol="features", labelCol="label").fit(train)
print("coefficients:", model.coefficients)
model.transform(test).select("price", "hour", "label", "prediction").show(3)
''')
    md('''
# Part 2. Why your PySpark job is still slow

The article's experiments, in the same order. Each pairs a slow query shape with a fast one, times both, and shows the plan node that gives the slow shape away.
''')
