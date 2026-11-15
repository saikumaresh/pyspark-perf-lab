"""Generate a skewed marketplace clickstream.

5 million events over 30 days. 80% of events belong to 3 "viral" products,
because uniform test data never shows you a skew problem.
Written twice: flat Parquet, and Parquet partitioned by event date (dt).
"""
from pyspark.sql import functions as F

from common import DATA, session

N_EVENTS = 5_000_000
N_PRODUCTS = 2_000

spark = session("gen-data")

ev = (spark.range(N_EVENTS).withColumnRenamed("id", "n")
      .withColumn("r", F.rand(7))
      .withColumn("product_id",
                  F.when(F.col("r") < 0.8, F.concat(F.lit("p-"), (F.floor(F.rand(11) * 3) + 1).cast("string")))
                   .otherwise(F.concat(F.lit("p-"), (F.floor(F.rand(13) * (N_PRODUCTS - 3)) + 4).cast("string"))))
      .withColumn("event_id", F.concat(F.lit("e-"), F.col("n").cast("string")))
      .withColumn("session_id", F.concat(F.lit("s-"), (F.floor(F.rand(17) * 400_000)).cast("string")))
      .withColumn("event_type", F.element_at(
          F.array(*[F.lit(x) for x in ["view", "add_to_cart", "checkout", "purchase", "search", "wishlist"]]),
          (F.floor(F.rand(19) * 6) + 1).cast("int")))
      .withColumn("event_ts", (F.lit(1_788_000_000) + F.floor(F.rand(23) * 2_592_000)).cast("long"))
      .withColumn("price", F.round(F.rand(29) * 200, 2))
      .withColumn("dt", F.to_date(F.from_unixtime("event_ts")))
      .drop("n", "r"))

ev.drop("dt").write.mode("overwrite").parquet(f"{DATA}/events")
ev.write.mode("overwrite").partitionBy("dt").parquet(f"{DATA}/events_by_dt")

products = (spark.range(1, N_PRODUCTS + 1)
            .withColumn("product_id", F.concat(F.lit("p-"), F.col("id").cast("string")))
            .withColumn("alt_sku", F.concat(F.lit("sku-"), F.col("id").cast("string")))
            .withColumn("category", F.element_at(
                F.array(*[F.lit(x) for x in ["beauty", "apparel", "home", "electronics", "grocery"]]),
                (F.col("id") % 5 + 1).cast("int")))
            .withColumn("brand", F.concat(F.lit("brand-"), (F.col("id") % 97).cast("string")))
            .drop("id"))
products.write.mode("overwrite").parquet(f"{DATA}/products")

print("events  :", spark.read.parquet(f"{DATA}/events").count())
print("products:", spark.read.parquet(f"{DATA}/products").count())
print("top products share:",
      spark.read.parquet(f"{DATA}/events").groupBy("product_id").count()
      .orderBy(F.desc("count")).limit(3).agg(F.sum("count")).first()[0] / N_EVENTS)
spark.stop()
