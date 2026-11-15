# pyspark-perf-lab

Seven small, reproducible experiments that show why most slow PySpark jobs are fixed by changing the query, not the cluster. Companion code for the article *Spark Is "100x Faster." So Why Is Your PySpark Job Still Slow?* by [Sai Kumaresh](https://medium.com/@saikumaresh).

## What is inside

| Path | What it is |
|---|---|
| `pyspark_basics_to_performance.ipynb` | Start here. Part 1 covers PySpark fundamentals (session, reading data, columns, nulls, filters, aggregations, SQL, pivot, lazy evaluation, caching, a first model). Part 2 runs every experiment from the article, with outputs. |
| `lab/gen_data.py` | Generates 5 million synthetic marketplace clickstream events. 3 of 2,000 products own 80% of events, on purpose: uniform test data never shows you a skew problem. |
| `lab/experiments.py` | Runs all experiments: median of 3 timed runs after one warm-up, metrics from Spark's monitoring API. |
| `lab/common.py` | SparkSession helper and the metrics reader. |
| `results/results.json` | The raw numbers behind the article. |

## Results (PySpark 4.0.1, local[2], 4 GB driver)

| Experiment | Slow shape | Fast shape | Speedup |
|---|---|---|---|
| Join to a small table | Sort merge join, 2.80 s | Broadcast hash join, 1.33 s | 2.1x |
| OR in a join condition | Nested loop join, 18.76 s | Two equi-joins, 1.72 s | 10.9x |
| Custom logic | Python UDF, 5.73 s | Native expression, 1.03 s | 5.6x |
| Deduplication | All columns, 15.15 s, 323 MB shuffled | Key only, 6.33 s, 40 MB shuffled | 2.4x |
| Date filter | Expression on timestamp, 1.50 s | Partition column, 0.16 s | 9.4x |
| Shuffle partitions | AQE off, 16.77 s | AQE on (default), 6.46 s | 2.6x |
| Skew handling | Not triggered at the default 256 MB threshold | Triggered at 4 MB | No change on 2 cores |

Every figure is the median of 3 timed runs after one warm-up run, from a single run of `experiments.py`.

Absolute times will differ on your machine. The direction and the plan nodes should not.

## Run it

Needs Java 17 or 21 and Python 3.10 to 3.12.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd lab && python gen_data.py && python experiments.py && cd ..
jupyter notebook pyspark_basics_to_performance.ipynb
```

`build_notebook.py` regenerates the notebook from source.

## Notes

- PySpark 4.0.1 is used because Spark 3.5 does not officially support Java 21, and pandas UDFs failed in that setup. The defaults these experiments depend on (200 shuffle partitions, AQE on, 10 MB broadcast threshold, 256 MB skew threshold) are the same in Spark 3.5.
- When benchmarking, rebuild the DataFrame for every run. Re-running the same DataFrame object lets Spark skip stages whose shuffle output already exists.
- All data is synthetic.

MIT licence.
