"""Pair-based ER metrics and size-distribution diagnostics.

Spark-native equivalent of ``er_metrics.py`` from the companion
``blockingandembeddinginer`` repo. Computes precision / recall / F1 at
the PAIR level so the result is directly comparable to algo1_2_v2's
output, plus the predicted/truth cluster-size distributions.

Truth file format (matches DWM ``truthABC*.txt`` convention):

    header_line
    refID,truthID
    refID,truthID
    ...

Metrics are written as ``metrics.json`` and ``metrics.log`` under
``<output_path>/_metrics/`` so they coexist with the parquet output
without confusing Athena's table scanner.
"""

from __future__ import annotations

import json
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


_TRUTH_SCHEMA = T.StructType([
    T.StructField("ref_id", T.StringType(), nullable=False),
    T.StructField("truth_id", T.StringType(), nullable=False),
])


def load_truth(spark: SparkSession, truth_file_path: str) -> DataFrame:
    """Load truth CSV (header + ``refID,truthID`` rows)."""
    return (
        spark.read.option("header", "true")
        .option("sep", ",")
        .schema(_TRUTH_SCHEMA)
        .csv(truth_file_path)
    )


def compute_pair_metrics(assignments: DataFrame,
                         ref_id_table: DataFrame,
                         truth: DataFrame) -> dict:
    """Pair-counting precision / recall / F1, computed entirely in Spark.

    For predicted clusters of size c and truth clusters of size t::

        L  = sum c*(c-1)/2 over predicted clusters
        E  = sum t*(t-1)/2 over truth clusters
        TP = sum k*(k-1)/2 over (cluster_id, truth_id) intersections
             of size k

    No O(n^2) pair enumeration anywhere; everything is groupBy + sum.
    """
    pred = (
        assignments.select("record_id", "cluster_id")
        .join(ref_id_table, on="record_id", how="inner")
        .select("ref_id", "cluster_id")
    )

    pred_sizes = pred.groupBy("cluster_id").agg(F.count("*").alias("c"))
    L_row = pred_sizes.select(
        F.sum(F.col("c") * (F.col("c") - F.lit(1)) / F.lit(2)).alias("L")
    ).collect()
    L = int((L_row[0]["L"] or 0))

    truth_sizes = truth.groupBy("truth_id").agg(F.count("*").alias("t"))
    E_row = truth_sizes.select(
        F.sum(F.col("t") * (F.col("t") - F.lit(1)) / F.lit(2)).alias("E")
    ).collect()
    E = int((E_row[0]["E"] or 0))

    inter = pred.join(truth, on="ref_id", how="inner")
    inter_sizes = (
        inter.groupBy("cluster_id", "truth_id")
        .agg(F.count("*").alias("k"))
    )
    TP_row = inter_sizes.select(
        F.sum(F.col("k") * (F.col("k") - F.lit(1)) / F.lit(2)).alias("TP")
    ).collect()
    TP = int((TP_row[0]["TP"] or 0))

    FP = L - TP
    FN = E - TP
    precision = TP / L if L > 0 else 1.0
    recall = TP / E if E > 0 else 1.0
    denom = precision + recall
    f1 = 2 * precision * recall / denom if denom > 0 else 0.0

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "L": L,
        "E": E,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compute_predicted_distribution(clusters: DataFrame) -> dict:
    """Cluster-size distribution from the post-filter clusters DataFrame."""
    rows = (
        clusters.groupBy("size").agg(F.count("*").alias("n"))
        .orderBy(F.col("size").asc())
        .collect()
    )
    return {str(int(r["size"])): int(r["n"]) for r in rows}


def compute_truth_distribution(truth: DataFrame) -> dict:
    sizes = truth.groupBy("truth_id").agg(F.count("*").alias("size"))
    rows = (
        sizes.groupBy("size").agg(F.count("*").alias("n"))
        .orderBy(F.col("size").asc())
        .collect()
    )
    return {str(int(r["size"])): int(r["n"]) for r in rows}


def _format_metrics_log(metrics: dict) -> str:
    lines = ["=" * 72, "  RUN METRICS", "=" * 72]

    cfg = metrics.get("config", {})
    if cfg:
        lines.append("\n[Configuration]")
        for k, v in cfg.items():
            lines.append(f"  {k:<26} = {v}")

    pc = metrics.get("predicted_clusters", {})
    if pc:
        lines.append("\n[Predicted Clusters]")
        lines.append(f"  total                      = {pc.get('n_total', '?')}")
        if "size_distribution" in pc:
            lines.append("  size distribution (size: count):")
            for s, n in sorted(pc["size_distribution"].items(), key=lambda x: int(x[0])):
                lines.append(f"    size {int(s):>3} : {n}")

    if "truth_clusters" in metrics:
        tc = metrics["truth_clusters"]
        lines.append(f"\n[Truth Clusters]  ({tc.get('truth_file', '?')})")
        lines.append(f"  total                      = {tc.get('n_total', '?')}")
        if "size_distribution" in tc:
            lines.append("  size distribution (size: count):")
            for s, n in sorted(tc["size_distribution"].items(), key=lambda x: int(x[0])):
                lines.append(f"    size {int(s):>3} : {n}")

    if "pair_metrics" in metrics:
        p = metrics["pair_metrics"]
        lines.append("\n[Pair-based Metrics]")
        lines.append(f"  TP                         = {p['TP']}")
        lines.append(f"  FP                         = {p['FP']}")
        lines.append(f"  FN                         = {p['FN']}")
        lines.append(f"  Linked Pairs (L)           = {p['L']}")
        lines.append(f"  Expected Pairs (E)         = {p['E']}")
        lines.append(f"  Precision                  = {p['precision']}")
        lines.append(f"  Recall                     = {p['recall']}")
        lines.append(f"  F1                         = {p['f1']}")

    if "timings_seconds" in metrics:
        lines.append("\n[Timings (seconds)]")
        for k, v in metrics["timings_seconds"].items():
            lines.append(f"  {k:<26} = {v}")

    lines.append("\n" + "=" * 72)
    return "\n".join(lines) + "\n"


def _hadoop_write_text(spark: SparkSession, path: str, text: str) -> None:
    """Write a single string to ``path`` (local or s3://) via Hadoop FS.

    Avoids the part-00000-folder pattern Spark would produce with
    ``coalesce(1).write.text``; we want a real named file at the path.
    """
    sc = spark.sparkContext
    URI = sc._gateway.jvm.java.net.URI
    HPath = sc._gateway.jvm.org.apache.hadoop.fs.Path
    FileSystem = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem

    fs = FileSystem.get(URI(path), sc._jsc.hadoopConfiguration())
    out = fs.create(HPath(path), True)  # overwrite=True
    try:
        out.write(text.encode("utf-8"))
    finally:
        out.close()


def write_metrics_files(spark: SparkSession, metrics: dict,
                        output_path: str) -> str:
    """Write metrics.json + metrics.log under ``<output_path>/_metrics/``.

    Returns the directory path written to.
    """
    base = output_path.rstrip("/") + "/_metrics"
    json_text = json.dumps(metrics, ensure_ascii=False, indent=2)
    log_text = _format_metrics_log(metrics)
    _hadoop_write_text(spark, base + "/metrics.json", json_text)
    _hadoop_write_text(spark, base + "/metrics.log", log_text)
    return base


def run_metrics(spark: SparkSession,
                assignments: DataFrame,
                clusters: DataFrame,
                ref_id_table: Optional[DataFrame],
                truth_file: Optional[str],
                output_path: str,
                config: Optional[dict] = None,
                timings: Optional[dict] = None) -> dict:
    """Compute metrics and write metrics.{json,log}. Returns the dict.

    Pair-based P/R/F1 are computed only when both ``truth_file`` and
    ``ref_id_table`` are supplied. The predicted-cluster size
    distribution is always emitted.
    """
    pc_dist = compute_predicted_distribution(clusters)
    n_clusters = clusters.count()

    metrics: dict = {
        "config": config or {},
        "predicted_clusters": {
            "n_total": n_clusters,
            "size_distribution": pc_dist,
        },
    }

    if truth_file and ref_id_table is not None:
        truth = load_truth(spark, truth_file).cache()
        try:
            pair = compute_pair_metrics(assignments, ref_id_table, truth)
            truth_dist = compute_truth_distribution(truth)
            n_truth_clusters = truth.select("truth_id").distinct().count()
            metrics["truth_clusters"] = {
                "truth_file": truth_file,
                "n_total": n_truth_clusters,
                "size_distribution": truth_dist,
            }
            metrics["pair_metrics"] = pair
        finally:
            truth.unpersist()

    if timings:
        metrics["timings_seconds"] = {k: round(v, 3) for k, v in timings.items()}

    out_dir = write_metrics_files(spark, metrics, output_path)
    metrics["_written_to"] = out_dir
    return metrics
