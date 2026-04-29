"""End-to-end pipeline driver.

Composes the stages defined in the sibling modules. Each stage is a pure
DataFrame -> DataFrame transform, which keeps the driver thin and makes
individual stages independently testable / replaceable.
"""

from __future__ import annotations

from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from pipeline.blocking import filter_block_sizes, generate_block_keys
from pipeline.clustering import cluster_blocks
from pipeline.config import PipelineConfig
from pipeline.features import extract_features
from pipeline.flatten import flatten, read_raw
from pipeline.merge import connected_components
from pipeline.normalize import normalize
from pipeline.quality import aggregate_clusters, filter_clusters


def _build_spark(app_name: str, cfg: PipelineConfig) -> SparkSession:
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", str(cfg.shuffle_partitions))
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        # Coalesce post-shuffle partitions so small intermediate stages
        # don't each pay full task-launch overhead. Critical on small
        # local runs where there are far more partitions than data.
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.minPartitionSize", "1MB")
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "16MB")
    )
    return builder.getOrCreate()


def _is_alive(session: SparkSession) -> bool:
    try:
        return not session.sparkContext._jsc.sc().isStopped()
    except Exception:
        return False


def run_pipeline(input_path: str, output_path: str,
                 cfg: Optional[PipelineConfig] = None,
                 spark: Optional[SparkSession] = None) -> DataFrame:
    """Execute the full pipeline and write final clusters to ``output_path``.

    Returns the final (already-persisted) clusters DataFrame so callers in
    notebooks or tests can inspect it without re-reading.
    """
    cfg = cfg or PipelineConfig()
    # If the caller passed a dead session (common in notebooks after a
    # previous run stopped the context), transparently rebuild one. We
    # only stop the session in `finally` if WE built it.
    caller_alive = spark is not None and _is_alive(spark)
    owns_spark = not caller_alive
    if not caller_alive:
        spark = _build_spark("syntactic-clustering", cfg)

    try:
        raw = read_raw(spark, input_path, cfg)
        cells = flatten(raw)
        cells = normalize(cells, cfg)
        cells = extract_features(cells, cfg)
        # Materialize the per-cell features once. Without an eager count
        # the feature UDF would re-execute every time the DataFrame is
        # branched (blocking, clustering, cohesion stats).
        cells = cells.persist()
        cells.count()

        keyed = generate_block_keys(cells, cfg)
        keyed = filter_block_sizes(keyed, cfg)

        edges = cluster_blocks(keyed, cfg).persist()
        edges.count()  # materialize before the iterative CC loop

        assignments = connected_components(edges, cfg)

        clusters = aggregate_clusters(assignments, cells, cfg)
        clusters = filter_clusters(clusters, cfg)

        writer = clusters.write.mode("overwrite")
        if cfg.output_format == "parquet":
            writer.parquet(output_path)
        elif cfg.output_format == "json":
            # JSON struggles with nested arrays of longs at scale, but is
            # convenient for small debug runs.
            writer.json(output_path)
        else:
            raise ValueError(f"Unsupported output_format: {cfg.output_format}")

        return spark.read.format(cfg.output_format).load(output_path)
    finally:
        if owns_spark:
            # Caller created Spark for us implicitly; tear it down so we
            # do not leak driver resources in a script context.
            spark.stop()
