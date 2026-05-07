"""End-to-end pipeline driver.

Composes the stages defined in the sibling modules. Each stage is a pure
DataFrame -> DataFrame transform, which keeps the driver thin and makes
individual stages independently testable / replaceable.
"""

from __future__ import annotations

import time
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from pipeline.blocking import filter_block_sizes, generate_block_keys
from pipeline.clustering import cluster_blocks
from pipeline.clustering_arcs import cluster_blocks_arcs
from pipeline.config import PipelineConfig
from pipeline.features import extract_features
from pipeline.flatten import extract_ref_id_table, flatten, read_raw
from pipeline.merge import connected_components
from pipeline.metrics import run_metrics
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


def _run_clustering(keyed: DataFrame, cfg: PipelineConfig) -> DataFrame:
    method = cfg.clustering_method.lower()
    if method == "arcs":
        return cluster_blocks_arcs(keyed, cfg)
    if method == "dbscan":
        return cluster_blocks(keyed, cfg)
    raise ValueError(
        f"Unknown clustering_method: {cfg.clustering_method!r} "
        f"(expected 'dbscan' or 'arcs')"
    )


def _metrics_config(cfg: PipelineConfig) -> dict:
    """Subset of cfg surfaced in metrics.{json,log} for run identification."""
    return {
        "clustering_method": cfg.clustering_method,
        "input_format": cfg.input_format,
        "min_block_size": cfg.min_block_size,
        "max_block_size": cfg.max_block_size,
        "min_cluster_size": cfg.min_cluster_size,
        "dbscan_eps": cfg.dbscan_eps,
        "dbscan_min_samples": cfg.dbscan_min_samples,
        "arcs_weighting": cfg.arcs_weighting,
        "arcs_tau": cfg.arcs_tau,
        "arcs_top_k": cfg.arcs_top_k,
        "arcs_max_recursion_depth": cfg.arcs_max_recursion_depth,
        "arcs_do_merge": cfg.arcs_do_merge,
        "arcs_do_purge": cfg.arcs_do_purge,
    }


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

    timings: dict = {}

    try:
        raw = read_raw(spark, input_path, cfg)

        # Capture (record_id -> ref_id) BEFORE flatten/normalize so the
        # same monotonic record_ids are shared with the metrics path.
        # Built only when a truth file will actually be consumed; the
        # action is deferred to the metrics call so we don't pay it for
        # runs without truth.
        ref_id_table = None
        if cfg.metrics_enabled and cfg.truth_file:
            ref_id_table = extract_ref_id_table(
                raw, cfg, column_index=cfg.ref_id_column,
            ).persist()
            ref_id_table.count()

        cells = flatten(raw, cfg)
        cells = normalize(cells, cfg)
        cells = extract_features(cells, cfg)
        # Materialize the per-cell features once. Without an eager count
        # the feature UDF would re-execute every time the DataFrame is
        # branched (blocking, clustering, cohesion stats).
        cells = cells.persist()
        cells.count()

        keyed = generate_block_keys(cells, cfg)
        keyed = filter_block_sizes(keyed, cfg)

        t0 = time.time()
        edges = _run_clustering(keyed, cfg).persist()
        edges.count()  # materialize before the iterative CC loop
        timings["cluster"] = time.time() - t0

        t0 = time.time()
        assignments = connected_components(edges, cfg).persist()
        assignments.count()
        timings["connected_components"] = time.time() - t0

        clusters = aggregate_clusters(assignments, cells, cfg)
        clusters = filter_clusters(clusters, cfg).persist()
        clusters.count()

        writer = clusters.write.mode("overwrite")
        if cfg.output_format == "parquet":
            writer.parquet(output_path)
        elif cfg.output_format == "json":
            # JSON struggles with nested arrays of longs at scale, but is
            # convenient for small debug runs.
            writer.json(output_path)
        else:
            raise ValueError(f"Unsupported output_format: {cfg.output_format}")

        # Metrics are written AFTER the cluster output so Spark's
        # write.mode("overwrite") doesn't wipe them. They land under
        # <output_path>/_metrics/ so Athena's table scanner ignores them.
        if cfg.metrics_enabled:
            run_metrics(
                spark,
                assignments=assignments,
                clusters=clusters,
                ref_id_table=ref_id_table,
                truth_file=cfg.truth_file or None,
                output_path=output_path,
                config=_metrics_config(cfg),
                timings=timings,
            )

        return spark.read.format(cfg.output_format).load(output_path)
    finally:
        if owns_spark:
            # Caller created Spark for us implicitly; tear it down so we
            # do not leak driver resources in a script context.
            spark.stop()
