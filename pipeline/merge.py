"""Distributed connected components for cross-block cluster merging.

A single record_id can appear in many blocks, and DBSCAN inside each
block emits its own edges. The same physical entity is therefore likely
to be linked by an edge chain that crosses block boundaries. We treat
all those edges as an undirected graph on record ids and run a
distributed connected-components algorithm over Spark.

Implementation: the *small-star / large-star* algorithm of Kiveris et al.
(SIGMOD 2014). It converges in O(log n) MapReduce rounds, requires no
external library (no GraphFrames dependency), and uses only joins +
groupBy operations that Spark already optimizes well.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from pipeline.config import PipelineConfig


def _symmetrize(edges: DataFrame) -> DataFrame:
    return (
        edges.select("src", "dst")
        .unionByName(edges.select(F.col("dst").alias("src"), F.col("src").alias("dst")))
        .dropDuplicates(["src", "dst"])
    )


def _large_star(edges: DataFrame) -> DataFrame:
    """For every node u, attach all *larger* neighbors to min(neighbors ∪ {u})."""
    sym = _symmetrize(edges)
    mins = sym.groupBy("src").agg(F.min("dst").alias("m"))
    joined = sym.join(mins, on="src", how="inner")
    larger = joined.where(F.col("dst") >= F.col("src"))
    return (
        larger.select(
            F.col("dst").alias("src"),
            F.col("m").alias("dst"),
        )
        .where(F.col("src") != F.col("dst"))
        .dropDuplicates(["src", "dst"])
    )


def _small_star(edges: DataFrame) -> DataFrame:
    """For every node u, attach all neighbors <= u to min(neighbors ≤ u)."""
    sym = _symmetrize(edges)
    smaller = sym.where(F.col("dst") <= F.col("src"))
    mins = smaller.groupBy("src").agg(F.min("dst").alias("m"))
    joined = smaller.join(mins, on="src", how="inner")
    return (
        joined.select(
            F.col("dst").alias("src"),
            F.col("m").alias("dst"),
        )
        .where(F.col("src") != F.col("dst"))
        .dropDuplicates(["src", "dst"])
    )


def _edges_signature(edges: DataFrame) -> int:
    """Cheap convergence check: edge count is monotone-non-increasing."""
    return edges.count()


def connected_components(edges: DataFrame, cfg: PipelineConfig) -> DataFrame:
    """Run small-star/large-star until the edge set is stable.

    Returns a DataFrame ``(record_id, cluster_id)`` where ``cluster_id``
    is the smallest record_id reachable from ``record_id``.
    """
    e = edges.select(F.col("src").cast("long"), F.col("dst").cast("long"))
    e = e.where(F.col("src") != F.col("dst")).dropDuplicates(["src", "dst"])

    if cfg.checkpoint_dir:
        e.sparkSession.sparkContext.setCheckpointDir(cfg.checkpoint_dir)

    prev_count = -1
    for i in range(cfg.cc_max_iterations):
        e = _large_star(e)
        e = _small_star(e)
        if cfg.checkpoint_dir and (i % 3 == 2):
            e = e.checkpoint(eager=True)
        cnt = _edges_signature(e)
        if cnt == prev_count:
            break
        prev_count = cnt

    # After convergence each surviving edge is (node -> root). We also need
    # a self-row for nodes that started as isolated singletons; those were
    # already injected upstream as (id, id) edges so they appear here too.
    return (
        e.select(
            F.col("src").alias("record_id"),
            F.col("dst").alias("cluster_id"),
        )
        .unionByName(
            e.select(
                F.col("dst").alias("record_id"),
                F.col("dst").alias("cluster_id"),
            )
        )
        .groupBy("record_id")
        .agg(F.min("cluster_id").alias("cluster_id"))
    )
