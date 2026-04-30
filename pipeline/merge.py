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


def _truncate_lineage(df: DataFrame, cfg: PipelineConfig):
    """Cut the logical-plan tree so the driver doesn't accumulate it.

    Without this the CC loop's plan grows on every iteration. After 5-10
    rounds the plan is large enough that Spark's optimizer + serializer
    OOMs the driver before any executor work begins. Two strategies:

    * If a checkpoint_dir is set, do a real checkpoint (S3 round-trip,
      slowest but bullet-proof at scale).
    * Otherwise localCheckpoint to executor memory+disk: cuts the plan
      without leaving the cluster.
    """
    if cfg.checkpoint_dir:
        return df.checkpoint(eager=True)
    return df.localCheckpoint(eager=True)


def _local_union_find(edges: DataFrame) -> DataFrame:
    """Collect edges to the driver and run union-find in pure Python.

    For edge counts up to ~10-50M this is dramatically faster than the
    distributed loop because it pays one shuffle and zero per-iteration
    framework overhead. Above that range it OOMs the driver.
    """
    pairs = edges.select("src", "dst").toPandas()
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # Always orient toward the smaller id so cluster_id == min member.
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for s, d in pairs.itertuples(index=False):
        union(int(s), int(d))

    rows = [(node, find(node)) for node in parent]
    spark = edges.sparkSession
    return spark.createDataFrame(rows, ["record_id", "cluster_id"])


def connected_components(edges: DataFrame, cfg: PipelineConfig) -> DataFrame:
    """Pick a merge strategy and run it.

    Returns a DataFrame ``(record_id, cluster_id)`` where ``cluster_id``
    is the smallest record_id reachable from ``record_id``.
    """
    strategy = cfg.merge_strategy.lower()
    if strategy == "local":
        return _local_union_find(edges)
    if strategy == "auto":
        n = edges.count()
        if n <= cfg.merge_local_edge_limit:
            return _local_union_find(edges)
        # else fall through to distributed.
    return _connected_components_starstar(edges, cfg)


def _connected_components_starstar(edges: DataFrame, cfg: PipelineConfig) -> DataFrame:
    """Run small-star/large-star until the edge set is stable."""
    if cfg.checkpoint_dir:
        edges.sparkSession.sparkContext.setCheckpointDir(cfg.checkpoint_dir)

    e = (
        edges.select(F.col("src").cast("long"), F.col("dst").cast("long"))
        .where(F.col("src") != F.col("dst"))
        .dropDuplicates(["src", "dst"])
    )
    # Materialize once: the loop below reads `e` repeatedly, and without a
    # cache every iteration would replay the entire upstream lineage
    # (feature UDF, blocking, per-block DBSCAN). That single missing cache
    # is what turns a 30 s job into a 30 min job on small inputs.
    e = _truncate_lineage(e, cfg)
    prev_count = e.count()

    for i in range(cfg.cc_max_iterations):
        new_e = _small_star(_large_star(e))
        # Truncate lineage EVERY iteration. The previous policy of every
        # third was not enough at scale: by round 5 the in-memory plan
        # tree was big enough to OOM the driver during query optimization
        # (visible in logs as a giant *(N) WholeStageCodegen chain ending
        # in a driver OutOfMemory before any executor work happened).
        new_e = _truncate_lineage(new_e, cfg)
        cnt = new_e.count()
        e.unpersist()
        e = new_e
        if cnt == prev_count:
            break
        prev_count = cnt

    # After convergence each surviving edge is (node -> root). We also need
    # a self-row for nodes that started as isolated singletons; those were
    # already injected upstream as (id, id) edges so they appear here too.
    out = (
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
    return out
