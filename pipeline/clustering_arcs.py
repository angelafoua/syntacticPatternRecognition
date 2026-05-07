"""ARCS-style clustering: build a weighted graph from block redundancy.

This is an alternative to ``cluster_blocks`` (DBSCAN-via-applyInPandas).
It mirrors the post-blocking pipeline of ``recursive_algo1_2_v2.py`` from
the companion ``blockingandembeddinginer`` repo (Algo1_2_v2):

    initial_blocks
      -> recursive_refine     (refine_blocks ∪ merge_blocks ∪ purge_subset_blocks)
      -> filter_top_k_smallest
      -> build_arcs_graph     (1/|B| or log(N/|B|)/|B| per pair, summed)
      -> threshold edges by tau
      -> connected components (handed off to pipeline.merge)

Output schema matches ``cluster_blocks``: ``(src long, dst long)`` plus
self-edges per record so isolated records flow through CC and emerge as
singleton clusters.

The density-floor cluster splitting from ``_split_cluster_by_density``
is not implemented in v1; setting ``cfg.arcs_density_floor > 0`` raises.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

from pipeline.config import PipelineConfig


def _initial_blocks(keyed: DataFrame) -> DataFrame:
    """Project the keyed cell stream into ARCS' minimal block form.

    Schema: ``(block_id: string, used_tokens: array<string>, record_id: long)``.
    ``used_tokens`` records which blocking keys are already encoded in the
    block_id so refinement avoids re-using them.
    """
    return (
        keyed.select(
            F.col("block_key").alias("block_id"),
            F.array(F.col("block_key")).alias("used_tokens"),
            F.col("record_id"),
        )
        .dropDuplicates(["block_id", "record_id"])
    )


def _record_keys(keyed: DataFrame) -> DataFrame:
    """Inverted index: ``(record_id, all_keys: array<string>)``.

    Records the full set of original blocking keys each record carries.
    Used by ``refine_blocks_spark`` to discover co-occurring keys; stays
    fixed across recursion depths because algo1_2_v2's refine_blocks
    looks at the record's ORIGINAL token list, not the compound block ids.
    """
    return keyed.groupBy("record_id").agg(
        F.collect_set("block_key").alias("all_keys")
    )


def _checkpoint(df: DataFrame, cfg: PipelineConfig) -> DataFrame:
    """Cut the lineage between recursion iterations.

    Without this the logical plan grows on every refine/merge/purge
    round and the driver eventually OOMs during query optimization
    (same failure mode as the small/large-star CC loop). Mirrors
    ``pipeline.merge._truncate_lineage``.
    """
    if cfg.checkpoint_dir:
        return df.checkpoint(eager=True)
    return df.localCheckpoint(eager=True)


def refine_blocks_spark(blocks: DataFrame, record_keys: DataFrame,
                       min_intra_freq: int) -> DataFrame:
    """Per-block: count co-occurring keys among members, emit refined sub-blocks.

    For each block, every member contributes the keys it carries
    (excluding ones already in ``used_tokens``). A co-key with support
    >= ``min_intra_freq`` seeds a refined block keyed by
    ``sorted(used_tokens ∪ {co_key})``, restricted to members carrying
    that co-key. Spark analog of ``refine_blocks`` from
    Algo1_2_v2/refine_blocks.py.
    """
    enriched = blocks.join(record_keys, on="record_id", how="inner")
    candidates = (
        enriched
        .select("block_id", "used_tokens", "record_id",
                F.explode("all_keys").alias("cand"))
        .where(~F.array_contains(F.col("used_tokens"), F.col("cand")))
    )
    support = (
        candidates
        .groupBy("block_id", "cand")
        .agg(F.countDistinct("record_id").alias("cnt"))
        .where(F.col("cnt") >= F.lit(min_intra_freq))
        .select("block_id", "cand")
    )
    refined = (
        candidates.join(support, on=["block_id", "cand"], how="inner")
        .withColumn(
            "new_used",
            F.array_sort(
                F.array_union(F.col("used_tokens"), F.array(F.col("cand")))
            ),
        )
        .withColumn("new_block_id", F.concat_ws("|", F.col("new_used")))
        .select(
            F.col("new_block_id").alias("block_id"),
            F.col("new_used").alias("used_tokens"),
            F.col("record_id"),
        )
        .dropDuplicates(["block_id", "record_id"])
    )
    return refined


def merge_blocks_spark(blocks: DataFrame) -> DataFrame:
    """Collapse blocks with identical member sets into one canonical block.

    Member fingerprint = sha2 of sorted member ids. Among blocks sharing
    a fingerprint, keep the one with the lexicographically smallest
    block_id. This matches the dedupe semantics of ``merge_blocks`` in
    Algo1_2_v2/recursive_algo1_2_v2.py without tracking alias keys (the
    rest of the ARCS pipeline does not need them).
    """
    fp = (
        blocks.groupBy("block_id")
        .agg(F.collect_list("record_id").alias("members"))
        .withColumn(
            "fp",
            F.sha2(
                F.concat_ws(",", F.array_sort(F.col("members"))),
                256,
            ),
        )
        .select("block_id", "fp")
    )
    canon = fp.groupBy("fp").agg(F.min("block_id").alias("canonical_id"))
    keep_ids = (
        fp.join(canon, on="fp", how="inner")
        .where(F.col("block_id") == F.col("canonical_id"))
        .select("block_id")
    )
    return blocks.join(F.broadcast(keep_ids), on="block_id", how="inner")


def purge_subset_blocks_spark(blocks: DataFrame) -> DataFrame:
    """Drop blocks whose member set is a strict subset of another's.

    For each block, count how many of its members each candidate block
    also contains. If ``shared == self_size`` and the candidate is
    strictly larger, self is a strict subset and is dropped.

    This is the heaviest stage in algo1_2_v2 and is gated behind
    ``cfg.arcs_do_purge`` because the per-record block-membership
    explode can be quadratic in the worst case.
    """
    sizes = blocks.groupBy("block_id").agg(F.count("*").alias("size"))
    rec_blocks = (
        blocks.groupBy("record_id")
        .agg(F.collect_set("block_id").alias("rec_block_ids"))
    )
    enriched = blocks.join(rec_blocks, on="record_id", how="inner")
    expanded = (
        enriched.select(
            "block_id",
            "record_id",
            F.explode("rec_block_ids").alias("candidate_id"),
        )
        .where(F.col("block_id") != F.col("candidate_id"))
    )
    shared = (
        expanded
        .groupBy("block_id", "candidate_id")
        .agg(F.countDistinct("record_id").alias("shared"))
        .join(sizes.withColumnRenamed("size", "self_size"), on="block_id")
        .join(
            sizes.withColumnRenamed("block_id", "candidate_id")
                 .withColumnRenamed("size", "other_size"),
            on="candidate_id",
        )
    )
    subsets = (
        shared
        .where(
            (F.col("shared") == F.col("self_size")) &
            (F.col("other_size") > F.col("self_size"))
        )
        .select("block_id")
        .distinct()
    )
    return blocks.join(subsets, on="block_id", how="left_anti")


def filter_top_k_spark(blocks: DataFrame, k: int,
                       min_block_size: int) -> DataFrame:
    """Per-record top-k smallest-block filter (Block Filtering, Papadakis et al.).

    Spark analog of ``filter_top_k_smallest`` from algo1_2_v2. Each
    record keeps memberships only in its k smallest blocks; blocks that
    fall below ``min_block_size`` after the thinning are dropped.
    """
    sizes = blocks.groupBy("block_id").agg(F.count("*").alias("size"))
    enriched = blocks.join(sizes, on="block_id", how="inner")
    w = Window.partitionBy("record_id").orderBy(
        F.col("size").asc(), F.col("block_id").asc()
    )
    ranked = (
        enriched
        .withColumn("rn", F.row_number().over(w))
        .where(F.col("rn") <= F.lit(k))
        .drop("rn", "size")
    )
    new_sizes = ranked.groupBy("block_id").agg(F.count("*").alias("new_size"))
    keep = (
        new_sizes.where(F.col("new_size") >= F.lit(min_block_size))
        .select("block_id")
    )
    return ranked.join(F.broadcast(keep), on="block_id", how="inner")


def build_arcs_edges_spark(blocks: DataFrame, weighting: str,
                           corpus_size: int,
                           max_block_pair_cost: int) -> DataFrame:
    """Build ARCS-weighted edges by self-joining within each block.

    Per block of size s, every pair (a, b) accumulates a contribution:
      weighting="uniform":  1 / s
      weighting="idf":      log(corpus_size / s) / s

    Blocks with more than ``max_block_pair_cost`` pairs are skipped
    entirely (compute guardrail mirroring algo1_2_v2's check).
    """
    sizes = blocks.groupBy("block_id").agg(F.count("*").alias("size"))
    if max_block_pair_cost is not None and max_block_pair_cost > 0:
        sizes = sizes.where(
            (F.col("size") * (F.col("size") - F.lit(1)) / F.lit(2))
            <= F.lit(max_block_pair_cost)
        )
    if weighting == "idf":
        N = F.lit(float(corpus_size))
        sizes = sizes.withColumn(
            "contribution",
            F.when(N > F.col("size"),
                   F.log(N / F.col("size")) / F.col("size"))
             .otherwise(F.lit(0.0)),
        )
    else:
        sizes = sizes.withColumn(
            "contribution", F.lit(1.0) / F.col("size"),
        )
    sizes = sizes.where(F.col("contribution") > F.lit(0.0))

    blocks_with_contrib = blocks.join(
        sizes.select("block_id", "contribution"),
        on="block_id", how="inner",
    )
    a = blocks_with_contrib.alias("a")
    b = blocks_with_contrib.alias("b")
    pairs = (
        a.join(b, F.col("a.block_id") == F.col("b.block_id"), how="inner")
         .where(F.col("a.record_id") < F.col("b.record_id"))
         .select(
             F.col("a.record_id").alias("src"),
             F.col("b.record_id").alias("dst"),
             F.col("a.contribution").alias("contribution"),
         )
    )
    edges = (
        pairs
        .groupBy("src", "dst")
        .agg(F.sum("contribution").alias("weight"))
    )
    return edges


def cluster_blocks_arcs(keyed: DataFrame, cfg: PipelineConfig) -> DataFrame:
    """ARCS-style edge construction from block redundancy.

    Returns a DataFrame ``(src long, dst long)`` matching the DBSCAN
    path's output schema, with self-edges unioned in for singleton
    coverage.
    """
    if cfg.arcs_density_floor > 0:
        raise NotImplementedError(
            "arcs_density_floor > 0 is not implemented for the Spark ARCS "
            "path in v1. Set it to 0.0."
        )

    blocks = _initial_blocks(keyed)
    record_keys = _record_keys(keyed)

    if cfg.arcs_max_recursion_depth > 0:
        record_keys = _checkpoint(record_keys, cfg)
        for depth in range(1, cfg.arcs_max_recursion_depth + 1):
            floor = max(depth, cfg.arcs_min_intra_freq)
            refined = refine_blocks_spark(blocks, record_keys, floor)
            blocks = (
                blocks.unionByName(refined)
                .dropDuplicates(["block_id", "record_id"])
            )
            if cfg.arcs_do_merge:
                blocks = merge_blocks_spark(blocks)
            if cfg.arcs_do_purge:
                blocks = purge_subset_blocks_spark(blocks)
            blocks = _checkpoint(blocks, cfg)

    blocks = filter_top_k_spark(
        blocks,
        k=cfg.arcs_top_k,
        min_block_size=cfg.min_block_size,
    )

    corpus_size = (
        keyed.select("record_id").distinct().count()
        if cfg.arcs_weighting == "idf" else 0
    )

    edges = build_arcs_edges_spark(
        blocks,
        weighting=cfg.arcs_weighting,
        corpus_size=corpus_size,
        max_block_pair_cost=cfg.arcs_max_block_pair_cost,
    )
    edges = (
        edges
        .where(F.col("weight") >= F.lit(cfg.arcs_tau))
        .select(
            F.col("src").cast(T.LongType()).alias("src"),
            F.col("dst").cast(T.LongType()).alias("dst"),
        )
    )

    selves = (
        keyed.select(
            F.col("record_id").cast(T.LongType()).alias("src"),
            F.col("record_id").cast(T.LongType()).alias("dst"),
        )
        .dropDuplicates(["src"])
    )
    return edges.unionByName(selves)
