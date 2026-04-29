"""Multi-key blocking.

For every cell we emit several blocking keys; values that share *any* key
are co-located in the same partition for local clustering. This trades a
modest fan-out per cell for a guaranteed escape from O(n^2) global pairs.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from pipeline.config import PipelineConfig


def _length_bucket_expr(length_col: F.Column, edges: tuple[int, ...]) -> F.Column:
    # Build a CASE WHEN chain: length <= edge[i] -> bucket index i.
    # Static unrolling keeps this as pure Catalyst with no UDFs.
    expr = F.lit(len(edges))  # default: above last edge
    for i in range(len(edges) - 1, -1, -1):
        expr = F.when(length_col <= F.lit(edges[i]), F.lit(i)).otherwise(expr)
    return expr


def generate_block_keys(df: DataFrame, cfg: PipelineConfig) -> DataFrame:
    """Explode each cell into one row per blocking key.

    Output columns: record_id, source, col_index, norm_value, feat, block_key.
    The same (record_id, value) may appear multiple times (once per key).
    """
    v = F.col("norm_value")
    feat = F.col("feat")
    pattern = feat["pattern"]
    pattern_compact = feat["pattern_compact"]
    length = feat["length"]
    tokens = feat["token_count"]

    keys = []

    if cfg.emit_prefix_key:
        keys.append(F.concat(F.lit("PFX_"), F.substring(v, 1, cfg.prefix_len)))

    if cfg.emit_pattern_key:
        # crc32 keeps cardinality bounded yet stable across runs/executors.
        bucket = F.pmod(F.crc32(pattern), F.lit(cfg.pattern_hash_buckets))
        keys.append(F.concat(F.lit("PAT_"), bucket.cast(T.StringType())))

    if cfg.emit_compressed_pattern_key:
        bucket = F.pmod(F.crc32(pattern_compact), F.lit(cfg.pattern_hash_buckets))
        keys.append(F.concat(F.lit("CPT_"), bucket.cast(T.StringType())))

    if cfg.emit_length_key:
        bkt = _length_bucket_expr(length, cfg.length_bucket_edges)
        keys.append(F.concat(F.lit("LEN_"), bkt.cast(T.StringType())))

    if cfg.emit_token_key:
        # Cap token bucket so a 10000-token outlier doesn't create its own block.
        capped = F.when(tokens > F.lit(32), F.lit(32)).otherwise(tokens)
        keys.append(F.concat(F.lit("TOK_"), capped.cast(T.StringType())))

    if not keys:
        raise ValueError("At least one blocking key must be enabled.")

    keyed = df.withColumn("_keys", F.array(*keys))
    return (
        keyed
        .select(
            F.col("record_id"),
            F.col("source"),
            F.col("col_index"),
            F.col("norm_value"),
            F.col("feat"),
            F.explode("_keys").alias("block_key"),
        )
    )


def filter_block_sizes(keyed: DataFrame, cfg: PipelineConfig) -> DataFrame:
    """Drop tiny blocks (no comparisons possible) and runaway giants.

    The size statistic is computed once and joined back; we deliberately
    avoid groupBy().agg().filter().join() round-trips by using a window-
    free aggregate with broadcast on the filtered key set.
    """
    sizes = keyed.groupBy("block_key").count()
    keep_keys = sizes.where(
        (F.col("count") >= cfg.min_block_size) &
        (F.col("count") <= cfg.max_block_size)
    ).select("block_key")
    return keyed.join(F.broadcast(keep_keys), on="block_key", how="inner")
