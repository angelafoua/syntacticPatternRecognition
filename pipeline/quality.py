"""Quality filtering and final aggregation.

We collapse the per-record cluster assignments into one row per cluster,
attach lightweight cohesion / size statistics, and drop clusters that
are degenerate or pathological (e.g. a single mega-block with a million
unrelated members).
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from pipeline.config import PipelineConfig


def aggregate_clusters(assignments: DataFrame, cells: DataFrame,
                       cfg: PipelineConfig) -> DataFrame:
    """Build (cluster_id, members[], stats) rows.

    `assignments` has columns (record_id, cluster_id).
    `cells` has the original normalized cells (record_id, norm_value, feat).
    """
    # Average syntactic feature vector across cluster members serves as
    # cheap cohesion signal: tighter average dispersion => stronger
    # cluster. We approximate dispersion via std of length / digit_ratio.
    feats = cells.select(
        F.col("record_id"),
        F.col("feat")["length"].alias("_length"),
        F.col("feat")["digit_ratio"].alias("_dr"),
        F.col("feat")["alpha_ratio"].alias("_ar"),
    ).dropDuplicates(["record_id"])

    joined = assignments.join(feats, on="record_id", how="left")

    grouped = (
        joined.groupBy("cluster_id")
        .agg(
            F.collect_set("record_id").alias("members"),
            F.count("record_id").alias("size"),
            F.stddev_pop("_length").alias("std_length"),
            F.stddev_pop("_dr").alias("std_digit_ratio"),
            F.stddev_pop("_ar").alias("std_alpha_ratio"),
        )
    )

    # Cohesion in [0, 1]: 1 means perfectly homogeneous on the dispersion
    # signals; falls off smoothly. Length std is normalized by 64 so the
    # scale matches the ratio fields (whose std cannot exceed 0.5).
    cohesion = (
        F.lit(1.0)
        - (F.coalesce(F.col("std_length"), F.lit(0.0)) / F.lit(64.0)).cast("double")
        - F.coalesce(F.col("std_digit_ratio"), F.lit(0.0)).cast("double")
        - F.coalesce(F.col("std_alpha_ratio"), F.lit(0.0)).cast("double")
    )
    cohesion = F.greatest(F.lit(0.0), F.least(F.lit(1.0), cohesion))

    return grouped.withColumn("cohesion", cohesion)


def filter_clusters(clusters: DataFrame, cfg: PipelineConfig) -> DataFrame:
    return clusters.where(
        (F.col("size") >= F.lit(cfg.min_cluster_size)) &
        (F.col("size") <= F.lit(cfg.max_cluster_size)) &
        (F.col("cohesion") >= F.lit(cfg.min_cluster_cohesion))
    )
