"""Configuration object for the clustering pipeline.

All tunable knobs live here so the rest of the modules stay declarative.
The defaults are chosen to behave sensibly on heterogeneous, dirty data
where the distribution of cell values is unknown a priori.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class PipelineConfig:
    # ---- Ingestion -------------------------------------------------------
    # Input format: "csv", "tsv", "text" (one record per line, tab-delim
    # auto-detected), or "parquet". Unknown columns are tolerated; no header
    # is assumed.
    input_format: str = "csv"
    input_has_header: bool = False
    input_delimiter: str = ","
    # Drop the first N physical columns from the cell stream (e.g. a
    # leading record-id or row-number column the operator wants ignored).
    # The pipeline still stamps its own internal record_id for lineage.
    skip_leading_columns: int = 0
    # When ingesting a single column of text, the value is split on this
    # regex into pseudo-columns. Empty string disables splitting.
    text_split_regex: str = r"[\t|;]"

    # ---- Normalization ---------------------------------------------------
    lowercase: bool = True
    collapse_whitespace: bool = True
    strip_non_printable: bool = True
    # Drop values whose normalized form is empty or shorter than this.
    min_value_length: int = 2
    # Cap value length to bound feature work on pathological cells.
    max_value_length: int = 512

    # ---- Blocking --------------------------------------------------------
    prefix_len: int = 3
    pattern_hash_buckets: int = 10000
    length_bucket_edges: Tuple[int, ...] = (
        0, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256, 512,
    )
    # Multi-key blocking flags. Each key emitted independently.
    emit_prefix_key: bool = True
    emit_pattern_key: bool = True
    emit_length_key: bool = True
    emit_token_key: bool = True
    emit_compressed_pattern_key: bool = True  # collapses runs (e.g. AAA9 -> A9)

    # Skip oversized blocks that would dominate the shuffle. Values landing
    # only in oversized blocks are still emitted as singleton clusters.
    min_block_size: int = 2
    max_block_size: int = 50_000

    # ---- Local clustering -----------------------------------------------
    # Selects which clustering path runs after blocking:
    #   "dbscan" - per-block DBSCAN via applyInPandas (legacy default).
    #   "arcs"   - Spark-native ARCS edge graph mirroring
    #              blockingandembeddinginer/Algo1_2_v2.
    clustering_method: str = "dbscan"

    # DBSCAN parameters used per block on the normalized feature vector.
    dbscan_eps: float = 0.35
    dbscan_min_samples: int = 2
    # Subsample very large blocks before clustering; remaining points are
    # assigned to the nearest core via 1-NN to bound per-block cost.
    cluster_sample_cap: int = 5_000

    # ---- ARCS clustering (only used when clustering_method == "arcs") ---
    # Per-block pair contribution: "uniform" = 1/|B|, "idf" = log(N/|B|)/|B|.
    arcs_weighting: str = "uniform"
    # Edge-weight threshold; pairs with summed weight >= tau become edges.
    arcs_tau: float = 0.2
    # Per-record top-k smallest-block filter (Block Filtering).
    arcs_top_k: int = 3
    # Lower-bound floor on intra-block co-key support during refinement.
    # Effective floor at depth d is max(d, this).
    arcs_min_intra_freq: int = 2
    # 0 disables refinement; >0 runs that many refine/merge/purge rounds.
    arcs_max_recursion_depth: int = 0
    # Mirror of algo1_2_v2's do_merge / do_purge ablation flags.
    arcs_do_merge: bool = True
    # Purge is the heaviest stage in Spark; opt-in.
    arcs_do_purge: bool = False
    # Compute guardrail: skip blocks whose pair count exceeds this.
    arcs_max_block_pair_cost: int = 100_000
    # Density-floor cluster splitting is not implemented in v1. Setting
    # > 0 raises NotImplementedError until the follow-up lands.
    arcs_density_floor: float = 0.0
    arcs_density_min_size: int = 3

    # ---- Merge -----------------------------------------------------------
    # Strategy for the connected-components / merge step:
    #   "auto"       - local UF if edges fit, else small/large-star.
    #   "local"      - always collect edges to driver and run union-find.
    #                  Trivially fast for ~tens of millions of edges,
    #                  but blows up the driver if edges don't fit.
    #   "starstar"   - always run distributed small-star/large-star.
    #                  Designed for billion-scale.
    merge_strategy: str = "auto"
    # Threshold (number of edges) for "auto" to switch from local to
    # distributed. Below this, edges are pulled to the driver. Above,
    # the small/large-star loop runs.
    merge_local_edge_limit: int = 5_000_000
    # Iterations for the small-star/large-star connected-components loop.
    # Convergence is typically reached in O(log n) rounds, so 12 is plenty
    # for hundreds of millions of edges. The loop also exits early on a
    # fixed point.
    cc_max_iterations: int = 12

    # ---- Quality filter --------------------------------------------------
    min_cluster_size: int = 2
    max_cluster_size: int = 1_000_000
    min_cluster_cohesion: float = 0.0  # 0 disables cohesion filter

    # ---- Metrics ---------------------------------------------------------
    # Emit metrics.{json,log} under <output>/_metrics/. Always writes the
    # predicted size distribution; pair-based P/R/F1 added when truth_file
    # is provided. Mirror of algo1_2_v2's compute_run_metrics output.
    metrics_enabled: bool = True
    # Truth CSV (header + refID,truthID rows). Empty disables pair metrics.
    truth_file: str = ""
    # Which physical column of the raw input is the refID used in truth.
    ref_id_column: int = 0

    # ---- Spark / IO ------------------------------------------------------
    shuffle_partitions: int = 400
    output_format: str = "parquet"
    checkpoint_dir: str = ""  # set to enable Spark checkpointing for CC

    # ---- Reproducibility -------------------------------------------------
    random_seed: int = 0xC0FFEE

    def with_overrides(self, **kwargs) -> "PipelineConfig":
        return PipelineConfig(**{**self.__dict__, **kwargs})
