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
    # DBSCAN parameters used per block on the normalized feature vector.
    dbscan_eps: float = 0.35
    dbscan_min_samples: int = 2
    # Subsample very large blocks before clustering; remaining points are
    # assigned to the nearest core via 1-NN to bound per-block cost.
    cluster_sample_cap: int = 5_000

    # ---- Merge -----------------------------------------------------------
    # Iterations for the small-star/large-star connected-components loop.
    cc_max_iterations: int = 30

    # ---- Quality filter --------------------------------------------------
    min_cluster_size: int = 2
    max_cluster_size: int = 1_000_000
    min_cluster_cohesion: float = 0.0  # 0 disables cohesion filter

    # ---- Spark / IO ------------------------------------------------------
    shuffle_partitions: int = 400
    output_format: str = "parquet"
    checkpoint_dir: str = ""  # set to enable Spark checkpointing for CC

    # ---- Reproducibility -------------------------------------------------
    random_seed: int = 0xC0FFEE

    def with_overrides(self, **kwargs) -> "PipelineConfig":
        return PipelineConfig(**{**self.__dict__, **kwargs})
