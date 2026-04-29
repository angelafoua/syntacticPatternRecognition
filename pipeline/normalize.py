"""Cheap, purely lexical normalization.

Heavy regex per-cell at billion scale is the dominant cost, so each step
uses a single Spark expression. We do *not* strip punctuation: punctuation
is signal for the pattern signature.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from pipeline.config import PipelineConfig


# Non-printable / control characters except tab. Matched once at scale.
_NON_PRINTABLE = r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]"


def normalize(df: DataFrame, cfg: PipelineConfig) -> DataFrame:
    v = F.col("value")

    if cfg.strip_non_printable:
        v = F.regexp_replace(v, _NON_PRINTABLE, "")
    if cfg.lowercase:
        v = F.lower(v)
    if cfg.collapse_whitespace:
        v = F.regexp_replace(F.trim(v), r"\s+", " ")
    v = F.substring(v, 1, cfg.max_value_length)

    out = df.withColumn("norm_value", v)
    return out.where(F.length(F.col("norm_value")) >= cfg.min_value_length)
