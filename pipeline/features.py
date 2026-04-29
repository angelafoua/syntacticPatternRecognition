"""Syntactic, type-blind feature extraction.

Every feature here is computed from the raw character stream and carries
no assumption about what the value represents. The output is a small,
fixed-width numeric vector suitable for clustering, plus the string
pattern signature used for blocking.

We push the heavy-lifting into a Pandas UDF so each executor processes
batches of strings in vectorized C code rather than per-row Python.
"""

from __future__ import annotations

import re
from typing import Iterator

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from pipeline.config import PipelineConfig


FEATURE_FIELDS = [
    "length",
    "digit_ratio",
    "alpha_ratio",
    "symbol_ratio",
    "space_ratio",
    "token_count",
    "unique_chars",
    "upper_ratio",
    "vowel_ratio",
    "entropy",
]

FEATURE_SCHEMA = T.StructType([
    T.StructField("length", T.IntegerType(), True),
    T.StructField("digit_ratio", T.FloatType(), True),
    T.StructField("alpha_ratio", T.FloatType(), True),
    T.StructField("symbol_ratio", T.FloatType(), True),
    T.StructField("space_ratio", T.FloatType(), True),
    T.StructField("token_count", T.IntegerType(), True),
    T.StructField("unique_chars", T.IntegerType(), True),
    T.StructField("upper_ratio", T.FloatType(), True),
    T.StructField("vowel_ratio", T.FloatType(), True),
    T.StructField("entropy", T.FloatType(), True),
    T.StructField("pattern", T.StringType(), True),
    T.StructField("pattern_compact", T.StringType(), True),
])


_TOKEN_SPLIT = re.compile(r"[\s\-_/.,;:|]+")
_VOWELS = set("aeiouAEIOU")


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    h = 0.0
    inv_n = 1.0 / n
    for c in counts.values():
        p = c * inv_n
        h -= p * np.log2(p)
    return float(h)


def _signature(s: str) -> tuple[str, str]:
    """Return (pattern, compact_pattern).

    pattern: digits -> '9', letters -> 'A', whitespace -> ' ', else verbatim.
    compact_pattern: pattern with adjacent duplicates collapsed.
    """
    if not s:
        return "", ""
    out = []
    for ch in s:
        if ch.isdigit():
            out.append("9")
        elif ch.isalpha():
            out.append("A")
        elif ch.isspace():
            out.append(" ")
        else:
            out.append(ch)
    pattern = "".join(out)

    compact = []
    prev = None
    for ch in pattern:
        if ch != prev:
            compact.append(ch)
            prev = ch
    return pattern, "".join(compact)


def _featurize_batch(values: pd.Series) -> pd.DataFrame:
    rows = []
    for raw in values:
        s = "" if raw is None else str(raw)
        n = len(s)
        if n == 0:
            rows.append((0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, "", ""))
            continue

        digits = letters = spaces = uppers = vowels = 0
        seen: set[str] = set()
        for ch in s:
            seen.add(ch)
            if ch.isdigit():
                digits += 1
            elif ch.isalpha():
                letters += 1
                if ch.isupper():
                    uppers += 1
                if ch in _VOWELS:
                    vowels += 1
            elif ch.isspace():
                spaces += 1
        symbols = n - digits - letters - spaces

        tokens = [t for t in _TOKEN_SPLIT.split(s) if t]
        pattern, compact = _signature(s)

        rows.append((
            n,
            digits / n,
            letters / n,
            symbols / n,
            spaces / n,
            len(tokens),
            len(seen),
            (uppers / letters) if letters else 0.0,
            (vowels / letters) if letters else 0.0,
            _shannon_entropy(s),
            pattern,
            compact,
        ))

    return pd.DataFrame(rows, columns=[f.name for f in FEATURE_SCHEMA.fields])


def extract_features(df: DataFrame, _cfg: PipelineConfig) -> DataFrame:
    """Add a `feat` struct column populated by a vectorized Pandas UDF."""

    @F.pandas_udf(FEATURE_SCHEMA)
    def featurize(batch_iter: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
        for batch in batch_iter:
            yield _featurize_batch(batch)

    return df.withColumn("feat", featurize(F.col("norm_value")))
