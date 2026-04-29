"""Tests for the syntactic feature extraction primitives.

These tests exercise the pure-Python helpers directly (no Spark) so they
can run anywhere `numpy` and `pandas` are installed.
"""

from __future__ import annotations

import pandas as pd

from pipeline.features import _featurize_batch, _signature


def test_signature_basic():
    assert _signature("ab12") == ("AA99", "A9")
    assert _signature("A B") == ("A A", "A A")
    assert _signature("hello-world") == ("AAAAA-AAAAA", "A-A")
    assert _signature("") == ("", "")


def test_signature_preserves_punctuation():
    pat, compact = _signature("(415) 555-1212")
    assert pat == "(999) 999-9999"
    assert compact == "(9) 9-9"


def test_featurize_batch_shape_and_ratios():
    s = pd.Series(["abc123", "   ", "ABCDE"])
    df = _featurize_batch(s)
    assert list(df.columns) == [
        "length", "digit_ratio", "alpha_ratio", "symbol_ratio",
        "space_ratio", "token_count", "unique_chars", "upper_ratio",
        "vowel_ratio", "entropy", "pattern", "pattern_compact",
    ]
    assert df.loc[0, "length"] == 6
    assert df.loc[0, "digit_ratio"] == 0.5
    assert df.loc[0, "alpha_ratio"] == 0.5
    assert df.loc[2, "upper_ratio"] == 1.0


def test_featurize_handles_empty():
    df = _featurize_batch(pd.Series([""]))
    assert df.loc[0, "length"] == 0
    assert df.loc[0, "pattern"] == ""
    assert df.loc[0, "entropy"] == 0.0
