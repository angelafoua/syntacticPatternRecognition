"""Tests for the in-block DBSCAN helpers (no Spark required)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.clustering import _block_to_edges, _dbscan_labels


class _Cfg:
    dbscan_eps = 0.2
    dbscan_min_samples = 2
    cluster_sample_cap = 1000
    random_seed = 0


def test_dbscan_finds_two_clusters():
    X = np.array([
        [0.0, 0.0], [0.05, 0.05], [0.1, 0.0],   # cluster A
        [5.0, 5.0], [5.1, 5.1],                 # cluster B
        [10.0, 10.0],                           # noise
    ], dtype=np.float32)
    labels = _dbscan_labels(X, eps=0.5, min_samples=2)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4]
    assert labels[0] != labels[3]
    assert labels[5] == -1


def test_block_to_edges_emits_anchor_star():
    block = pd.DataFrame({
        "record_id": [10, 11, 12, 99],
        "length": [3, 3, 3, 50],
        "digit_ratio": [0.0, 0.0, 0.0, 0.0],
        "alpha_ratio": [1.0, 1.0, 1.0, 1.0],
        "symbol_ratio": [0.0, 0.0, 0.0, 0.0],
        "space_ratio": [0.0, 0.0, 0.0, 0.0],
        "token_count": [1, 1, 1, 1],
        "unique_chars": [3, 3, 3, 25],
        "upper_ratio": [0.0, 0.0, 0.0, 0.0],
        "vowel_ratio": [0.5, 0.5, 0.5, 0.5],
        "entropy": [1.5, 1.5, 1.5, 4.0],
    })
    edges = _block_to_edges(block, _Cfg)
    # Three near-identical rows -> star edges to anchor 10.
    assert set(map(tuple, edges.values.tolist())) == {(10, 11), (10, 12)}
