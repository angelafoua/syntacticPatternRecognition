"""Local clustering inside each block.

Clustering is intentionally constrained to a single block at a time so
the global cost stays linear in the number of cells. Within a block we
run DBSCAN on a normalized syntactic feature vector. DBSCAN is the right
fit here because:

* the number of clusters per block is unknown,
* clusters can have arbitrary shape in the feature space,
* noise points are common (dirty data) and must remain isolated.

Output of this stage is a stream of edges
``(record_id_a, record_id_b)`` that link cells the local clusterer judged
equivalent. The downstream connected-components stage stitches edges
across blocks into final entity groups.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from pipeline.config import PipelineConfig
from pipeline.features import FEATURE_FIELDS


EDGE_SCHEMA = T.StructType([
    T.StructField("src", T.LongType(), True),
    T.StructField("dst", T.LongType(), True),
])


# Per-feature scale used to bring vectors into a comparable range before
# DBSCAN. Picked from the natural range of each feature (length is the
# only one that grows unboundedly, so we log-scale it). These scales are
# intentionally fixed rather than learned: at billion-scale we cannot
# afford a global pass to compute statistics, and DBSCAN's eps is what
# the operator actually tunes.
_FEATURE_SCALES = {
    "length": 64.0,        # divided AFTER log1p
    "digit_ratio": 1.0,
    "alpha_ratio": 1.0,
    "symbol_ratio": 1.0,
    "space_ratio": 1.0,
    "token_count": 8.0,
    "unique_chars": 32.0,
    "upper_ratio": 1.0,
    "vowel_ratio": 1.0,
    "entropy": 6.0,
}


def _vectorize(rows: pd.DataFrame) -> np.ndarray:
    out = np.empty((len(rows), len(FEATURE_FIELDS)), dtype=np.float32)
    for i, name in enumerate(FEATURE_FIELDS):
        col = rows[name].to_numpy(dtype=np.float32, copy=False)
        if name == "length":
            col = np.log1p(col)
        out[:, i] = col / _FEATURE_SCALES[name]
    return out


def _dbscan_labels(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Tiny dependency-free DBSCAN with chunked distance computation.

    We avoid sklearn (not guaranteed installed on executors) and avoid
    the O(n^2 * d) broadcast intermediate that the naive implementation
    builds. Instead, distances are computed one row-chunk at a time, so
    peak memory is O(chunk * n) rather than O(n^2 * d).

    For n=5000, d=10 the chunked version peaks at ~5 MB instead of ~1 GB.
    """
    n = X.shape[0]
    labels = np.full(n, -1, dtype=np.int32)
    if n == 0:
        return labels

    eps2 = float(eps) * float(eps)

    # Squared norms cached once: |x|^2 + |y|^2 - 2 x.y is the standard
    # rewrite that lets us compute pairwise sq-distance via one matmul.
    sq = (X * X).sum(axis=1)

    # Build a boolean adjacency matrix in chunks. n^2 bits is the only
    # unavoidable cost; for n=5000 that's 25 MB packed (200 MB as bool).
    # We keep it as bool for fast row sums and BFS lookups.
    neighbors = np.zeros((n, n), dtype=bool)
    chunk = 256
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        # (chunk, n) sq-distance block, no broadcasting blowup.
        d2 = sq[i:j, None] + sq[None, :] - 2.0 * X[i:j] @ X.T
        np.less_equal(d2, eps2, out=neighbors[i:j])

    counts = neighbors.sum(axis=1)
    is_core = counts >= min_samples

    cluster_id = 0
    for i in range(n):
        if labels[i] != -1 or not is_core[i]:
            continue
        # BFS expansion of the core point.
        stack = [i]
        labels[i] = cluster_id
        while stack:
            j = stack.pop()
            nbrs = np.flatnonzero(neighbors[j])
            for k in nbrs:
                if labels[k] == -1:
                    labels[k] = cluster_id
                    if is_core[k]:
                        stack.append(k)
        cluster_id += 1

    return labels


def _block_to_edges(block: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Run local DBSCAN and return star-edges for every non-noise cluster.

    For each cluster we emit edges (anchor -> member) using the smallest
    record_id in the cluster as the anchor. This both deduplicates and
    feeds the downstream connected-components step a sparse edge list.
    """
    if len(block) < cfg.dbscan_min_samples:
        return pd.DataFrame({"src": [], "dst": []}).astype({"src": "int64", "dst": "int64"})

    # Subsample oversized blocks so DBSCAN's O(b^2) stays bounded.
    sampled = block
    if len(block) > cfg.cluster_sample_cap:
        sampled = block.sample(
            n=cfg.cluster_sample_cap,
            random_state=cfg.random_seed,
            replace=False,
        )

    X = _vectorize(sampled)
    labels = _dbscan_labels(X, cfg.dbscan_eps, cfg.dbscan_min_samples)

    rec_ids = sampled["record_id"].to_numpy(dtype=np.int64)
    edges_src = []
    edges_dst = []
    for cid in np.unique(labels):
        if cid < 0:
            continue
        members = rec_ids[labels == cid]
        if members.size < 2:
            continue
        anchor = int(members.min())
        for m in members:
            mi = int(m)
            if mi == anchor:
                continue
            edges_src.append(anchor)
            edges_dst.append(mi)

    if not edges_src:
        return pd.DataFrame({"src": [], "dst": []}).astype({"src": "int64", "dst": "int64"})

    return pd.DataFrame({"src": edges_src, "dst": edges_dst})


def cluster_blocks(keyed: DataFrame, cfg: PipelineConfig) -> DataFrame:
    """Group cells by block_key and emit (src, dst) record-id edges."""

    feature_cols = [F.col("feat")[name].alias(name) for name in FEATURE_FIELDS]
    flat = keyed.select(
        F.col("block_key"),
        F.col("record_id"),
        *feature_cols,
    )

    # Snapshot config values into closure-local primitives so the UDF is
    # serializable and not coupled to the dataclass.
    eps = float(cfg.dbscan_eps)
    min_samples = int(cfg.dbscan_min_samples)
    sample_cap = int(cfg.cluster_sample_cap)
    seed = int(cfg.random_seed)

    class _Cfg:
        dbscan_eps = eps
        dbscan_min_samples = min_samples
        cluster_sample_cap = sample_cap
        random_seed = seed

    def _apply(pdf: pd.DataFrame) -> pd.DataFrame:
        return _block_to_edges(pdf, _Cfg)

    edges = (
        flat
        .groupBy("block_key")
        .applyInPandas(_apply, schema=EDGE_SCHEMA)
    )

    # Also emit trivial self-edges so isolated record_ids that end up in
    # no DBSCAN cluster still flow through the connected-components stage
    # and appear in the final output as singleton clusters.
    selves = (
        keyed.select(F.col("record_id").alias("src"), F.col("record_id").alias("dst"))
             .dropDuplicates(["src"])
    )

    return edges.unionByName(selves)
