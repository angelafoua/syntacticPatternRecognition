#!/usr/bin/env python3
"""CLI entrypoint for the syntactic clustering pipeline.

Examples:

    # Default DBSCAN path:
    python run_pipeline.py \\
        --input /data/raw \\
        --output /data/clusters \\
        --format csv \\
        --shuffle-partitions 2000 \\
        --eps 0.3

    # ARCS path (mirrors blockingandembeddinginer/Algo1_2_v2):
    python run_pipeline.py \\
        --input /data/raw \\
        --output /data/clusters \\
        --clustering-method arcs \\
        --arcs-tau 0.2 \\
        --arcs-top-k 3

    # With pair-based metrics against a truth file:
    python run_pipeline.py \\
        --input s3://bucket/data/S12PX.txt \\
        --output s3://bucket/output/clusters_arcs/ \\
        --clustering-method arcs \\
        --truth-file s3://bucket/data/truthABCpoorDQ.txt
"""

from __future__ import annotations

import argparse
import sys

from pipeline.config import PipelineConfig
from pipeline.pipeline import run_pipeline


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--format", default="csv",
                   choices=["csv", "tsv", "text", "parquet"])
    p.add_argument("--has-header", action="store_true",
                   help="Treat the first row of the input as a header and skip it.")
    p.add_argument("--delimiter", default=",")
    p.add_argument("--skip-leading-columns", type=int, default=0,
                   help="Drop the first N data columns (e.g. an existing "
                        "record-id column) from the cell stream.")
    p.add_argument("--shuffle-partitions", type=int, default=400)

    # ---- Clustering selection ----
    p.add_argument("--clustering-method", default="dbscan",
                   choices=["dbscan", "arcs"],
                   help="Per-block clustering path. 'dbscan' (default) "
                        "runs the original applyInPandas DBSCAN. 'arcs' "
                        "runs the Spark-native ARCS graph path mirroring "
                        "blockingandembeddinginer/Algo1_2_v2.")

    # ---- DBSCAN-only knobs ----
    p.add_argument("--eps", type=float, default=0.35)
    p.add_argument("--min-samples", type=int, default=2)

    # ---- ARCS-only knobs ----
    p.add_argument("--arcs-weighting", default="uniform",
                   choices=["uniform", "idf"],
                   help="ARCS pair contribution. uniform=1/|B|, "
                        "idf=log(N/|B|)/|B|. Re-tune --arcs-tau when "
                        "switching modes.")
    p.add_argument("--arcs-tau", type=float, default=0.2,
                   help="Edge-weight threshold for the ARCS graph.")
    p.add_argument("--arcs-top-k", type=int, default=3,
                   help="Per-record top-k smallest-block filter.")
    p.add_argument("--arcs-min-intra-freq", type=int, default=2,
                   help="Lower-bound floor on intra-block co-key support "
                        "during recursive refinement.")
    p.add_argument("--arcs-max-recursion-depth", type=int, default=0,
                   help="Number of refine/merge/purge rounds. 0 disables.")
    p.add_argument("--arcs-no-merge", action="store_true",
                   help="Disable identical-member-set block dedup.")
    p.add_argument("--arcs-do-purge", action="store_true",
                   help="Enable strict-subset block purge (heavy stage).")
    p.add_argument("--arcs-max-block-pair-cost", type=int, default=100_000,
                   help="Skip any block whose pair count exceeds this.")

    # ---- Block-size and downstream filters (shared) ----
    p.add_argument("--min-block-size", type=int, default=2)
    p.add_argument("--max-block-size", type=int, default=50_000)
    p.add_argument("--min-cluster-size", type=int, default=2)
    p.add_argument("--max-cluster-size", type=int, default=1_000_000)
    p.add_argument("--cohesion", type=float, default=0.0)
    p.add_argument("--checkpoint-dir", default="")
    p.add_argument("--merge-strategy", default="auto",
                   choices=["auto", "local", "starstar"],
                   help="How to merge per-block clusters into final entities. "
                        "'local' collects edges to the driver and runs "
                        "union-find (fast for <~10M edges). 'starstar' is the "
                        "distributed small-star/large-star loop for billion-scale. "
                        "'auto' picks based on --merge-local-edge-limit.")
    p.add_argument("--merge-local-edge-limit", type=int, default=5_000_000)
    p.add_argument("--output-format", default="parquet",
                   choices=["parquet", "json"])

    # ---- Metrics ----
    p.add_argument("--no-metrics", action="store_true",
                   help="Skip metrics computation and the metrics.{json,log} "
                        "output. Predicted-cluster size distribution is "
                        "otherwise always emitted.")
    p.add_argument("--truth-file", default="",
                   help="Path (local or s3://) to a truth CSV with header "
                        "and 'refID,truthID' rows. When set, pair-based "
                        "precision / recall / F1 are computed and written "
                        "alongside the size distributions.")
    p.add_argument("--ref-id-column", type=int, default=0,
                   help="Physical column index of the input that holds the "
                        "refID matching the truth file (default 0). "
                        "Independent of --skip-leading-columns: the same "
                        "column can be captured for metrics AND dropped "
                        "from clustering.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    cfg = PipelineConfig(
        input_format=args.format,
        input_has_header=args.has_header,
        input_delimiter=args.delimiter,
        skip_leading_columns=args.skip_leading_columns,
        shuffle_partitions=args.shuffle_partitions,
        clustering_method=args.clustering_method,
        dbscan_eps=args.eps,
        dbscan_min_samples=args.min_samples,
        arcs_weighting=args.arcs_weighting,
        arcs_tau=args.arcs_tau,
        arcs_top_k=args.arcs_top_k,
        arcs_min_intra_freq=args.arcs_min_intra_freq,
        arcs_max_recursion_depth=args.arcs_max_recursion_depth,
        arcs_do_merge=not args.arcs_no_merge,
        arcs_do_purge=args.arcs_do_purge,
        arcs_max_block_pair_cost=args.arcs_max_block_pair_cost,
        min_block_size=args.min_block_size,
        max_block_size=args.max_block_size,
        min_cluster_size=args.min_cluster_size,
        max_cluster_size=args.max_cluster_size,
        min_cluster_cohesion=args.cohesion,
        checkpoint_dir=args.checkpoint_dir,
        merge_strategy=args.merge_strategy,
        merge_local_edge_limit=args.merge_local_edge_limit,
        output_format=args.output_format,
        metrics_enabled=not args.no_metrics,
        truth_file=args.truth_file,
        ref_id_column=args.ref_id_column,
    )
    run_pipeline(args.input, args.output, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
