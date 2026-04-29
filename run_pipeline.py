#!/usr/bin/env python3
"""CLI entrypoint for the syntactic clustering pipeline.

Examples:

    python run_pipeline.py \
        --input /data/raw \
        --output /data/clusters \
        --format csv \
        --shuffle-partitions 2000 \
        --eps 0.3
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
    p.add_argument("--eps", type=float, default=0.35)
    p.add_argument("--min-samples", type=int, default=2)
    p.add_argument("--min-block-size", type=int, default=2)
    p.add_argument("--max-block-size", type=int, default=50_000)
    p.add_argument("--min-cluster-size", type=int, default=2)
    p.add_argument("--max-cluster-size", type=int, default=1_000_000)
    p.add_argument("--cohesion", type=float, default=0.0)
    p.add_argument("--checkpoint-dir", default="")
    p.add_argument("--output-format", default="parquet",
                   choices=["parquet", "json"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    cfg = PipelineConfig(
        input_format=args.format,
        input_has_header=args.has_header,
        input_delimiter=args.delimiter,
        skip_leading_columns=args.skip_leading_columns,
        shuffle_partitions=args.shuffle_partitions,
        dbscan_eps=args.eps,
        dbscan_min_samples=args.min_samples,
        min_block_size=args.min_block_size,
        max_block_size=args.max_block_size,
        min_cluster_size=args.min_cluster_size,
        max_cluster_size=args.max_cluster_size,
        min_cluster_cohesion=args.cohesion,
        checkpoint_dir=args.checkpoint_dir,
        output_format=args.output_format,
    )
    run_pipeline(args.input, args.output, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
