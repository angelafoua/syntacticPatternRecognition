"""Flatten arbitrary tabular / text input into (record_id, col, value) cells.

The pipeline is schema-agnostic, so we deliberately refuse to interpret
columns. We assign a stable ``record_id`` per input row and keep the cell
location only as metadata for diagnostics. Lineage is preserved through
every later stage by carrying ``record_id`` on every emitted row.
"""

from __future__ import annotations

from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from pipeline.config import PipelineConfig


CELL_SCHEMA = T.StructType([
    T.StructField("record_id", T.LongType(), nullable=False),
    T.StructField("source", T.StringType(), nullable=True),
    T.StructField("col_index", T.IntegerType(), nullable=False),
    T.StructField("value", T.StringType(), nullable=True),
])


def read_raw(spark: SparkSession, path: str, cfg: PipelineConfig,
             source_tag: Optional[str] = None) -> DataFrame:
    """Read input as a wide DataFrame with no schema assumptions.

    For delimited formats every physical column becomes one logical cell
    column. For free-text inputs the line is split on ``text_split_regex``
    so we still get per-token cells.
    """
    fmt = cfg.input_format.lower()
    if fmt in ("csv", "tsv"):
        delim = "\t" if fmt == "tsv" else cfg.input_delimiter
        df = (
            spark.read.option("header", str(cfg.input_has_header).lower())
                      .option("sep", delim)
                      .option("mode", "PERMISSIVE")
                      .option("multiLine", "false")
                      .option("ignoreLeadingWhiteSpace", "false")
                      .option("ignoreTrailingWhiteSpace", "false")
                      .csv(path)
        )
    elif fmt == "parquet":
        df = spark.read.parquet(path)
    elif fmt == "text":
        raw = spark.read.text(path).withColumnRenamed("value", "_line")
        if cfg.text_split_regex:
            df = raw.select(F.split(F.col("_line"), cfg.text_split_regex).alias("_cells"))
        else:
            df = raw.select(F.array(F.col("_line")).alias("_cells"))
    else:
        raise ValueError(f"Unsupported input_format: {cfg.input_format}")

    # Stamp a stable, monotonic record id BEFORE explosion so all cells
    # from the same physical row inherit the same lineage id.
    df = df.withColumn("record_id", F.monotonically_increasing_id())
    if source_tag is not None:
        df = df.withColumn("_source", F.lit(source_tag))
    else:
        df = df.withColumn("_source", F.input_file_name())
    return df


def extract_ref_id_table(raw: DataFrame, cfg: PipelineConfig,
                         column_index: int = 0) -> DataFrame:
    """Build a ``(record_id, ref_id)`` mapping from the un-flattened raw rows.

    Used by ``pipeline.metrics`` to align the pipeline's internal
    ``record_id`` (a monotonic id stamped in ``read_raw``) with the
    refIDs in a ground-truth file. The mapping is taken from the same
    DataFrame used downstream so the ids match exactly.

    For text inputs the ref_id is the ``column_index``-th element of
    the split cells array. For CSV / TSV / parquet it is the
    ``column_index``-th physical data column (skipping the internal
    ``record_id`` / ``_source`` markers). Independent of
    ``cfg.skip_leading_columns`` so the same column can simultaneously
    be captured for metrics and dropped from clustering.
    """
    if "_cells" in raw.columns:
        ref_col = F.col("_cells").getItem(column_index)
    else:
        data_cols = [c for c in raw.columns if c not in ("record_id", "_source")]
        if column_index >= len(data_cols):
            raise ValueError(
                f"ref_id column_index={column_index} is out of range; "
                f"raw has {len(data_cols)} data columns: {data_cols}"
            )
        ref_col = F.col(data_cols[column_index]).cast(T.StringType())

    return (
        raw.select(
            F.col("record_id").cast(T.LongType()),
            ref_col.alias("ref_id"),
        )
        .where(F.col("ref_id").isNotNull() & (F.length(F.col("ref_id")) > 0))
    )


def flatten(df: DataFrame, cfg: Optional[PipelineConfig] = None) -> DataFrame:
    """Explode any wide row into one cell per (record_id, col_index).

    If ``cfg.skip_leading_columns`` is set, that many leading data columns
    are dropped before explosion. The internal ``record_id`` is preserved.
    """
    skip = int(cfg.skip_leading_columns) if cfg is not None else 0

    if "_cells" in df.columns:
        cells_arr = F.col("_cells")
        if skip > 0:
            # slice is 1-indexed in Spark SQL.
            cells_arr = F.expr(f"slice(_cells, {skip + 1}, size(_cells))")
        cells = df.select(
            F.col("record_id"),
            F.col("_source").alias("source"),
            F.posexplode(cells_arr).alias("col_index", "value"),
        )
    else:
        data_cols = [c for c in df.columns if c not in ("record_id", "_source")]
        if skip > 0:
            data_cols = data_cols[skip:]
        if not data_cols:
            raise ValueError(
                f"skip_leading_columns={skip} drops every input column; "
                "nothing left to cluster."
            )
        # Build an array column preserving original column order, then
        # posexplode so we materialize one row per (record_id, col_index).
        arr = F.array(*[F.col(c).cast(T.StringType()) for c in data_cols])
        cells = (
            df.select(
                F.col("record_id"),
                F.col("_source").alias("source"),
                arr.alias("_cells"),
            )
            .select(
                F.col("record_id"),
                F.col("source"),
                F.posexplode("_cells").alias("col_index", "value"),
            )
        )

    return (
        cells
        .where(F.col("value").isNotNull() & (F.length(F.col("value")) > 0))
        .select(
            F.col("record_id").cast(T.LongType()),
            F.col("source"),
            F.col("col_index").cast(T.IntegerType()),
            F.col("value").cast(T.StringType()),
        )
    )
