"""Schema-free, syntactic-pattern entity clustering pipeline.

Distributed PySpark implementation of a blocking-first entity resolution
pipeline that operates without schema, type, or semantic assumptions.
Stages mirror the project pseudocode: flatten -> normalize -> feature
extraction -> pattern signature -> multi-key blocking -> per-block local
clustering -> distributed union-find merge -> quality filtering.
"""

from pipeline.config import PipelineConfig
from pipeline.pipeline import run_pipeline

__all__ = ["PipelineConfig", "run_pipeline"]
