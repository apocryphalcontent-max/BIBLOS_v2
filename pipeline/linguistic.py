"""
BIBLOS v2 - Linguistic Pipeline Phase (Deprecated Shim)

DEPRECATED: This module has been moved to pipeline.phases.linguistic.
Please import from pipeline.phases.linguistic instead.
"""
import warnings

warnings.warn(
    "pipeline.linguistic is deprecated. Use pipeline.phases.linguistic instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from canonical location
from pipeline.phases.linguistic import LinguisticPhase

__all__ = ["LinguisticPhase"]
