"""
BIBLOS v2 - Validation Pipeline Phase (Deprecated Shim)

DEPRECATED: This module has been moved to pipeline.phases.validation.
Please import from pipeline.phases.validation instead.
"""
import warnings

warnings.warn(
    "pipeline.validation is deprecated. Use pipeline.phases.validation instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from canonical location
from pipeline.phases.validation import ValidationPhase

__all__ = ["ValidationPhase"]
