"""
BIBLOS v2 - Theological Pipeline Phase (Deprecated Shim)

DEPRECATED: This module has been moved to pipeline.phases.theological.
Please import from pipeline.phases.theological instead.
"""
import warnings

warnings.warn(
    "pipeline.theological is deprecated. Use pipeline.phases.theological instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from canonical location
from pipeline.phases.theological import TheologicalPhase

__all__ = ["TheologicalPhase"]
