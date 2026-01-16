"""
BIBLOS v2 - Intertextual Pipeline Phase (Deprecated Shim)

DEPRECATED: This module has been moved to pipeline.phases.intertextual.
Please import from pipeline.phases.intertextual instead.
"""
import warnings

warnings.warn(
    "pipeline.intertextual is deprecated. Use pipeline.phases.intertextual instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from canonical location
from pipeline.phases.intertextual import IntertextualPhase

__all__ = ["IntertextualPhase"]
