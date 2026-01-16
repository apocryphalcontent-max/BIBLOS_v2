"""
BIBLOS v2 - SQLAlchemy ORM Models (Deprecated Shim)

DEPRECATED: This module has been superseded by db.models_optimized.
Please use db.models_optimized or import from db directly.

All imports are re-exported from the optimized module for backwards compatibility.
This shim will be removed in a future version.

The optimized module includes:
- Composite indexes for common query patterns
- pgvector integration for embeddings
- GIN indexes for JSONB columns
- Optimized relationship loading strategies
"""
import warnings

warnings.warn(
    "db.models is deprecated. Use db.models_optimized or 'from db import Base, Book, Verse, ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the optimized module
from db.models_optimized import (
    Base,
    Book,
    Verse,
    CrossReference,
    PatristicCitation,
    ExtractionResult,
    Testament,
    ConnectionType,
    StrengthLevel,
)

__all__ = [
    "Base",
    "Book",
    "Verse",
    "CrossReference",
    "PatristicCitation",
    "ExtractionResult",
    "Testament",
    "ConnectionType",
    "StrengthLevel",
]
