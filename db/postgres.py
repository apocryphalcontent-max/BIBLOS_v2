"""
BIBLOS v2 - PostgreSQL Client (Deprecated Shim)

DEPRECATED: This module has been superseded by db.postgres_optimized.
Please use db.postgres_optimized or import from db directly.

All imports are re-exported from the optimized module for backwards compatibility.
This shim will be removed in a future version.
"""
import warnings

warnings.warn(
    "db.postgres is deprecated. Use db.postgres_optimized or 'from db import PostgresClient' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the optimized module
from db.postgres_optimized import (
    PostgresClient,
    get_db_session,
    get_db_client,
    CacheConfig,
    chunked,
)

__all__ = [
    "PostgresClient",
    "get_db_session",
    "get_db_client",
    "CacheConfig",
    "chunked",
]
