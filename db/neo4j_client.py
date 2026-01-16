"""
BIBLOS v2 - Neo4j Client (Deprecated Shim)

DEPRECATED: This module has been superseded by db.neo4j_optimized.
Please use db.neo4j_optimized or import from db directly.

All imports are re-exported from the optimized module for backwards compatibility.
This shim will be removed in a future version.
"""
import warnings

warnings.warn(
    "db.neo4j_client is deprecated. Use db.neo4j_optimized or 'from db import Neo4jClient' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the optimized module
from db.neo4j_optimized import (
    Neo4jClient,
    Neo4jConfig,
    GraphNode,
    GraphRelationship,
    QueryResult,
)

# Re-export exception types if available
try:
    from neo4j.exceptions import ServiceUnavailable, AuthError
except ImportError:
    ServiceUnavailable = Exception
    AuthError = Exception

__all__ = [
    "Neo4jClient",
    "Neo4jConfig",
    "GraphNode",
    "GraphRelationship",
    "QueryResult",
    "ServiceUnavailable",
    "AuthError",
]
