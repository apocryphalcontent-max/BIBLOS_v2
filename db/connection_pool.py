"""
BIBLOS v2 - Unified Connection Manager (Deprecated Shim)

DEPRECATED: This module has been superseded by db.connection_pool_optimized.
Please use db.connection_pool_optimized or import from db directly.

All imports are re-exported from the optimized module for backwards compatibility.
This shim will be removed in a future version.

The optimized module includes:
- Circuit breaker pattern for fault tolerance
- Connection pool warmup on startup
- Metrics tracking for connection utilization
- Graceful degradation for non-critical services
"""
import warnings

warnings.warn(
    "db.connection_pool is deprecated. Use db.connection_pool_optimized or 'from db import ConnectionManager' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the optimized module
from db.connection_pool_optimized import (
    ConnectionManager,
    ConnectionConfig,
    ConnectionState,
    ConnectionStatus,
    ConnectionMetrics,
    get_connection_manager,
    shutdown_connections,
)

__all__ = [
    "ConnectionManager",
    "ConnectionConfig",
    "ConnectionState",
    "ConnectionStatus",
    "ConnectionMetrics",
    "get_connection_manager",
    "shutdown_connections",
]
