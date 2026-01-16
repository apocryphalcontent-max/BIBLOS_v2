"""
BIBLOS v2 - API Security Module

Provides comprehensive security components:
- Authentication (API key, JWT, OAuth2)
- Rate limiting with Redis backend
- Secure CORS configuration
- Request validation
- Security headers

All components integrate with OpenTelemetry for audit logging.
"""

from api.security.auth import (
    AuthConfig,
    AuthProvider,
    APIKeyAuth,
    JWTAuth,
    get_current_user,
    require_auth,
    require_scope,
)
from api.security.rate_limit import (
    RateLimiter,
    RateLimitConfig,
    RateLimitMiddleware,
    rate_limit,
)
from api.security.cors import (
    CORSConfig,
    get_cors_middleware,
)
from api.security.headers import (
    SecurityHeadersMiddleware,
    get_security_headers,
)

__all__ = [
    # Auth
    "AuthConfig",
    "AuthProvider",
    "APIKeyAuth",
    "JWTAuth",
    "get_current_user",
    "require_auth",
    "require_scope",
    # Rate limiting
    "RateLimiter",
    "RateLimitConfig",
    "RateLimitMiddleware",
    "rate_limit",
    # CORS
    "CORSConfig",
    "get_cors_middleware",
    # Headers
    "SecurityHeadersMiddleware",
    "get_security_headers",
]
