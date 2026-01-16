"""
BIBLOS v2 - Secure CORS Configuration

Provides secure CORS configuration that:
- Never combines wildcard with credentials
- Validates origins against allowlist
- Supports environment-based configuration
- Logs CORS violations
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Pattern, Set, Union
from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from opentelemetry import trace

tracer = trace.get_tracer(__name__)


@dataclass
class CORSConfig:
    """
    Secure CORS configuration.

    IMPORTANT: Never use allow_origins=["*"] with allow_credentials=True
    This is a security vulnerability that allows any origin to make
    credentialed requests.
    """

    # Allowed origins (exact match)
    # Can be loaded from CORS_ORIGINS environment variable (comma-separated)
    allow_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ])

    # Origin patterns for regex matching
    # e.g., r"https://.*\.example\.com"
    allow_origin_patterns: List[str] = field(default_factory=list)

    # Whether to allow credentials (cookies, auth headers)
    allow_credentials: bool = True

    # Allowed HTTP methods
    allow_methods: List[str] = field(default_factory=lambda: [
        "GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"
    ])

    # Allowed headers
    allow_headers: List[str] = field(default_factory=lambda: [
        "Accept",
        "Accept-Language",
        "Content-Type",
        "Authorization",
        "X-API-Key",
        "X-Request-ID",
    ])

    # Headers exposed to the browser
    expose_headers: List[str] = field(default_factory=lambda: [
        "X-Request-ID",
        "X-Trace-ID",
        "X-Response-Time",
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
    ])

    # Max age for preflight cache (seconds)
    max_age: int = 600

    # Log CORS rejections
    log_rejections: bool = True

    def __post_init__(self):
        """Load origins from environment and validate configuration."""
        # Load from environment
        env_origins = os.environ.get("CORS_ORIGINS", "")
        if env_origins:
            self.allow_origins.extend(
                origin.strip()
                for origin in env_origins.split(",")
                if origin.strip()
            )

        # Remove duplicates while preserving order
        seen = set()
        unique_origins = []
        for origin in self.allow_origins:
            if origin not in seen:
                seen.add(origin)
                unique_origins.append(origin)
        self.allow_origins = unique_origins

        # Validate: no wildcard with credentials
        if "*" in self.allow_origins and self.allow_credentials:
            raise ValueError(
                "SECURITY ERROR: Cannot use wildcard origin ('*') with allow_credentials=True. "
                "This would allow any website to make credentialed requests. "
                "Either specify exact origins or disable credentials."
            )


class SecureCORSMiddleware(BaseHTTPMiddleware):
    """
    Secure CORS middleware with origin validation and logging.

    Unlike the default CORSMiddleware, this:
    - Validates origins against allowlist/patterns
    - Logs rejections for security monitoring
    - Never reflects arbitrary origins
    """

    def __init__(self, app, config: CORSConfig):
        super().__init__(app)
        self.config = config
        self._origin_set: Set[str] = set(config.allow_origins)
        self._patterns: List[Pattern] = [
            re.compile(p) for p in config.allow_origin_patterns
        ]

    def _is_allowed_origin(self, origin: str) -> bool:
        """Check if origin is allowed."""
        if not origin:
            return False

        # Exact match
        if origin in self._origin_set:
            return True

        # Pattern match
        for pattern in self._patterns:
            if pattern.match(origin):
                return True

        return False

    def _get_cors_headers(self, origin: str) -> dict:
        """Generate CORS headers for allowed origin."""
        headers = {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": ", ".join(self.config.allow_methods),
            "Access-Control-Allow-Headers": ", ".join(self.config.allow_headers),
            "Access-Control-Expose-Headers": ", ".join(self.config.expose_headers),
            "Access-Control-Max-Age": str(self.config.max_age),
        }

        if self.config.allow_credentials:
            headers["Access-Control-Allow-Credentials"] = "true"

        return headers

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with CORS handling."""
        origin = request.headers.get("origin")

        # Non-CORS request
        if not origin:
            return await call_next(request)

        # Check if origin is allowed
        if not self._is_allowed_origin(origin):
            if self.config.log_rejections:
                with tracer.start_as_current_span("cors.rejected") as span:
                    span.set_attribute("cors.origin", origin)
                    span.set_attribute("cors.path", request.url.path)
                    span.set_attribute("cors.method", request.method)

            # Return response without CORS headers (browser will reject)
            if request.method == "OPTIONS":
                return Response(status_code=403)
            return await call_next(request)

        # Handle preflight
        if request.method == "OPTIONS":
            headers = self._get_cors_headers(origin)
            headers["Vary"] = "Origin"
            return Response(status_code=204, headers=headers)

        # Handle actual request
        response = await call_next(request)

        # Add CORS headers
        cors_headers = self._get_cors_headers(origin)
        for key, value in cors_headers.items():
            response.headers[key] = value
        response.headers["Vary"] = "Origin"

        return response


def get_cors_middleware(config: Optional[CORSConfig] = None) -> SecureCORSMiddleware:
    """
    Factory function to create secure CORS middleware.

    Usage:
        app = FastAPI()
        cors_middleware = get_cors_middleware(CORSConfig(
            allow_origins=["https://myapp.com"],
            allow_credentials=True,
        ))
        # Add manually since it's a custom middleware

    Or use add_cors_middleware for convenience.
    """
    return SecureCORSMiddleware(app=None, config=config or CORSConfig())


def add_cors_middleware(app: FastAPI, config: Optional[CORSConfig] = None) -> None:
    """
    Add secure CORS middleware to a FastAPI application.

    Usage:
        app = FastAPI()
        add_cors_middleware(app, CORSConfig(
            allow_origins=["https://myapp.com"],
        ))
    """
    cfg = config or CORSConfig()

    # If no credentials required, can use standard middleware
    if not cfg.allow_credentials and "*" in cfg.allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=cfg.allow_methods,
            allow_headers=cfg.allow_headers,
            expose_headers=cfg.expose_headers,
            max_age=cfg.max_age,
        )
    else:
        # Use our secure middleware
        app.add_middleware(SecureCORSMiddleware, config=cfg)


def validate_origin(origin: str) -> bool:
    """
    Validate that an origin is well-formed.

    Returns True if origin is a valid URL with scheme and host.
    """
    try:
        parsed = urlparse(origin)
        return bool(parsed.scheme and parsed.netloc)
    except Exception:
        return False
