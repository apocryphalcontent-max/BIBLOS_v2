"""
BIBLOS v2 - Security Headers Middleware

Adds security headers to all responses:
- Content-Security-Policy
- X-Content-Type-Options
- X-Frame-Options
- X-XSS-Protection
- Strict-Transport-Security
- Referrer-Policy
- Permissions-Policy

Configurable per environment (development vs production).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


@dataclass
class SecurityHeadersConfig:
    """Configuration for security headers."""

    # Environment (affects header strictness)
    environment: str = field(default_factory=lambda: os.environ.get("ENV", "development"))

    # Content Security Policy
    csp_enabled: bool = True
    csp_report_only: bool = False
    csp_directives: Dict[str, str] = field(default_factory=lambda: {
        "default-src": "'self'",
        "script-src": "'self'",
        "style-src": "'self' 'unsafe-inline'",
        "img-src": "'self' data: https:",
        "font-src": "'self'",
        "connect-src": "'self'",
        "frame-ancestors": "'none'",
        "base-uri": "'self'",
        "form-action": "'self'",
    })

    # Strict-Transport-Security
    hsts_enabled: bool = True
    hsts_max_age: int = 31536000  # 1 year
    hsts_include_subdomains: bool = True
    hsts_preload: bool = False

    # X-Frame-Options
    frame_options: str = "DENY"  # DENY, SAMEORIGIN, or ALLOW-FROM uri

    # X-Content-Type-Options
    content_type_nosniff: bool = True

    # X-XSS-Protection (legacy, but still useful)
    xss_protection: str = "1; mode=block"

    # Referrer-Policy
    referrer_policy: str = "strict-origin-when-cross-origin"

    # Permissions-Policy (formerly Feature-Policy)
    permissions_policy: Dict[str, str] = field(default_factory=lambda: {
        "camera": "()",
        "microphone": "()",
        "geolocation": "()",
        "payment": "()",
    })

    # Cache-Control for sensitive responses
    cache_control: str = "no-store, no-cache, must-revalidate, private"

    # Paths to exclude from security headers
    exclude_paths: List[str] = field(default_factory=lambda: [
        "/docs",
        "/redoc",
        "/openapi.json",
    ])


def get_security_headers(config: Optional[SecurityHeadersConfig] = None) -> Dict[str, str]:
    """
    Generate security headers dictionary.

    Usage:
        headers = get_security_headers()
        response.headers.update(headers)
    """
    cfg = config or SecurityHeadersConfig()
    headers: Dict[str, str] = {}

    # Content-Security-Policy
    if cfg.csp_enabled:
        csp_value = "; ".join(
            f"{directive} {value}"
            for directive, value in cfg.csp_directives.items()
        )
        header_name = "Content-Security-Policy"
        if cfg.csp_report_only:
            header_name = "Content-Security-Policy-Report-Only"
        headers[header_name] = csp_value

    # Strict-Transport-Security (only in production)
    if cfg.hsts_enabled and cfg.environment == "production":
        hsts_value = f"max-age={cfg.hsts_max_age}"
        if cfg.hsts_include_subdomains:
            hsts_value += "; includeSubDomains"
        if cfg.hsts_preload:
            hsts_value += "; preload"
        headers["Strict-Transport-Security"] = hsts_value

    # X-Frame-Options
    if cfg.frame_options:
        headers["X-Frame-Options"] = cfg.frame_options

    # X-Content-Type-Options
    if cfg.content_type_nosniff:
        headers["X-Content-Type-Options"] = "nosniff"

    # X-XSS-Protection
    if cfg.xss_protection:
        headers["X-XSS-Protection"] = cfg.xss_protection

    # Referrer-Policy
    if cfg.referrer_policy:
        headers["Referrer-Policy"] = cfg.referrer_policy

    # Permissions-Policy
    if cfg.permissions_policy:
        permissions_value = ", ".join(
            f"{feature}={value}"
            for feature, value in cfg.permissions_policy.items()
        )
        headers["Permissions-Policy"] = permissions_value

    # Cache-Control
    if cfg.cache_control:
        headers["Cache-Control"] = cfg.cache_control

    return headers


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to all responses.

    Usage:
        app = FastAPI()
        app.add_middleware(SecurityHeadersMiddleware)

        # With custom config
        app.add_middleware(
            SecurityHeadersMiddleware,
            config=SecurityHeadersConfig(environment="production")
        )
    """

    def __init__(self, app, config: Optional[SecurityHeadersConfig] = None):
        super().__init__(app)
        self.config = config or SecurityHeadersConfig()
        self._headers = get_security_headers(self.config)

    async def dispatch(self, request: Request, call_next) -> Response:
        """Add security headers to response."""
        response = await call_next(request)

        # Skip excluded paths (like docs)
        if request.url.path in self.config.exclude_paths:
            return response

        # Add security headers
        for key, value in self._headers.items():
            # Don't override existing headers
            if key not in response.headers:
                response.headers[key] = value

        return response


def create_csp_nonce() -> str:
    """
    Create a CSP nonce for inline scripts.

    Usage:
        nonce = create_csp_nonce()
        # Add to CSP: script-src 'nonce-{nonce}'
        # Use in HTML: <script nonce="{nonce}">...</script>
    """
    import secrets
    import base64

    random_bytes = secrets.token_bytes(16)
    return base64.b64encode(random_bytes).decode("utf-8")
