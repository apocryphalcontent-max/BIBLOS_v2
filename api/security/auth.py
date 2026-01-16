"""
BIBLOS v2 - Authentication Module

Provides multiple authentication strategies:
- API Key authentication
- JWT token authentication
- OAuth2 integration (optional)

Features:
- Configurable providers
- Scope-based authorization
- Audit logging with OpenTelemetry
- Secure credential handling
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
)
import functools

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials

from opentelemetry import trace

tracer = trace.get_tracer(__name__)


class AuthError(Exception):
    """Authentication/authorization error."""

    def __init__(self, message: str, status_code: int = 401):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass
class User:
    """Authenticated user representation."""

    id: str
    name: Optional[str] = None
    email: Optional[str] = None
    scopes: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    authenticated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def has_scope(self, scope: str) -> bool:
        """Check if user has a specific scope."""
        return scope in self.scopes or "*" in self.scopes

    def has_any_scope(self, scopes: List[str]) -> bool:
        """Check if user has any of the specified scopes."""
        return any(self.has_scope(s) for s in scopes)


@dataclass
class AuthConfig:
    """Authentication configuration."""

    # API Key settings
    api_key_header: str = "X-API-Key"
    api_key_env_var: str = "BIBLOS_API_KEY"
    api_keys: Dict[str, Set[str]] = field(default_factory=dict)  # key -> scopes

    # JWT settings
    jwt_secret_env_var: str = "JWT_SECRET"
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24

    # General settings
    enabled: bool = True
    public_endpoints: Set[str] = field(default_factory=lambda: {
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/metrics",
    })


class AuthProvider(ABC):
    """Abstract base class for authentication providers."""

    @abstractmethod
    async def authenticate(self, request: Request) -> Optional[User]:
        """Authenticate a request and return user or None."""
        pass

    @abstractmethod
    def get_credentials(self, request: Request) -> Optional[str]:
        """Extract credentials from request."""
        pass


class APIKeyAuth(AuthProvider):
    """API Key authentication provider."""

    def __init__(self, config: AuthConfig):
        self.config = config
        self._header = APIKeyHeader(name=config.api_key_header, auto_error=False)
        self._load_api_keys()

    def _load_api_keys(self) -> None:
        """Load API keys from environment variables."""
        # Load primary API key
        primary_key = os.environ.get(self.config.api_key_env_var)
        if primary_key:
            self.config.api_keys[self._hash_key(primary_key)] = {"*"}

        # Load additional keys from BIBLOS_API_KEYS (comma-separated)
        additional_keys = os.environ.get("BIBLOS_API_KEYS", "")
        for key in additional_keys.split(","):
            key = key.strip()
            if key:
                # Format: key:scope1,scope2 or just key (all scopes)
                if ":" in key:
                    api_key, scopes_str = key.split(":", 1)
                    scopes = set(s.strip() for s in scopes_str.split(";"))
                else:
                    api_key = key
                    scopes = {"read"}  # Default to read-only
                self.config.api_keys[self._hash_key(api_key)] = scopes

    def _hash_key(self, key: str) -> str:
        """Hash an API key for secure storage."""
        return hashlib.sha256(key.encode()).hexdigest()

    def get_credentials(self, request: Request) -> Optional[str]:
        """Extract API key from request header."""
        return request.headers.get(self.config.api_key_header)

    async def authenticate(self, request: Request) -> Optional[User]:
        """Authenticate using API key."""
        api_key = self.get_credentials(request)
        if not api_key:
            return None

        key_hash = self._hash_key(api_key)
        scopes = self.config.api_keys.get(key_hash)

        if scopes is None:
            with tracer.start_as_current_span("auth.api_key.failed") as span:
                span.set_attribute("auth.method", "api_key")
                span.set_attribute("auth.success", False)
                # Don't log the actual key for security
                span.set_attribute("auth.key_prefix", api_key[:8] + "..." if len(api_key) > 8 else "***")
            return None

        with tracer.start_as_current_span("auth.api_key.success") as span:
            span.set_attribute("auth.method", "api_key")
            span.set_attribute("auth.success", True)
            span.set_attribute("auth.scopes", list(scopes))

        return User(
            id=f"api_key:{key_hash[:16]}",
            name="API Key User",
            scopes=scopes,
            metadata={"auth_method": "api_key"},
        )


class JWTAuth(AuthProvider):
    """JWT token authentication provider."""

    def __init__(self, config: AuthConfig):
        self.config = config
        self._bearer = HTTPBearer(auto_error=False)
        self._secret = os.environ.get(config.jwt_secret_env_var, "")

        if not self._secret:
            import warnings
            warnings.warn(
                f"JWT secret not configured. Set {config.jwt_secret_env_var} environment variable.",
                UserWarning
            )

    def get_credentials(self, request: Request) -> Optional[str]:
        """Extract JWT from Authorization header."""
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:]
        return None

    async def authenticate(self, request: Request) -> Optional[User]:
        """Authenticate using JWT token."""
        token = self.get_credentials(request)
        if not token or not self._secret:
            return None

        try:
            import jwt

            payload = jwt.decode(
                token,
                self._secret,
                algorithms=[self.config.jwt_algorithm],
            )

            with tracer.start_as_current_span("auth.jwt.success") as span:
                span.set_attribute("auth.method", "jwt")
                span.set_attribute("auth.success", True)
                span.set_attribute("auth.user_id", payload.get("sub", "unknown"))

            return User(
                id=payload.get("sub", "unknown"),
                name=payload.get("name"),
                email=payload.get("email"),
                scopes=set(payload.get("scopes", [])),
                metadata={
                    "auth_method": "jwt",
                    "issued_at": payload.get("iat"),
                    "expires_at": payload.get("exp"),
                },
            )

        except Exception as e:
            with tracer.start_as_current_span("auth.jwt.failed") as span:
                span.set_attribute("auth.method", "jwt")
                span.set_attribute("auth.success", False)
                span.set_attribute("auth.error", str(e))
            return None

    def generate_token(
        self,
        user_id: str,
        scopes: List[str],
        expiry_hours: Optional[int] = None,
        **extra_claims: Any,
    ) -> str:
        """Generate a JWT token."""
        import jwt

        now = datetime.now(timezone.utc)
        expiry = now + timedelta(hours=expiry_hours or self.config.jwt_expiry_hours)

        payload = {
            "sub": user_id,
            "iat": int(now.timestamp()),
            "exp": int(expiry.timestamp()),
            "scopes": scopes,
            **extra_claims,
        }

        return jwt.encode(payload, self._secret, algorithm=self.config.jwt_algorithm)


class CompositeAuthProvider(AuthProvider):
    """Combines multiple authentication providers."""

    def __init__(self, providers: List[AuthProvider]):
        self.providers = providers

    def get_credentials(self, request: Request) -> Optional[str]:
        """Try to get credentials from any provider."""
        for provider in self.providers:
            creds = provider.get_credentials(request)
            if creds:
                return creds
        return None

    async def authenticate(self, request: Request) -> Optional[User]:
        """Try to authenticate with each provider in order."""
        for provider in self.providers:
            user = await provider.authenticate(request)
            if user:
                return user
        return None


# Global auth configuration and provider
_auth_config: Optional[AuthConfig] = None
_auth_provider: Optional[AuthProvider] = None


def configure_auth(config: Optional[AuthConfig] = None) -> None:
    """Configure the global authentication provider."""
    global _auth_config, _auth_provider

    _auth_config = config or AuthConfig()

    providers: List[AuthProvider] = [
        APIKeyAuth(_auth_config),
    ]

    # Add JWT if secret is configured
    if os.environ.get(_auth_config.jwt_secret_env_var):
        providers.append(JWTAuth(_auth_config))

    _auth_provider = CompositeAuthProvider(providers)


async def get_current_user(request: Request) -> Optional[User]:
    """
    FastAPI dependency to get the current authenticated user.

    Returns None if not authenticated (for optional auth).
    """
    global _auth_config, _auth_provider

    if _auth_config is None:
        configure_auth()

    # Skip auth for public endpoints
    if request.url.path in _auth_config.public_endpoints:
        return None

    # Skip auth if disabled
    if not _auth_config.enabled:
        return User(id="anonymous", scopes={"*"})

    if _auth_provider is None:
        return None

    return await _auth_provider.authenticate(request)


async def require_auth(request: Request) -> User:
    """
    FastAPI dependency that requires authentication.

    Raises HTTPException if not authenticated.
    """
    user = await get_current_user(request)

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer, API-Key"},
        )

    return user


def require_scope(*required_scopes: str) -> Callable:
    """
    Decorator/dependency that requires specific scopes.

    Usage:
        @app.get("/admin")
        async def admin_endpoint(user: User = Depends(require_scope("admin"))):
            ...
    """
    async def dependency(user: User = Depends(require_auth)) -> User:
        if not user.has_any_scope(list(required_scopes)):
            with tracer.start_as_current_span("auth.scope.denied") as span:
                span.set_attribute("auth.required_scopes", list(required_scopes))
                span.set_attribute("auth.user_scopes", list(user.scopes))

            raise HTTPException(
                status_code=403,
                detail=f"Required scope(s): {', '.join(required_scopes)}",
            )

        return user

    return dependency


def generate_api_key() -> str:
    """Generate a secure API key."""
    return secrets.token_urlsafe(32)
