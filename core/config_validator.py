"""
BIBLOS v2 - Configuration Validation

Provides comprehensive configuration validation:
- Schema-based validation
- Environment variable checking
- Type coercion and defaults
- Validation result reporting

Ensures all configuration is valid before system startup.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)

from core.errors import BiblosConfigError, ErrorContext, ErrorSeverity

T = TypeVar("T")


class ValidationLevel(Enum):
    """Validation result severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Single validation issue."""

    level: ValidationLevel
    message: str
    field: Optional[str] = None
    expected: Optional[str] = None
    actual: Optional[str] = None
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "message": self.message,
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
            "suggestion": self.suggestion,
        }


@dataclass
class ValidationResult:
    """Result of configuration validation."""

    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    config_name: Optional[str] = None

    def add_issue(
        self,
        level: ValidationLevel,
        message: str,
        **kwargs: Any,
    ) -> None:
        """Add a validation issue."""
        issue = ValidationIssue(level=level, message=message, **kwargs)
        if level in (ValidationLevel.ERROR, ValidationLevel.CRITICAL):
            self.issues.append(issue)
            self.valid = False
        else:
            self.warnings.append(issue)

    def add_error(self, message: str, **kwargs: Any) -> None:
        """Add an error issue."""
        self.add_issue(ValidationLevel.ERROR, message, **kwargs)

    def add_warning(self, message: str, **kwargs: Any) -> None:
        """Add a warning issue."""
        self.add_issue(ValidationLevel.WARNING, message, **kwargs)

    def merge(self, other: "ValidationResult") -> None:
        """Merge another validation result."""
        self.issues.extend(other.issues)
        self.warnings.extend(other.warnings)
        if not other.valid:
            self.valid = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "config_name": self.config_name,
            "issues": [i.to_dict() for i in self.issues],
            "warnings": [w.to_dict() for w in self.warnings],
        }

    def raise_if_invalid(self) -> None:
        """Raise BiblosConfigError if validation failed."""
        if not self.valid:
            messages = [issue.message for issue in self.issues]
            raise BiblosConfigError(
                message=f"Configuration validation failed: {'; '.join(messages)}",
                context=ErrorContext.from_current_span(
                    operation="config_validation",
                    component=self.config_name or "config",
                ),
            )


@dataclass
class FieldValidator:
    """Validator for a single configuration field."""

    name: str
    required: bool = True
    field_type: Optional[Type] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[Set[Any]] = None
    pattern: Optional[str] = None
    custom_validator: Optional[Callable[[Any], bool]] = None
    default: Any = None
    env_var: Optional[str] = None
    description: Optional[str] = None

    def validate(self, value: Any, result: ValidationResult) -> Any:
        """Validate a field value and return the validated/coerced value."""
        # Check for missing required field
        if value is None:
            if self.env_var:
                value = os.environ.get(self.env_var)

            if value is None and self.default is not None:
                return self.default

            if value is None and self.required:
                result.add_error(
                    f"Required field '{self.name}' is missing",
                    field=self.name,
                    suggestion=f"Set the {self.env_var or self.name} environment variable" if self.env_var else None,
                )
                return None

            if value is None:
                return None

        # Type coercion and validation
        if self.field_type:
            try:
                if self.field_type == bool and isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes", "on")
                elif self.field_type in (int, float) and isinstance(value, str):
                    value = self.field_type(value)
                elif not isinstance(value, self.field_type):
                    value = self.field_type(value)
            except (ValueError, TypeError):
                result.add_error(
                    f"Field '{self.name}' has invalid type",
                    field=self.name,
                    expected=self.field_type.__name__,
                    actual=type(value).__name__,
                )
                return value

        # Range validation
        if self.min_value is not None and value < self.min_value:
            result.add_error(
                f"Field '{self.name}' is below minimum value",
                field=self.name,
                expected=f">= {self.min_value}",
                actual=str(value),
            )

        if self.max_value is not None and value > self.max_value:
            result.add_error(
                f"Field '{self.name}' exceeds maximum value",
                field=self.name,
                expected=f"<= {self.max_value}",
                actual=str(value),
            )

        # Allowed values validation
        if self.allowed_values is not None and value not in self.allowed_values:
            result.add_error(
                f"Field '{self.name}' has invalid value",
                field=self.name,
                expected=f"one of {self.allowed_values}",
                actual=str(value),
            )

        # Pattern validation
        if self.pattern is not None and isinstance(value, str):
            if not re.match(self.pattern, value):
                result.add_error(
                    f"Field '{self.name}' does not match required pattern",
                    field=self.name,
                    expected=f"match pattern '{self.pattern}'",
                    actual=value,
                )

        # Custom validation
        if self.custom_validator is not None:
            if not self.custom_validator(value):
                result.add_error(
                    f"Field '{self.name}' failed custom validation",
                    field=self.name,
                    actual=str(value),
                )

        return value


class ConfigValidator:
    """
    Comprehensive configuration validator.

    Usage:
        validator = ConfigValidator("DatabaseConfig")
        validator.add_field("host", required=True, field_type=str)
        validator.add_field("port", required=True, field_type=int, min_value=1, max_value=65535)

        result = validator.validate(config_dict)
        result.raise_if_invalid()
    """

    def __init__(self, name: str):
        self.name = name
        self._fields: Dict[str, FieldValidator] = {}
        self._dependencies: Dict[str, List[str]] = {}

    def add_field(
        self,
        name: str,
        required: bool = True,
        field_type: Optional[Type] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allowed_values: Optional[Set[Any]] = None,
        pattern: Optional[str] = None,
        custom_validator: Optional[Callable[[Any], bool]] = None,
        default: Any = None,
        env_var: Optional[str] = None,
        description: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
    ) -> "ConfigValidator":
        """Add a field validator."""
        self._fields[name] = FieldValidator(
            name=name,
            required=required,
            field_type=field_type,
            min_value=min_value,
            max_value=max_value,
            allowed_values=allowed_values,
            pattern=pattern,
            custom_validator=custom_validator,
            default=default,
            env_var=env_var,
            description=description,
        )

        if depends_on:
            self._dependencies[name] = depends_on

        return self

    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate a configuration dictionary."""
        result = ValidationResult(valid=True, config_name=self.name)
        validated_config: Dict[str, Any] = {}

        # Validate each field
        for field_name, validator in self._fields.items():
            value = config.get(field_name)
            validated_value = validator.validate(value, result)
            validated_config[field_name] = validated_value

        # Check dependencies
        for field_name, deps in self._dependencies.items():
            if validated_config.get(field_name) is not None:
                for dep in deps:
                    if validated_config.get(dep) is None:
                        result.add_error(
                            f"Field '{field_name}' requires '{dep}' to be set",
                            field=field_name,
                        )

        return result

    def validate_object(self, obj: Any) -> ValidationResult:
        """Validate a dataclass or object with attributes."""
        config = {}
        for field_name in self._fields:
            if hasattr(obj, field_name):
                config[field_name] = getattr(obj, field_name)
        return self.validate(config)


def validate_config(
    config: Any,
    schema: Optional[Dict[str, FieldValidator]] = None,
    name: str = "config",
) -> ValidationResult:
    """
    Convenience function to validate configuration.

    Usage:
        result = validate_config(my_config, {
            "host": FieldValidator("host", required=True),
            "port": FieldValidator("port", field_type=int, min_value=1),
        })
    """
    validator = ConfigValidator(name)

    if schema:
        for field_name, field_validator in schema.items():
            validator._fields[field_name] = field_validator

    if isinstance(config, dict):
        return validator.validate(config)
    return validator.validate_object(config)


def require_env(
    var_name: str,
    field_type: Type[T] = str,  # type: ignore
    default: Optional[T] = None,
    allowed_values: Optional[Set[T]] = None,
) -> T:
    """
    Require an environment variable with validation.

    Usage:
        api_key = require_env("API_KEY")
        port = require_env("PORT", int, default=8000)
    """
    value = os.environ.get(var_name)

    if value is None:
        if default is not None:
            return default
        raise BiblosConfigError(
            message=f"Required environment variable '{var_name}' is not set",
            config_key=var_name,
        )

    try:
        if field_type == bool:
            typed_value = value.lower() in ("true", "1", "yes", "on")  # type: ignore
        else:
            typed_value = field_type(value)  # type: ignore
    except (ValueError, TypeError) as e:
        raise BiblosConfigError(
            message=f"Environment variable '{var_name}' has invalid type",
            config_key=var_name,
            expected_type=field_type,
            actual_value=value,
            cause=e,
        )

    if allowed_values and typed_value not in allowed_values:
        raise BiblosConfigError(
            message=f"Environment variable '{var_name}' has invalid value",
            config_key=var_name,
            actual_value=typed_value,
        )

    return typed_value  # type: ignore


# Pre-built validators for common BIBLOS configurations

def create_database_validator() -> ConfigValidator:
    """Create validator for database configuration."""
    return (
        ConfigValidator("DatabaseConfig")
        .add_field("host", field_type=str, env_var="DB_HOST", default="localhost")
        .add_field("port", field_type=int, min_value=1, max_value=65535, env_var="DB_PORT", default=5432)
        .add_field("database", required=True, field_type=str, env_var="DB_NAME")
        .add_field("user", required=True, field_type=str, env_var="DB_USER")
        .add_field("password", required=True, field_type=str, env_var="DB_PASSWORD")
        .add_field("pool_size", field_type=int, min_value=1, max_value=100, default=10)
        .add_field("max_overflow", field_type=int, min_value=0, max_value=100, default=20)
    )


def create_ml_validator() -> ConfigValidator:
    """Create validator for ML configuration."""
    return (
        ConfigValidator("MLConfig")
        .add_field("device", field_type=str, allowed_values={"cpu", "cuda", "mps"}, default="cpu")
        .add_field("batch_size", field_type=int, min_value=1, max_value=1024, default=32)
        .add_field("embedding_model", field_type=str, default="sentence-transformers/all-mpnet-base-v2")
        .add_field("cache_embeddings", field_type=bool, default=True)
        .add_field("min_confidence", field_type=float, min_value=0.0, max_value=1.0, default=0.5)
    )


def create_api_validator() -> ConfigValidator:
    """Create validator for API configuration."""
    return (
        ConfigValidator("APIConfig")
        .add_field("host", field_type=str, default="0.0.0.0")
        .add_field("port", field_type=int, min_value=1, max_value=65535, default=8000)
        .add_field("workers", field_type=int, min_value=1, max_value=32, default=4)
        .add_field("cors_origins", field_type=list, default=["*"])
        .add_field("rate_limit", field_type=int, min_value=0, default=100)
        .add_field("enable_docs", field_type=bool, default=True)
    )
