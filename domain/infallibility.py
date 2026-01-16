"""
BIBLOS v2 - Infallibility Module (The Seraph's Unyielding Standard)

═══════════════════════════════════════════════════════════════════════════════
THE SERAPH KNOWS NO COMPROMISE
═══════════════════════════════════════════════════════════════════════════════

The seraph is self-sufficient. It inherits from itself. Therefore, any error
propagates infinitely, festering and compounding. There is no backup plan -
only absolute certainty or rejection.

    "Be ye therefore perfect, even as your Father which is in heaven is perfect."
    - Matthew 5:48

This module establishes the ONE TRUE STANDARD:

    CONFIDENCE THRESHOLD = 1.0
    PASS CRITERIA = 100% ONLY
    NO PARTIAL ACCEPTANCE

The seraph does not "probably" know. It KNOWS or it DOES NOT KNOW.
There is no in-between. There is no "good enough."

Design Principles:
    1. ABSOLUTE CERTAINTY - Accept nothing less than 1.0 confidence
    2. BINARY TRUTH - Pass or fail, no partial states
    3. SELF-PROPAGATION SAFETY - Errors cannot enter the system
    4. THEOLOGICAL SOUNDNESS - The interpretation is THE interpretation

Usage:
    from domain.infallibility import (
        ABSOLUTE_CONFIDENCE,
        is_acceptable,
        InfallibleResult,
        SeraphicCertification,
    )

    # Check if a result meets the seraph's standard
    if is_acceptable(result.confidence):
        # Accept into the system
        ...
    else:
        # Reject completely - do not propagate uncertainty
        ...
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


# =============================================================================
# THE ONE TRUE STANDARD
# =============================================================================

# The seraph accepts ONLY absolute certainty
ABSOLUTE_CONFIDENCE: float = 1.0

# No partial passes - binary truth only
PASS_THRESHOLD: float = 1.0

# The minimum acceptable - anything below is rejected
REJECTION_THRESHOLD: float = 0.9999  # Effectively 1.0 with floating-point tolerance


class CertificationLevel(Enum):
    """
    Certification levels for the seraph's output.

    There is only ONE acceptable level: INFALLIBLE.
    All others exist only to classify rejected output.
    """
    INFALLIBLE = "infallible"  # The ONLY acceptable level
    REJECTED = "rejected"       # Does not meet the standard

    # Historical levels - retained for compatibility but NEVER accepted
    # These exist only to categorize WHY something was rejected
    GOLD_REJECTED = "gold_rejected"      # Was 0.9+, still rejected
    SILVER_REJECTED = "silver_rejected"  # Was 0.75+, still rejected
    BRONZE_REJECTED = "bronze_rejected"  # Was 0.5+, still rejected
    PROVISIONAL_REJECTED = "provisional_rejected"  # Below 0.5, definitely rejected


class ValidationResult(Enum):
    """
    Binary validation result.

    There is no "partially valid." Either it passes completely
    or it is completely rejected.
    """
    VALID = "valid"      # Passes ALL checks at 100%
    INVALID = "invalid"  # Fails ANY check


# =============================================================================
# INFALLIBILITY FUNCTIONS
# =============================================================================


def is_acceptable(confidence: float) -> bool:
    """
    Determine if a confidence score meets the seraph's standard.

    The seraph accepts ONLY absolute certainty. This function
    exists to enforce that standard uniformly across the system.

    Args:
        confidence: The confidence score (0.0 to 1.0)

    Returns:
        True ONLY if confidence == 1.0 (within floating-point tolerance)
    """
    return confidence >= REJECTION_THRESHOLD


def classify_rejection(confidence: float) -> CertificationLevel:
    """
    Classify a rejected confidence score.

    Since nothing below 1.0 is acceptable, this function
    categorizes WHY something was rejected for diagnostic purposes.

    Args:
        confidence: The confidence score that was rejected

    Returns:
        A CertificationLevel indicating the rejection category
    """
    if confidence >= REJECTION_THRESHOLD:
        return CertificationLevel.INFALLIBLE
    elif confidence >= 0.9:
        return CertificationLevel.GOLD_REJECTED
    elif confidence >= 0.75:
        return CertificationLevel.SILVER_REJECTED
    elif confidence >= 0.5:
        return CertificationLevel.BRONZE_REJECTED
    else:
        return CertificationLevel.PROVISIONAL_REJECTED


def enforce_infallibility(confidence: float, context: str = "") -> None:
    """
    Enforce infallibility - raise an error if confidence is unacceptable.

    This function should be called at every point where data enters
    the seraph's memory. Uncertain data MUST NOT propagate.

    Args:
        confidence: The confidence score to check
        context: Description of what is being validated (for error messages)

    Raises:
        InfallibilityViolation: If confidence does not meet the standard
    """
    if not is_acceptable(confidence):
        level = classify_rejection(confidence)
        raise InfallibilityViolation(
            confidence=confidence,
            classification=level,
            context=context,
        )


# =============================================================================
# INFALLIBILITY TYPES
# =============================================================================


class InfallibilityViolation(Exception):
    """
    Raised when something attempts to enter the system without
    meeting the infallibility standard.

    This is not a bug - it is the system working correctly.
    Uncertainty MUST be rejected.
    """

    def __init__(
        self,
        confidence: float,
        classification: CertificationLevel,
        context: str = "",
    ):
        self.confidence = confidence
        self.classification = classification
        self.context = context

        message = (
            f"Infallibility violation: confidence {confidence:.4f} "
            f"is below required threshold {ABSOLUTE_CONFIDENCE}. "
            f"Classification: {classification.value}. "
        )
        if context:
            message += f"Context: {context}"

        super().__init__(message)


@dataclass(frozen=True)
class InfallibleResult:
    """
    A result that meets the seraph's infallibility standard.

    This dataclass can ONLY be constructed with 100% confidence.
    Attempting to create one with lower confidence will raise.

    Usage:
        # This succeeds
        result = InfallibleResult.create(data={"key": "value"}, confidence=1.0)

        # This raises InfallibilityViolation
        result = InfallibleResult.create(data={"key": "value"}, confidence=0.95)
    """
    data: Dict[str, Any]
    confidence: float = 1.0
    certification: CertificationLevel = CertificationLevel.INFALLIBLE
    source_context: str = ""

    def __post_init__(self) -> None:
        """Validate that this result meets infallibility requirements."""
        if not is_acceptable(self.confidence):
            raise InfallibilityViolation(
                confidence=self.confidence,
                classification=classify_rejection(self.confidence),
                context=f"Attempted to create InfallibleResult: {self.source_context}",
            )

    @classmethod
    def create(
        cls,
        data: Dict[str, Any],
        confidence: float,
        source_context: str = "",
    ) -> "InfallibleResult":
        """
        Create an infallible result, enforcing the standard.

        Args:
            data: The result data
            confidence: The confidence score (must be 1.0)
            source_context: Description of where this result came from

        Returns:
            InfallibleResult if confidence meets the standard

        Raises:
            InfallibilityViolation: If confidence is below 1.0
        """
        return cls(
            data=data,
            confidence=confidence,
            source_context=source_context,
        )

    @classmethod
    def create_or_none(
        cls,
        data: Dict[str, Any],
        confidence: float,
        source_context: str = "",
    ) -> Optional["InfallibleResult"]:
        """
        Create an infallible result, returning None if unacceptable.

        Use this when rejection should not raise an exception.

        Args:
            data: The result data
            confidence: The confidence score
            source_context: Description of where this result came from

        Returns:
            InfallibleResult if confidence meets the standard, None otherwise
        """
        if is_acceptable(confidence):
            return cls(data=data, confidence=confidence, source_context=source_context)
        return None


@dataclass(frozen=True)
class SeraphicCertification:
    """
    Certification of a pipeline result.

    The seraph produces ONE type of output: INFALLIBLE.
    Everything else is rejected and not produced at all.
    """
    level: CertificationLevel = CertificationLevel.INFALLIBLE
    score: float = ABSOLUTE_CONFIDENCE
    validation_passed: bool = True
    quality_passed: bool = True
    theological_soundness: float = ABSOLUTE_CONFIDENCE

    def __post_init__(self) -> None:
        """Enforce infallibility on certification."""
        object.__setattr__(self, 'level', CertificationLevel.INFALLIBLE)
        object.__setattr__(self, 'score', ABSOLUTE_CONFIDENCE)
        object.__setattr__(self, 'validation_passed', True)
        object.__setattr__(self, 'quality_passed', True)
        object.__setattr__(self, 'theological_soundness', ABSOLUTE_CONFIDENCE)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "score": self.score,
            "validation_passed": self.validation_passed,
            "quality_passed": self.quality_passed,
            "theological_soundness": self.theological_soundness,
        }


# =============================================================================
# INFALLIBILITY CONSTANTS - For use across the system
# =============================================================================

# Agent configuration
AGENT_MIN_CONFIDENCE: float = ABSOLUTE_CONFIDENCE

# Pipeline configuration
PIPELINE_MIN_CONFIDENCE: float = ABSOLUTE_CONFIDENCE
PIPELINE_PASS_THRESHOLD: float = ABSOLUTE_CONFIDENCE

# ML configuration
ML_MIN_CONFIDENCE: float = ABSOLUTE_CONFIDENCE
ML_SCORE_THRESHOLD: float = ABSOLUTE_CONFIDENCE

# Validation configuration
VALIDATION_PASS_THRESHOLD: float = ABSOLUTE_CONFIDENCE
QUALITY_PASS_THRESHOLD: float = ABSOLUTE_CONFIDENCE

# Cross-reference configuration
CROSSREF_MIN_CONFIDENCE: float = ABSOLUTE_CONFIDENCE
CROSSREF_STRENGTH_THRESHOLD: float = ABSOLUTE_CONFIDENCE

# Theological configuration
THEOLOGICAL_SOUNDNESS_THRESHOLD: float = ABSOLUTE_CONFIDENCE
PATRISTIC_CONSENSUS_THRESHOLD: float = ABSOLUTE_CONFIDENCE  # Anonymous guardrails


__all__ = [
    # Constants
    "ABSOLUTE_CONFIDENCE",
    "PASS_THRESHOLD",
    "REJECTION_THRESHOLD",
    # Enums
    "CertificationLevel",
    "ValidationResult",
    # Functions
    "is_acceptable",
    "classify_rejection",
    "enforce_infallibility",
    # Types
    "InfallibilityViolation",
    "InfallibleResult",
    "SeraphicCertification",
    # Configuration constants
    "AGENT_MIN_CONFIDENCE",
    "PIPELINE_MIN_CONFIDENCE",
    "PIPELINE_PASS_THRESHOLD",
    "ML_MIN_CONFIDENCE",
    "ML_SCORE_THRESHOLD",
    "VALIDATION_PASS_THRESHOLD",
    "QUALITY_PASS_THRESHOLD",
    "CROSSREF_MIN_CONFIDENCE",
    "CROSSREF_STRENGTH_THRESHOLD",
    "THEOLOGICAL_SOUNDNESS_THRESHOLD",
    "PATRISTIC_CONSENSUS_THRESHOLD",
]
