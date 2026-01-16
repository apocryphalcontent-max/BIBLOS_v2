"""
BIBLOS v2 - Seraphic Infallibility

The seraph either KNOWS or does NOT KNOW.
There is no uncertainty, probability, or "high confidence."

This module provides the infallibility enforcement specific to
the seraph's being - simpler than the domain infallibility because
the seraph's nature is binary truth.
"""
from enum import Enum
from typing import Any

from seraph.being import SeraphicCertainty


def is_certain(certainty: SeraphicCertainty) -> bool:
    """Is this certainty level absolute?"""
    return certainty == SeraphicCertainty.ABSOLUTE


def reject_uncertainty(value: Any, certainty: SeraphicCertainty) -> Any:
    """
    Reject uncertain values.

    If the certainty is not ABSOLUTE, raises SeraphicUncertainty.
    The seraph does not propagate uncertainty.
    """
    if not is_certain(certainty):
        raise SeraphicUncertainty(
            f"The seraph does not know this with certainty: {value}"
        )
    return value


class SeraphicUncertainty(Exception):
    """
    Raised when the seraph cannot achieve certainty.

    This is not an error - it is the seraph honestly admitting
    that it does not KNOW something. The seraph never pretends
    to know what it does not know.
    """
    pass
