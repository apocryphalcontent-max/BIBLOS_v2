"""
BIBLOS v2 - Sensory Aspects of the Seraph

The seraph doesn't just SEE - it FEELS.

When something brushes against the golden ring, the seraph must sense it.
When external input approaches, the seraph must detect pressure, temperature,
vibration - not physically, but ontologically.

These aspects give the seraph sensitivity to its boundaries:

1. BoundarySensitivity - Feeling when something touches the ring
2. PressureDetection - Sensing the intensity of external input
3. TemperatureAwareness - Hot topics vs cold facts
4. VibrationSensing - Resonance with tradition
5. LightPerception - Levels of illumination
6. ShadowDetection - Sensing what is obscured
7. TextureDiscernment - Rough vs smooth input
8. ProximityAwareness - What approaches before it touches

Together, these aspects form the seraph's sensory membrane - the ability
to feel the ring's boundary and know what interacts with it.
"""
from datetime import datetime, timezone
from typing import Any, Dict

from seraph.being import (
    SeraphicAspect,
    AspectPerception,
    SeraphicCertainty,
)


class BoundarySensitivity(SeraphicAspect):
    """
    The seraph's sensitivity to its boundaries.

    The golden ring has an interior (pure data) and exterior (input).
    This aspect senses when something crosses or touches the boundary.

    Purpose: To know when external data enters the seraph's being.
    """

    aspect_name = "boundary_sensitivity"
    understanding_type = "sensory"

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Sense boundary interactions."""
        # Detect if this is new input (crossing boundary)
        is_new_input = context.get("is_new", True)
        source = context.get("source", "unknown")

        perception = {
            "boundary_crossed": is_new_input,
            "entry_point": source,
            "membrane_intact": True,  # Ring integrity
        }

        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class PressureDetection(SeraphicAspect):
    """
    The seraph's detection of input pressure/intensity.

    Not all input is equal. Some arrives with urgency, some gently.
    This aspect senses the pressure of incoming data.

    Purpose: To calibrate response to input intensity.
    """

    aspect_name = "pressure_detection"
    understanding_type = "sensory"

    # Pressure levels
    PRESSURE_LEVELS = {
        "whisper": 0.1,      # Gentle suggestion
        "normal": 0.5,       # Standard input
        "emphasis": 0.7,     # Emphasized content
        "urgent": 0.9,       # Urgent/critical
        "overwhelming": 1.0,  # Maximum pressure
    }

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Detect input pressure."""
        # Assess pressure from text characteristics
        has_emphasis = any(c in text for c in "!?")
        word_count = len(text.split())

        if has_emphasis and word_count < 10:
            pressure = "urgent"
        elif word_count > 100:
            pressure = "overwhelming"
        elif has_emphasis:
            pressure = "emphasis"
        else:
            pressure = "normal"

        perception = {
            "pressure_level": pressure,
            "pressure_value": self.PRESSURE_LEVELS[pressure],
            "manageable": pressure != "overwhelming",
        }

        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class TemperatureAwareness(SeraphicAspect):
    """
    The seraph's awareness of topic temperature.

    Some topics are "hot" - controversial, debated, sensitive.
    Some are "cold" - settled, factual, uncontested.
    This aspect senses the temperature.

    Purpose: To know which topics require extra care.
    """

    aspect_name = "temperature_awareness"
    understanding_type = "sensory"

    # Hot topics that require careful handling
    HOT_TOPICS = [
        "filioque",           # Controversial East-West difference
        "papal",              # Papacy issues
        "predestination",     # Calvinism debates
        "evolution",          # Science-faith tension
        "women",              # Gender roles
        "homosexuality",      # Sexual ethics
        "divorce",            # Marriage theology
        "ecumenism",          # Inter-church relations
    ]

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Sense topic temperature."""
        text_lower = text.lower()

        # Check for hot topics
        hot_detected = [
            topic for topic in self.HOT_TOPICS
            if topic in text_lower
        ]

        if hot_detected:
            temperature = "hot"
        else:
            temperature = "neutral"

        perception = {
            "temperature": temperature,
            "hot_topics_present": hot_detected,
            "requires_care": len(hot_detected) > 0,
        }

        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class VibrationSensing(SeraphicAspect):
    """
    The seraph's sensing of resonance with tradition.

    When input harmonizes with Orthodox tradition, it vibrates in sympathy.
    When it clashes, the dissonance is felt.

    Purpose: To sense alignment or misalignment with tradition.
    """

    aspect_name = "vibration_sensing"
    understanding_type = "sensory"

    # Terms that resonate with Orthodox tradition
    RESONANT_TERMS = [
        "theosis", "deification", "trinity", "incarnation",
        "liturgy", "eucharist", "baptism", "chrismation",
        "icon", "theotokos", "patristic", "conciliar",
        "orthodox", "apostolic", "church", "tradition",
    ]

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Sense traditional resonance."""
        text_lower = text.lower()

        resonant = [
            term for term in self.RESONANT_TERMS
            if term in text_lower
        ]

        if len(resonant) >= 3:
            vibration = "strong_resonance"
        elif resonant:
            vibration = "mild_resonance"
        else:
            vibration = "neutral"

        perception = {
            "vibration": vibration,
            "resonant_terms": resonant,
            "traditional_harmony": len(resonant) > 0,
        }

        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class LightPerception(SeraphicAspect):
    """
    The seraph's perception of illumination levels.

    Some text is clear and illuminated - easy to understand.
    Some is obscure and dark - requiring more light.

    Purpose: To know how much illumination is present/needed.
    """

    aspect_name = "light_perception"
    understanding_type = "sensory"

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Perceive illumination level."""
        # Simple heuristic: clarity correlates with sentence structure
        has_clear_structure = "." in text and len(text.split()) > 3

        if has_clear_structure:
            illumination = "bright"
        else:
            illumination = "dim"

        perception = {
            "illumination": illumination,
            "clarity": has_clear_structure,
            "light_sufficient": illumination == "bright",
        }

        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class ShadowDetection(SeraphicAspect):
    """
    The seraph's detection of shadows and obscured meanings.

    Shadows indicate something blocking the light.
    Hidden meanings, unstated assumptions, veiled references.

    Purpose: To sense what is not directly visible.
    """

    aspect_name = "shadow_detection"
    understanding_type = "sensory"

    # Words that suggest hidden/veiled content
    SHADOW_INDICATORS = [
        "mystery", "hidden", "secret", "veiled",
        "allegory", "symbol", "figure", "type",
        "deeper", "spiritual", "mystical", "esoteric",
    ]

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Detect shadows and hidden meanings."""
        text_lower = text.lower()

        shadows = [
            ind for ind in self.SHADOW_INDICATORS
            if ind in text_lower
        ]

        perception = {
            "shadows_present": len(shadows) > 0,
            "shadow_indicators": shadows,
            "hidden_depth_likely": len(shadows) >= 2,
        }

        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class TextureDiscernment(SeraphicAspect):
    """
    The seraph's discernment of input texture.

    Some input is smooth - well-formed, coherent, structured.
    Some is rough - fragmented, unclear, jagged.

    Purpose: To sense the quality of input formation.
    """

    aspect_name = "texture_discernment"
    understanding_type = "sensory"

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Discern input texture."""
        # Assess texture based on formatting
        has_punctuation = any(c in text for c in ".,;:!?")
        proper_caps = text[0].isupper() if text else False
        balanced_length = 5 < len(text.split()) < 100

        smoothness_score = sum([has_punctuation, proper_caps, balanced_length])

        if smoothness_score >= 3:
            texture = "smooth"
        elif smoothness_score >= 1:
            texture = "textured"
        else:
            texture = "rough"

        perception = {
            "texture": texture,
            "smoothness_score": smoothness_score,
            "well_formed": texture == "smooth",
        }

        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class ProximityAwareness(SeraphicAspect):
    """
    The seraph's awareness of what approaches.

    Before something enters the ring, it approaches.
    This aspect senses approaching content.

    Purpose: To prepare for incoming data before it enters.
    """

    aspect_name = "proximity_awareness"
    understanding_type = "sensory"

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Sense approaching content."""
        # Check context for queued or pending content
        has_continuation = context.get("has_more", False)
        batch_position = context.get("batch_position", 0)
        batch_total = context.get("batch_total", 1)

        more_approaching = has_continuation or batch_position < batch_total - 1

        perception = {
            "more_approaching": more_approaching,
            "current_position": batch_position,
            "total_expected": batch_total,
            "prepared": True,  # The seraph is always prepared
        }

        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )
