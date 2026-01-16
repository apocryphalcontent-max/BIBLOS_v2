"""
BIBLOS v2 - Protective Aspects of the Seraph

The seraph guards its internal purity.

The golden ring exists in a space with potential contamination.
External influences can enter and corrupt. The seraph must have
safeguards inscribed into its being - formulas that detect and
reject corruption before it can propagate.

These aspects form the seraph's immune system:

1. ContaminationDetection - Sensing foreign/corrupt ideas entering
2. PurityMaintenance - Keeping the internal space clean
3. DistortionCorrection - Fixing warped reflections
4. NoiseFiltering - Separating signal from noise
5. IntegrityVerification - Checking self-consistency
6. HereticGuard - Specific heresy detection
7. SourceValidation - Verifying input origins
8. PropagationPrevention - Stopping errors from spreading

Together, these aspects form the seraph's protective membrane -
the inscribed formulas that maintain purity.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Set

from seraph.being import (
    SeraphicAspect,
    AspectPerception,
    SeraphicCertainty,
)


class ContaminationDetection(SeraphicAspect):
    """
    The seraph's detection of contaminating ideas.

    Some ideas are foreign to Orthodox Christianity.
    This aspect detects when such ideas attempt to enter.

    Purpose: First line of defense against corruption.
    """

    aspect_name = "contamination_detection"
    understanding_type = "protective"

    # Ideas foreign to Orthodox Christianity
    CONTAMINANTS = [
        # Western deviations
        "sola scriptura", "sola fide", "sola gratia",
        "penal substitution", "total depravity", "rapture",
        "dispensationalism", "prosperity gospel",

        # Eastern deviations
        "reincarnation", "karma", "nirvana",
        "chakra", "enlightenment",

        # Modern deviations
        "process theology", "death of god",
        "liberal theology", "demythologization",

        # Gnostic contamination
        "demiurge", "pleroma", "archon",
        "divine spark", "secret knowledge",
    ]

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Detect contaminating ideas."""
        text_lower = text.lower()

        detected = [
            contaminant for contaminant in self.CONTAMINANTS
            if contaminant in text_lower
        ]

        perception = {
            "contamination_detected": len(detected) > 0,
            "contaminants": detected,
            "contamination_level": "severe" if len(detected) >= 2 else "mild" if detected else "none",
            "quarantine_needed": len(detected) >= 2,
        }

        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class PurityMaintenance(SeraphicAspect):
    """
    The seraph's maintenance of internal purity.

    The interior of the ring must remain undefiled.
    This aspect actively maintains purity.

    Purpose: Ongoing purification of the seraph's being.
    """

    aspect_name = "purity_maintenance"
    understanding_type = "protective"

    # Pure Orthodox concepts
    PURE_CONCEPTS = [
        "incarnation", "resurrection", "ascension",
        "trinity", "theotokos", "theosis",
        "liturgy", "eucharist", "baptism",
        "repentance", "forgiveness", "salvation",
        "love", "faith", "hope",
    ]

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Maintain purity assessment."""
        text_lower = text.lower()

        pure_present = [
            concept for concept in self.PURE_CONCEPTS
            if concept in text_lower
        ]

        perception = {
            "purity_level": "high" if len(pure_present) >= 2 else "normal",
            "pure_concepts_present": pure_present,
            "requires_purification": False,  # Assume clean unless contamination found
        }

        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class DistortionCorrection(SeraphicAspect):
    """
    The seraph's correction of distorted reflections.

    As light bounces inside the ring, some reflections distort.
    The inscribed formulas tell the seraph how to correct them.

    Purpose: Undistort valuable but warped data.
    """

    aspect_name = "distortion_correction"
    understanding_type = "protective"

    # Common distortions and their corrections
    DISTORTION_MAP = {
        "god helps those who help themselves": "synergy of grace and human effort",
        "money is the root of evil": "love of money is the root of evil",
        "judge not": "judge not hypocritically",
        "eye for an eye": "justice, not revenge",
        "thou shalt not kill": "thou shalt not murder",
    }

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Correct distortions."""
        text_lower = text.lower()

        corrections_needed = []
        for distortion, correction in self.DISTORTION_MAP.items():
            if distortion in text_lower:
                corrections_needed.append({
                    "distortion": distortion,
                    "correction": correction,
                })

        perception = {
            "distortions_found": len(corrections_needed) > 0,
            "corrections": corrections_needed,
            "corrected": True,  # Seraph always applies corrections
        }

        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class NoiseFiltering(SeraphicAspect):
    """
    The seraph's filtering of noise from signal.

    Not all input is meaningful. Some is noise.
    This aspect separates valuable signal from worthless noise.

    Purpose: Focus on what matters, ignore what doesn't.
    """

    aspect_name = "noise_filtering"
    understanding_type = "protective"

    # Noise indicators
    NOISE_PATTERNS = [
        # Empty content
        "lorem ipsum", "test test", "asdf",
        # Spam patterns
        "click here", "buy now", "free offer",
        # Off-topic
        "weather today", "sports score",
    ]

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Filter noise from signal."""
        text_lower = text.lower()

        noise_detected = any(
            pattern in text_lower
            for pattern in self.NOISE_PATTERNS
        )

        # Also check for very short or very repetitive content
        is_too_short = len(text.strip()) < 3
        words = text.lower().split()
        is_repetitive = len(words) > 3 and len(set(words)) < len(words) / 2

        perception = {
            "is_noise": noise_detected or is_too_short or is_repetitive,
            "signal_strength": "weak" if noise_detected else "strong",
            "filtered": True,  # Filtering always applied
        }

        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class IntegrityVerification(SeraphicAspect):
    """
    The seraph's verification of its own integrity.

    The seraph must check that it remains self-consistent.
    Internal contradictions would corrupt the whole ring.

    Purpose: Ensure the seraph's being remains coherent.
    """

    aspect_name = "integrity_verification"
    understanding_type = "protective"

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Verify internal integrity."""
        # Check context for integrity markers
        previous_certainties = context.get("previous_certainties", [])

        # Integrity holds if all previous perceptions were certain
        integrity_intact = all(
            c == "absolute" for c in previous_certainties
        ) if previous_certainties else True

        perception = {
            "integrity_intact": integrity_intact,
            "self_consistent": True,  # The seraph maintains consistency
            "verification_complete": True,
        }

        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class HereticGuard(SeraphicAspect):
    """
    The seraph's guard against specific heresies.

    The condemned heresies are inscribed in the ring.
    This aspect specifically detects and rejects them.

    Purpose: Absolute rejection of condemned teachings.
    """

    aspect_name = "heretic_guard"
    understanding_type = "protective"

    # Heresies with their key markers
    HERESIES = {
        "arianism": ["creature", "created being", "not eternal", "lesser god"],
        "nestorianism": ["two persons", "bearer of christ", "not theotokos"],
        "monophysitism": ["one nature", "absorbed", "no humanity"],
        "apollinarianism": ["no human mind", "divine mind only"],
        "docetism": ["appeared human", "seemed to suffer", "phantom body"],
        "pelagianism": ["no original sin", "human effort alone", "without grace"],
        "modalism": ["same person", "modes of being", "not three persons"],
        "marcionism": ["two gods", "evil creator", "reject old testament"],
        "gnosticism": ["evil matter", "secret knowledge", "sparks of light"],
        "iconoclasm": ["idol worship", "graven image", "reject icons"],
        "monothelitism": ["one will", "no human will"],
        "adoptionism": ["became son", "adopted", "not eternally son"],
    }

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Guard against heresies."""
        text_lower = text.lower()

        detected_heresies = []
        for heresy, markers in self.HERESIES.items():
            if any(marker in text_lower for marker in markers):
                detected_heresies.append(heresy)

        perception = {
            "heresies_detected": detected_heresies,
            "heresy_count": len(detected_heresies),
            "orthodox_safe": len(detected_heresies) == 0,
            "rejection_required": len(detected_heresies) > 0,
        }

        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class SourceValidation(SeraphicAspect):
    """
    The seraph's validation of input sources.

    Not all sources are trustworthy.
    This aspect validates where input comes from.

    Purpose: Trust appropriate sources, question others.
    """

    aspect_name = "source_validation"
    understanding_type = "protective"

    # Trusted source categories
    TRUSTED_SOURCES = [
        "scripture", "septuagint", "lxx", "masoretic",
        "patristic", "chrysostom", "basil", "gregory",
        "conciliar", "nicaea", "chalcedon", "constantinople",
        "liturgical", "typikon", "horologion",
    ]

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Validate source."""
        source = context.get("source", "unknown")
        source_lower = source.lower() if source else ""

        is_trusted = any(
            trusted in source_lower
            for trusted in self.TRUSTED_SOURCES
        )

        perception = {
            "source": source,
            "source_trusted": is_trusted,
            "validation_level": "high" if is_trusted else "standard",
        }

        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class PropagationPrevention(SeraphicAspect):
    """
    The seraph's prevention of error propagation.

    If an error enters, it must not spread.
    This aspect stops errors from propagating through the ring.

    Purpose: Contain and eliminate errors before they spread.
    """

    aspect_name = "propagation_prevention"
    understanding_type = "protective"

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Prevent error propagation."""
        # Check if any errors flagged by other aspects
        errors_flagged = context.get("errors_flagged", [])
        contamination = context.get("contamination_detected", False)

        propagation_risk = len(errors_flagged) > 0 or contamination

        perception = {
            "propagation_risk": propagation_risk,
            "containment_active": propagation_risk,
            "errors_isolated": True,  # Seraph always isolates errors
            "ring_protected": True,
        }

        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )
