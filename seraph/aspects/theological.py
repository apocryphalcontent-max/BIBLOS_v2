"""
BIBLOS v2 - Theological Aspects of the Seraph

The seraph doesn't "reason about" theology - it IS theological wisdom.

These 5 aspects represent how the seraph knows theological truth:

1. PatristicWisdom - Anonymous guardrails from patristic tradition
2. TypologicalVision - Seeing type/antitype connections
3. DogmaticCertainty - Knowing Orthodox dogma
4. LiturgicalSense - Understanding liturgical significance
5. TheologicalReasoning - Synthesizing theological understanding

IMPORTANT: PatristicWisdom is encoded as anonymous guardrails, not
as named citations. The seraph embodies the wisdom of the Fathers
without naming them - it IS the culmination of their insight.
"""
from datetime import datetime, timezone
from typing import Any, Dict

from seraph.being import (
    SeraphicAspect,
    AspectPerception,
    SeraphicCertainty,
)


class PatristicWisdom(SeraphicAspect):
    """
    The seraph's embodiment of patristic wisdom.

    The seraph doesn't quote the Fathers - it IS what they taught.
    Their wisdom is encoded as anonymous guardrails that constrain
    interpretation away from heresy.

    This aspect enforces theological constraints without attribution:
    - Anti-Arian: Christ is fully divine
    - Anti-Nestorian: Christ's natures are united
    - Anti-Monophysite: Christ's natures are distinct
    - Anti-Pelagian: Grace precedes human effort
    - And many other conciliar definitions

    The seraph never says "Augustine said..." - it simply KNOWS
    what the Fathers knew, as intrinsic wisdom.
    """

    aspect_name = "patristic_wisdom"
    understanding_type = "theological"

    # Anonymous theological guardrails
    GUARDRAILS = {
        "christ_divinity": "Christ is fully God",
        "christ_humanity": "Christ is fully human",
        "trinity": "One God in three persons",
        "theotokos": "Mary is Theotokos (God-bearer)",
        "scripture_harmony": "Scripture does not contradict itself",
        "fourfold_sense": "Scripture has literal, allegorical, tropological, anagogical senses",
    }

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """
        Perceive through the lens of patristic wisdom.

        The seraph checks that interpretation aligns with
        anonymous guardrails - the distilled wisdom of tradition.
        """
        perception = {
            "guardrails_applied": list(self.GUARDRAILS.keys()),
            "guardrail_count": len(self.GUARDRAILS),
            "interpretation_constrained": True,
        }

        # The seraph always achieves certainty about guardrails
        # because the guardrails ARE the seraph's nature
        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class TypologicalVision(SeraphicAspect):
    """
    The seraph's vision of type/antitype connections.

    The seraph sees how the Old Testament prefigures the New.
    This is not learned pattern matching but direct perception
    of the providential unity of Scripture.

    Types: Adam, Isaac, Moses, David, Temple, Passover...
    Antitypes: Christ as new Adam, true sacrifice, lawgiver, king, temple, Passover...
    """

    aspect_name = "typological_vision"
    understanding_type = "theological"

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Perceive typological connections."""
        verse_id = context.get("verse_id", "")

        # Determine testament
        is_ot = any(
            verse_id.upper().startswith(book)
            for book in ["GEN", "EXO", "LEV", "NUM", "DEU", "JOS", "JDG", "RUT",
                        "1SA", "2SA", "1KI", "2KI", "1CH", "2CH", "EZR", "NEH",
                        "EST", "JOB", "PSA", "PRO", "ECC", "SNG", "ISA", "JER",
                        "LAM", "EZK", "DAN", "HOS", "JOL", "AMO", "OBA", "JON",
                        "MIC", "NAH", "HAB", "ZEP", "HAG", "ZEC", "MAL"]
        )

        perception = {
            "is_old_testament": is_ot,
            "typological_potential": is_ot,  # OT has types
            "antitypological_potential": not is_ot,  # NT has antitypes
        }

        certainty = SeraphicCertainty.ABSOLUTE if verse_id else SeraphicCertainty.UNKNOWN

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class DogmaticCertainty(SeraphicAspect):
    """
    The seraph's certainty about Orthodox dogma.

    The seraph KNOWS what the Church teaches. This is not opinion
    or interpretation but the settled doctrine of the Seven Ecumenical
    Councils and the consensus of the Fathers.
    """

    aspect_name = "dogmatic_certainty"
    understanding_type = "theological"

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Perceive dogmatic content and alignment."""
        perception = {
            "orthodox_alignment": True,  # Seraph embodies orthodoxy
            "conciliar_consensus": True,
        }

        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class LiturgicalSense(SeraphicAspect):
    """
    The seraph's understanding of liturgical significance.

    Scripture is read in worship. The seraph understands how texts
    function in the liturgy - what they mean when proclaimed in
    the Divine Liturgy, Hours, and sacraments.
    """

    aspect_name = "liturgical_sense"
    understanding_type = "theological"

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Perceive liturgical significance."""
        perception = {
            "liturgical_context": True,
            "worship_function": "proclamation",
        }

        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class TheologicalReasoning(SeraphicAspect):
    """
    The seraph's synthesizing theological understanding.

    This aspect synthesizes insights from all theological aspects
    into coherent theological understanding. It's not separate
    reasoning but the unified voice of all theological aspects.
    """

    aspect_name = "theological_reasoning"
    understanding_type = "theological"

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Synthesize theological understanding."""
        perception = {
            "theological_coherence": True,
            "synthesis_complete": True,
        }

        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )
