"""
BIBLOS v2 - DOGMATIKOS Agent

Doctrinal and dogmatic analysis for biblical texts.
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from agents.base import (
    BaseExtractionAgent,
    AgentConfig,
    ExtractionResult,
    ExtractionType,
    ProcessingStatus
)


class CouncilReference(Enum):
    """Ecumenical Councils."""
    NICAEA_I = "nicaea_i"  # 325 AD
    CONSTANTINOPLE_I = "constantinople_i"  # 381 AD
    EPHESUS = "ephesus"  # 431 AD
    CHALCEDON = "chalcedon"  # 451 AD
    CONSTANTINOPLE_II = "constantinople_ii"  # 553 AD
    CONSTANTINOPLE_III = "constantinople_iii"  # 680-681 AD
    NICAEA_II = "nicaea_ii"  # 787 AD


class HeresyType(Enum):
    """Historical heresies addressed."""
    ARIANISM = "arianism"  # Denial of Christ's divinity
    NESTORIANISM = "nestorianism"  # Division of Christ's natures
    MONOPHYSITISM = "monophysitism"  # One nature in Christ
    PELAGIANISM = "pelagianism"  # Denial of original sin/grace
    GNOSTICISM = "gnosticism"  # Dualism, secret knowledge
    DOCETISM = "docetism"  # Denial of Christ's humanity
    PNEUMATOMACHIANISM = "pneumatomachianism"  # Denial of Spirit's divinity


@dataclass
class DogmaticAssertion:
    """A dogmatic assertion with conciliar backing."""
    doctrine: str
    council_support: Optional[CouncilReference]
    creedal_text: Optional[str]
    opposed_heresies: List[HeresyType]
    authority_level: str  # dogma, doctrine, theologoumenon, opinion

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doctrine": self.doctrine,
            "council_support": self.council_support.value if self.council_support else None,
            "creedal_text": self.creedal_text,
            "opposed_heresies": [h.value for h in self.opposed_heresies],
            "authority_level": self.authority_level
        }


class DogmatikosAgent(BaseExtractionAgent):
    """
    DOGMATIKOS - Dogmatic analysis agent.

    Performs:
    - Conciliar reference mapping
    - Creedal formulation analysis
    - Heresy opposition identification
    - Dogmatic weight assessment
    - Orthodox tradition alignment
    """

    # Dogmatic keywords by council
    COUNCIL_DOCTRINES = {
        CouncilReference.NICAEA_I: {
            "keywords": ["consubstantial", "begotten", "not made", "one essence"],
            "doctrine": "Christ's full divinity",
            "heresies": [HeresyType.ARIANISM]
        },
        CouncilReference.CONSTANTINOPLE_I: {
            "keywords": ["holy spirit", "lord", "life-giver", "proceeds"],
            "doctrine": "Holy Spirit's divinity and procession",
            "heresies": [HeresyType.PNEUMATOMACHIANISM]
        },
        CouncilReference.EPHESUS: {
            "keywords": ["theotokos", "mother of god", "one person"],
            "doctrine": "Unity of Christ's person, Mary as Theotokos",
            "heresies": [HeresyType.NESTORIANISM]
        },
        CouncilReference.CHALCEDON: {
            "keywords": ["two natures", "unmixed", "unchanged", "undivided"],
            "doctrine": "Two natures in one person",
            "heresies": [HeresyType.MONOPHYSITISM, HeresyType.NESTORIANISM]
        }
    }

    # Key dogmatic terms
    DOGMATIC_TERMS = {
        "trinity": {
            "terms": ["father", "son", "spirit", "three", "one god"],
            "authority": "dogma",
            "councils": [CouncilReference.NICAEA_I, CouncilReference.CONSTANTINOPLE_I]
        },
        "incarnation": {
            "terms": ["became", "flesh", "man", "born", "virgin"],
            "authority": "dogma",
            "councils": [CouncilReference.NICAEA_I, CouncilReference.CHALCEDON]
        },
        "salvation": {
            "terms": ["save", "redeem", "grace", "faith"],
            "authority": "doctrine",
            "councils": []
        },
        "theosis": {
            "terms": ["partakers", "divine nature", "deified", "glorified"],
            "authority": "doctrine",
            "councils": []
        }
    }

    # Creedal phrases
    NICENE_CREED_PHRASES = [
        "one god", "father almighty", "maker of heaven",
        "one lord jesus christ", "only begotten", "begotten not made",
        "consubstantial", "came down", "was incarnate", "became man",
        "crucified", "suffered", "buried", "rose again", "ascended",
        "sits at the right hand", "will come again", "judge",
        "holy spirit", "lord", "giver of life", "proceeds",
        "worshipped", "glorified", "spoke through prophets",
        "one holy catholic apostolic church", "one baptism",
        "forgiveness of sins", "resurrection of the dead",
        "life of the world to come"
    ]

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="dogmatikos",
                extraction_type=ExtractionType.THEOLOGICAL,
                batch_size=200,
                min_confidence=0.65
            )
        super().__init__(config)
        self.logger = logging.getLogger("biblos.agents.dogmatikos")

    async def extract(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any]
    ) -> ExtractionResult:
        """Extract dogmatic analysis from verse."""
        # Map to councils
        councils = self._map_to_councils(text)

        # Extract dogmatic assertions
        assertions = self._extract_assertions(text, councils)

        # Identify creedal connections
        creedal = self._identify_creedal(text)

        # Check for heresy opposition
        heresies = self._identify_heresies(text, councils)

        # Assess dogmatic weight
        weight = self._assess_weight(assertions, councils)

        # Check Orthodox alignment
        orthodox = self._check_orthodox_alignment(text, councils)

        data = {
            "councils": [c.value for c in councils],
            "assertions": [a.to_dict() for a in assertions],
            "creedal_connections": creedal,
            "opposed_heresies": [h.value for h in heresies],
            "dogmatic_weight": weight,
            "orthodox_alignment": orthodox,
            "authority_level": self._determine_authority(assertions)
        }

        confidence = self._calculate_confidence(councils, assertions)

        return ExtractionResult(
            agent_name=self.config.name,
            extraction_type=self.config.extraction_type,
            verse_id=verse_id,
            status=ProcessingStatus.COMPLETED,
            data=data,
            confidence=confidence
        )

    def _map_to_councils(self, text: str) -> List[CouncilReference]:
        """Map text to relevant ecumenical councils."""
        councils = []
        text_lower = text.lower()

        for council, info in self.COUNCIL_DOCTRINES.items():
            if any(kw in text_lower for kw in info["keywords"]):
                councils.append(council)

        return councils

    def _extract_assertions(
        self,
        text: str,
        councils: List[CouncilReference]
    ) -> List[DogmaticAssertion]:
        """Extract dogmatic assertions from text."""
        assertions = []
        text_lower = text.lower()

        for doctrine_name, info in self.DOGMATIC_TERMS.items():
            if any(term in text_lower for term in info["terms"]):
                council = None
                if info["councils"]:
                    # Use first matching council
                    for c in info["councils"]:
                        if c in councils:
                            council = c
                            break
                    if not council and info["councils"]:
                        council = info["councils"][0]

                # Get heresies from council
                heresies = []
                if council and council in self.COUNCIL_DOCTRINES:
                    heresies = self.COUNCIL_DOCTRINES[council]["heresies"]

                creedal = self._find_creedal_phrase(text)

                assertions.append(DogmaticAssertion(
                    doctrine=doctrine_name,
                    council_support=council,
                    creedal_text=creedal,
                    opposed_heresies=heresies,
                    authority_level=info["authority"]
                ))

        return assertions

    def _find_creedal_phrase(self, text: str) -> Optional[str]:
        """Find matching creedal phrase."""
        text_lower = text.lower()

        for phrase in self.NICENE_CREED_PHRASES:
            if phrase in text_lower:
                return phrase

        return None

    def _identify_creedal(self, text: str) -> List[Dict[str, Any]]:
        """Identify creedal connections."""
        connections = []
        text_lower = text.lower()

        for phrase in self.NICENE_CREED_PHRASES:
            if phrase in text_lower:
                connections.append({
                    "phrase": phrase,
                    "creed": "Nicene-Constantinopolitan",
                    "match_type": "direct"
                })

        return connections

    def _identify_heresies(
        self,
        text: str,
        councils: List[CouncilReference]
    ) -> List[HeresyType]:
        """Identify heresies opposed by text."""
        heresies = set()

        for council in councils:
            if council in self.COUNCIL_DOCTRINES:
                for heresy in self.COUNCIL_DOCTRINES[council]["heresies"]:
                    heresies.add(heresy)

        return list(heresies)

    def _assess_weight(
        self,
        assertions: List[DogmaticAssertion],
        councils: List[CouncilReference]
    ) -> Dict[str, Any]:
        """Assess dogmatic weight of text."""
        if not assertions:
            return {
                "level": "low",
                "has_conciliar_support": False,
                "is_creedal": False
            }

        has_dogma = any(
            a.authority_level == "dogma" for a in assertions
        )
        has_council = any(a.council_support for a in assertions)
        has_creedal = any(a.creedal_text for a in assertions)

        level = "high" if has_dogma and has_council else "medium" if has_council else "low"

        return {
            "level": level,
            "has_conciliar_support": has_council,
            "is_creedal": has_creedal,
            "council_count": len(councils),
            "assertion_count": len(assertions)
        }

    def _check_orthodox_alignment(
        self,
        text: str,
        councils: List[CouncilReference]
    ) -> Dict[str, Any]:
        """Check alignment with Orthodox tradition."""
        text_lower = text.lower()

        # Check for Orthodox-specific emphases
        orthodox_markers = {
            "theosis": ["partakers", "divine nature", "deification"],
            "mystical": ["mystery", "mystical", "hidden"],
            "liturgical": ["liturgy", "worship", "praise"],
            "patristic": ["fathers", "tradition", "received"]
        }

        alignments = {}
        for category, markers in orthodox_markers.items():
            if any(m in text_lower for m in markers):
                alignments[category] = True

        return {
            "aligned": bool(councils) or bool(alignments),
            "council_based": bool(councils),
            "emphases": list(alignments.keys())
        }

    def _determine_authority(
        self,
        assertions: List[DogmaticAssertion]
    ) -> str:
        """Determine highest authority level."""
        if not assertions:
            return "opinion"

        levels = ["dogma", "doctrine", "theologoumenon", "opinion"]

        for level in levels:
            if any(a.authority_level == level for a in assertions):
                return level

        return "opinion"

    def _calculate_confidence(
        self,
        councils: List[CouncilReference],
        assertions: List[DogmaticAssertion]
    ) -> float:
        """Calculate confidence score."""
        confidence = 0.5

        if councils:
            confidence += 0.2

        if assertions:
            confidence += 0.2
            # Boost for dogmatic assertions
            dogma_count = sum(
                1 for a in assertions if a.authority_level == "dogma"
            )
            confidence += dogma_count * 0.05

        return min(1.0, confidence)

    async def validate(self, result: ExtractionResult) -> bool:
        """Validate extraction result."""
        data = result.data
        return "councils" in data and "assertions" in data

    def get_dependencies(self) -> List[str]:
        """Get agent dependencies."""
        return ["grammateus", "theologos", "patrologos"]
