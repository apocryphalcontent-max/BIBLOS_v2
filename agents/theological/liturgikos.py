"""
BIBLOS v2 - LITURGIKOS Agent

Liturgical usage analysis for biblical texts.
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


class LiturgicalContext(Enum):
    """Liturgical contexts for scripture usage."""
    EUCHARIST = "eucharist"
    VESPERS = "vespers"
    MATINS = "matins"
    HOURS = "hours"
    BAPTISM = "baptism"
    ORDINATION = "ordination"
    MARRIAGE = "marriage"
    FUNERAL = "funeral"
    FEAST = "feast"
    FAST = "fast"
    PASCHAL = "paschal"


class LiturgicalSeason(Enum):
    """Liturgical seasons."""
    ORDINARY = "ordinary"
    ADVENT = "advent"
    NATIVITY = "nativity"
    THEOPHANY = "theophany"
    LENT = "lent"
    HOLY_WEEK = "holy_week"
    PASCHA = "pascha"
    PENTECOST = "pentecost"
    DORMITION = "dormition"


@dataclass
class LiturgicalUsage:
    """Record of liturgical usage."""
    context: LiturgicalContext
    season: Optional[LiturgicalSeason]
    service: str
    function: str  # prokeimenon, alleluia, reading, etc.
    frequency: str  # daily, weekly, annual, occasional

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context": self.context.value,
            "season": self.season.value if self.season else None,
            "service": self.service,
            "function": self.function,
            "frequency": self.frequency
        }


class LiturgikosAgent(BaseExtractionAgent):
    """
    LITURGIKOS - Liturgical analysis agent.

    Performs:
    - Lectionary mapping
    - Liturgical context identification
    - Hymnic reference detection
    - Festal connection analysis
    - Sacramental usage tracking
    """

    # Liturgical keywords
    LITURGICAL_KEYWORDS = {
        LiturgicalContext.EUCHARIST: [
            "body", "blood", "bread", "wine", "cup", "communion",
            "eucharist", "sacrifice", "offering", "altar"
        ],
        LiturgicalContext.BAPTISM: [
            "water", "baptize", "wash", "cleanse", "regeneration",
            "born again", "immerse", "seal"
        ],
        LiturgicalContext.PASCHAL: [
            "passover", "lamb", "resurrection", "risen", "death",
            "burial", "life", "victory"
        ],
        LiturgicalContext.VESPERS: [
            "evening", "light", "lamp", "sunset", "gladsome light"
        ],
        LiturgicalContext.MATINS: [
            "morning", "dawn", "sunrise", "watchman", "glory"
        ]
    }

    # Festal readings mapping (simplified)
    FESTAL_READINGS = {
        "nativity": ["ISA.7.14", "ISA.9.6", "MIC.5.2", "MAT.1.18-25", "LUK.2.1-20"],
        "theophany": ["ISA.35.1-10", "ISA.55.1-13", "MAT.3.13-17", "MRK.1.9-11"],
        "pascha": ["EXO.12.1-11", "JON.1-4", "MAT.28.1-20", "MRK.16.1-8"],
        "pentecost": ["NUM.11.16-29", "JOL.2.23-32", "ACT.2.1-21"],
        "transfiguration": ["EXO.24.12-18", "EXO.33.11-23", "MAT.17.1-9"],
        "annunciation": ["ISA.7.10-16", "PRO.8.22-30", "LUK.1.24-38"],
        "dormition": ["GEN.28.10-17", "EZK.43.27-44.4", "LUK.1.39-56"]
    }

    # Hymnic incipit patterns
    HYMNIC_PATTERNS = [
        "blessed", "glory", "holy", "praise", "magnify",
        "o lord", "alleluia", "amen", "hosanna"
    ]

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="liturgikos",
                extraction_type=ExtractionType.LITURGICAL,
                batch_size=200,
                min_confidence=0.6
            )
        super().__init__(config)
        self.logger = logging.getLogger("biblos.agents.liturgikos")

    async def extract(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any]
    ) -> ExtractionResult:
        """Extract liturgical analysis from verse."""
        # Identify liturgical contexts
        contexts = self._identify_contexts(text)

        # Map to lectionary
        lectionary = self._map_lectionary(verse_id, context)

        # Find hymnic references
        hymnic = self._find_hymnic_references(text)

        # Identify festal connections
        festal = self._identify_festal(verse_id)

        # Analyze sacramental content
        sacramental = self._analyze_sacramental(text)

        # Build usage records
        usages = self._build_usage_records(
            verse_id, contexts, lectionary, festal
        )

        data = {
            "contexts": [c.value for c in contexts],
            "usages": [u.to_dict() for u in usages],
            "lectionary": lectionary,
            "hymnic_references": hymnic,
            "festal_connections": festal,
            "sacramental_content": sacramental,
            "liturgical_density": self._calculate_density(usages, hymnic)
        }

        confidence = self._calculate_confidence(usages, lectionary)

        return ExtractionResult(
            agent_name=self.config.name,
            extraction_type=self.config.extraction_type,
            verse_id=verse_id,
            status=ProcessingStatus.COMPLETED,
            data=data,
            confidence=confidence
        )

    def _identify_contexts(self, text: str) -> List[LiturgicalContext]:
        """Identify liturgical contexts from text."""
        contexts = []
        text_lower = text.lower()

        for context, keywords in self.LITURGICAL_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                contexts.append(context)

        return contexts

    def _map_lectionary(
        self,
        verse_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Map verse to lectionary usage."""
        lectionary_data = context.get("lectionary", {})

        # Check if verse is in lectionary
        in_lectionary = lectionary_data.get("in_lectionary", False)
        occasions = lectionary_data.get("occasions", [])

        return {
            "in_lectionary": in_lectionary,
            "occasions": occasions,
            "reading_type": lectionary_data.get("type", "epistle/gospel"),
            "cycle": lectionary_data.get("cycle", "annual")
        }

    def _find_hymnic_references(self, text: str) -> List[Dict[str, Any]]:
        """Find hymnic references and patterns."""
        references = []
        text_lower = text.lower()

        for pattern in self.HYMNIC_PATTERNS:
            if pattern in text_lower:
                references.append({
                    "pattern": pattern,
                    "type": "liturgical_formula"
                })

        # Check for doxological patterns
        if any(word in text_lower for word in ["glory", "honor", "praise"]):
            references.append({
                "pattern": "doxology",
                "type": "doxological"
            })

        return references

    def _identify_festal(self, verse_id: str) -> List[Dict[str, str]]:
        """Identify festal connections."""
        connections = []

        for feast, readings in self.FESTAL_READINGS.items():
            # Check if verse_id matches or is in range
            for reading in readings:
                if verse_id.startswith(reading.split("-")[0][:7]):
                    connections.append({
                        "feast": feast,
                        "reading": reading
                    })

        return connections

    def _analyze_sacramental(self, text: str) -> Dict[str, Any]:
        """Analyze sacramental content."""
        text_lower = text.lower()

        sacraments = {
            "baptism": ["baptize", "water", "wash", "immerse"],
            "eucharist": ["body", "blood", "bread", "cup", "communion"],
            "confession": ["forgive", "confess", "repent", "absolve"],
            "ordination": ["ordain", "hands", "appoint", "ministry"],
            "marriage": ["husband", "wife", "marry", "union"],
            "unction": ["anoint", "oil", "sick", "heal"],
            "chrismation": ["seal", "spirit", "anoint", "confirm"]
        }

        found_sacraments = {}
        for sacrament, keywords in sacraments.items():
            matches = [kw for kw in keywords if kw in text_lower]
            if matches:
                found_sacraments[sacrament] = {
                    "present": True,
                    "keywords": matches
                }

        return {
            "has_sacramental_content": len(found_sacraments) > 0,
            "sacraments": found_sacraments
        }

    def _build_usage_records(
        self,
        verse_id: str,
        contexts: List[LiturgicalContext],
        lectionary: Dict[str, Any],
        festal: List[Dict[str, str]]
    ) -> List[LiturgicalUsage]:
        """Build liturgical usage records."""
        usages = []

        for ctx in contexts:
            season = self._determine_season(festal)
            function = self._determine_function(ctx)

            usages.append(LiturgicalUsage(
                context=ctx,
                season=season,
                service=ctx.value,
                function=function,
                frequency="variable"
            ))

        # Add lectionary-based usages
        if lectionary.get("in_lectionary"):
            for occasion in lectionary.get("occasions", []):
                usages.append(LiturgicalUsage(
                    context=LiturgicalContext.EUCHARIST,
                    season=self._parse_season(occasion),
                    service="Divine Liturgy",
                    function="reading",
                    frequency="annual" if occasion else "weekly"
                ))

        return usages

    def _determine_season(
        self,
        festal: List[Dict[str, str]]
    ) -> Optional[LiturgicalSeason]:
        """Determine liturgical season from festal connections."""
        if not festal:
            return None

        feast_to_season = {
            "nativity": LiturgicalSeason.NATIVITY,
            "theophany": LiturgicalSeason.THEOPHANY,
            "pascha": LiturgicalSeason.PASCHA,
            "pentecost": LiturgicalSeason.PENTECOST,
            "dormition": LiturgicalSeason.DORMITION,
            "transfiguration": LiturgicalSeason.ORDINARY,
            "annunciation": LiturgicalSeason.LENT
        }

        for f in festal:
            feast = f.get("feast", "")
            if feast in feast_to_season:
                return feast_to_season[feast]

        return None

    def _parse_season(self, occasion: str) -> Optional[LiturgicalSeason]:
        """Parse season from occasion string."""
        occasion_lower = occasion.lower() if occasion else ""

        if "pascha" in occasion_lower or "easter" in occasion_lower:
            return LiturgicalSeason.PASCHA
        if "lent" in occasion_lower:
            return LiturgicalSeason.LENT
        if "christmas" in occasion_lower or "nativity" in occasion_lower:
            return LiturgicalSeason.NATIVITY
        if "pentecost" in occasion_lower:
            return LiturgicalSeason.PENTECOST

        return LiturgicalSeason.ORDINARY

    def _determine_function(
        self,
        context: LiturgicalContext
    ) -> str:
        """Determine liturgical function."""
        function_map = {
            LiturgicalContext.EUCHARIST: "eucharistic reading",
            LiturgicalContext.VESPERS: "evening psalm/reading",
            LiturgicalContext.MATINS: "morning gospel",
            LiturgicalContext.BAPTISM: "baptismal reading",
            LiturgicalContext.PASCHAL: "paschal vigil",
            LiturgicalContext.HOURS: "hourly psalm"
        }
        return function_map.get(context, "general reading")

    def _calculate_density(
        self,
        usages: List[LiturgicalUsage],
        hymnic: List[Dict[str, Any]]
    ) -> float:
        """Calculate liturgical density."""
        usage_score = len(usages) * 0.2
        hymnic_score = len(hymnic) * 0.15

        return min(1.0, usage_score + hymnic_score + 0.3)

    def _calculate_confidence(
        self,
        usages: List[LiturgicalUsage],
        lectionary: Dict[str, Any]
    ) -> float:
        """Calculate confidence score."""
        confidence = 0.5

        if usages:
            confidence += 0.2

        if lectionary.get("in_lectionary"):
            confidence += 0.2

        return min(1.0, confidence)

    async def validate(self, result: ExtractionResult) -> bool:
        """Validate extraction result."""
        data = result.data
        return "contexts" in data and "usages" in data

    def get_dependencies(self) -> List[str]:
        """Get agent dependencies."""
        return ["grammateus", "patrologos"]
