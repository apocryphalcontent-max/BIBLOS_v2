"""
BIBLOS v2 - Theological Constraint Validator (Enhanced to Impossible Sophistication)

Encodes patristic theological principles as algorithmic constraints for
"covert theological governance" - truth enforcement without explicit
attribution to sources.

ENHANCED PRINCIPLES (12 Core Constraints):
1.  Antitype Escalation - Antitype must exceed type in scope/magnitude
2.  Prophetic Coherence - Fulfillment extends promise, never contradicts
3.  Chronological Priority - Type MUST historically precede antitype
4.  Christological Warrant - Requires apostolic use OR patristic consensus
5.  Liturgical Amplification - Liturgical connections boost confidence
6.  Fourfold Foundation - Allegorical reading requires literal foundation
7.  Trinitarian Grammar - Validates proper Trinitarian language patterns
8.  Theosis Trajectory - Validates deification themes and progression
9.  Conciliar Alignment - Checks against ecumenical council definitions
10. Canonical Priority - LXX/MT divergence handling with apostolic preference
11. Sacramental Typology - Validates eucharistic/baptismal connections
12. Eschatological Coherence - Validates already/not-yet tension

WITNESS HIERARCHY:
- Apostolic (NT) - Weight 1.0
- Ante-Nicene Fathers - Weight 0.95
- Nicene/Post-Nicene - Weight 0.90
- Byzantine Fathers - Weight 0.85
- Liturgical Tradition - Weight 0.80
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Any, FrozenSet
import logging
import re
import math
from datetime import date

# Import canonical book order for chronological validation
from config import BOOK_ORDER


logger = logging.getLogger("biblos.ml.validators.theological_constraints")


# =============================================================================
# ENUMS - Comprehensive Theological Categories
# =============================================================================

class ConstraintViolationSeverity(Enum):
    """
    Severity levels for constraint violations.

    IMPOSSIBLE: Logical impossibility, reject outright (confidence = 0)
    HERETICAL: Contradicts conciliar definition (confidence × 0.05)
    CRITICAL: Severe theological error (confidence × 0.2-0.3)
    SOFT: Marginal violation (confidence × 0.7-0.8)
    WARNING: Not ideal but acceptable (confidence × 0.9)
    NEUTRAL: No modification (confidence × 1.0)
    BOOST: Positive validation (confidence × 1.1-1.2)
    APOSTOLIC_BOOST: Apostolic warrant (confidence × 1.3)
    PASCHAL_BOOST: Paschal significance (confidence × 1.5)
    """
    IMPOSSIBLE = "IMPOSSIBLE"
    HERETICAL = "HERETICAL"
    CRITICAL = "CRITICAL"
    SOFT = "SOFT"
    WARNING = "WARNING"
    NEUTRAL = "NEUTRAL"
    BOOST = "BOOST"
    APOSTOLIC_BOOST = "APOSTOLIC_BOOST"
    PASCHAL_BOOST = "PASCHAL_BOOST"


class ConstraintType(Enum):
    """Types of theological constraints - expanded from 6 to 12."""
    TYPOLOGICAL_ESCALATION = "TYPOLOGICAL_ESCALATION"
    PROPHETIC_COHERENCE = "PROPHETIC_COHERENCE"
    CHRONOLOGICAL_PRIORITY = "CHRONOLOGICAL_PRIORITY"
    CHRISTOLOGICAL_WARRANT = "CHRISTOLOGICAL_WARRANT"
    LITURGICAL_AMPLIFICATION = "LITURGICAL_AMPLIFICATION"
    FOURFOLD_FOUNDATION = "FOURFOLD_FOUNDATION"
    TRINITARIAN_GRAMMAR = "TRINITARIAN_GRAMMAR"
    THEOSIS_TRAJECTORY = "THEOSIS_TRAJECTORY"
    CONCILIAR_ALIGNMENT = "CONCILIAR_ALIGNMENT"
    CANONICAL_PRIORITY = "CANONICAL_PRIORITY"
    SACRAMENTAL_TYPOLOGY = "SACRAMENTAL_TYPOLOGY"
    ESCHATOLOGICAL_COHERENCE = "ESCHATOLOGICAL_COHERENCE"


class Scope(Enum):
    """Scope levels for typological analysis - expanded hierarchy."""
    INDIVIDUAL = "INDIVIDUAL"       # Single person, momentary
    FAMILIAL = "FAMILIAL"           # Household, patriarchal
    LOCAL = "LOCAL"                 # City, tribe, place-specific
    NATIONAL = "NATIONAL"           # Israel, nation, assembly
    INTERNATIONAL = "INTERNATIONAL" # Multiple nations, gentiles
    UNIVERSAL = "UNIVERSAL"         # All humanity, worldwide
    COSMIC = "COSMIC"               # All creation, heavens and earth
    ETERNAL = "ETERNAL"             # Beyond time, eschatological


class WitnessEra(Enum):
    """Historical era of patristic witness."""
    APOSTOLIC = "APOSTOLIC"             # NT authors (30-100 AD)
    APOSTOLIC_FATHERS = "APOSTOLIC_FATHERS"  # Clement, Ignatius, Polycarp (70-150)
    ANTE_NICENE = "ANTE_NICENE"         # Pre-325 AD
    NICENE = "NICENE"                   # 325-451 AD (first four councils)
    POST_CHALCEDONIAN = "POST_CHALCEDONIAN"  # 451-787 AD
    BYZANTINE = "BYZANTINE"             # 787-1453 AD
    MODERN = "MODERN"                   # Post-1453


class ConciliarAuthority(Enum):
    """Ecumenical Council authority levels."""
    NICAEA_I = "NICAEA_I"               # 325 - Trinity, Arianism
    CONSTANTINOPLE_I = "CONSTANTINOPLE_I"  # 381 - Holy Spirit
    EPHESUS = "EPHESUS"                 # 431 - Theotokos, Nestorianism
    CHALCEDON = "CHALCEDON"             # 451 - Two Natures
    CONSTANTINOPLE_II = "CONSTANTINOPLE_II"  # 553 - Three Chapters
    CONSTANTINOPLE_III = "CONSTANTINOPLE_III"  # 680-681 - Two Wills
    NICAEA_II = "NICAEA_II"             # 787 - Icons


class TrinitarianPattern(Enum):
    """Patterns of Trinitarian language."""
    MONARCHIA = "MONARCHIA"             # Father as source
    PERICHORESIS = "PERICHORESIS"       # Mutual indwelling
    HOMOOUSIOS = "HOMOOUSIOS"           # Same essence
    HYPOSTASIS = "HYPOSTASIS"           # Distinct persons
    PROCESSION = "PROCESSION"           # Spirit from Father
    GENERATION = "GENERATION"           # Son from Father
    ECONOMIC = "ECONOMIC"               # Actions in creation
    IMMANENT = "IMMANENT"               # Eternal relations


class TheosisStage(Enum):
    """Stages in the theosis trajectory."""
    CREATION = "CREATION"               # Image of God
    FALL = "FALL"                       # Corruption of image
    PURIFICATION = "PURIFICATION"       # Katharsis
    ILLUMINATION = "ILLUMINATION"       # Photisis
    DEIFICATION = "DEIFICATION"         # Theosis
    GLORIFICATION = "GLORIFICATION"     # Eschatological completion


class SacramentalType(Enum):
    """Types of sacramental prefiguration."""
    BAPTISMAL = "BAPTISMAL"             # Water, cleansing, rebirth
    EUCHARISTIC = "EUCHARISTIC"         # Bread, wine, sacrifice, meal
    CHRISMATION = "CHRISMATION"         # Anointing, Spirit
    ORDINATION = "ORDINATION"           # Priesthood, laying on hands
    MATRIMONIAL = "MATRIMONIAL"         # Bride, bridegroom, covenant
    UNCTION = "UNCTION"                 # Healing, oil
    CONFESSION = "CONFESSION"           # Forgiveness, restoration


class EschatologicalTension(Enum):
    """Already/not-yet eschatological categories."""
    INAUGURATED = "INAUGURATED"         # Already fulfilled
    PROGRESSIVE = "PROGRESSIVE"         # Being fulfilled
    CONSUMMATED = "CONSUMMATED"         # Not yet, awaiting
    REALIZED = "REALIZED"               # Fully present
    APOCALYPTIC = "APOCALYPTIC"         # Dramatic intervention


class LiturgicalSeason(Enum):
    """Byzantine liturgical calendar seasons."""
    PASCHA = "PASCHA"                   # Highest feast
    BRIGHT_WEEK = "BRIGHT_WEEK"         # Easter week
    PENTECOSTARION = "PENTECOSTARION"   # Easter to Pentecost
    TRIODION = "TRIODION"               # Pre-Lenten
    GREAT_LENT = "GREAT_LENT"           # 40 days
    HOLY_WEEK = "HOLY_WEEK"             # Passion week
    NATIVITY = "NATIVITY"               # Christmas cycle
    THEOPHANY = "THEOPHANY"             # Epiphany
    TRANSFIGURATION = "TRANSFIGURATION" # August 6
    DORMITION = "DORMITION"             # August 15
    EXALTATION_CROSS = "EXALTATION_CROSS"  # September 14
    ORDINARY = "ORDINARY"               # Regular Sundays


# =============================================================================
# WITNESS HIERARCHY WEIGHTS
# =============================================================================

WITNESS_ERA_WEIGHTS: Dict[WitnessEra, float] = {
    WitnessEra.APOSTOLIC: 1.00,
    WitnessEra.APOSTOLIC_FATHERS: 0.97,
    WitnessEra.ANTE_NICENE: 0.95,
    WitnessEra.NICENE: 0.92,
    WitnessEra.POST_CHALCEDONIAN: 0.88,
    WitnessEra.BYZANTINE: 0.85,
    WitnessEra.MODERN: 0.75,
}


# =============================================================================
# PATRISTIC CLASSIFICATION
# =============================================================================

FATHER_CLASSIFICATION: Dict[str, Tuple[WitnessEra, float]] = {
    # Apostolic era - highest authority
    "Paul": (WitnessEra.APOSTOLIC, 1.0),
    "John": (WitnessEra.APOSTOLIC, 1.0),
    "Peter": (WitnessEra.APOSTOLIC, 1.0),
    "Matthew": (WitnessEra.APOSTOLIC, 1.0),
    "Luke": (WitnessEra.APOSTOLIC, 1.0),
    "James": (WitnessEra.APOSTOLIC, 1.0),

    # Apostolic Fathers
    "Clement_Rome": (WitnessEra.APOSTOLIC_FATHERS, 0.97),
    "Ignatius": (WitnessEra.APOSTOLIC_FATHERS, 0.97),
    "Polycarp": (WitnessEra.APOSTOLIC_FATHERS, 0.97),
    "Didache": (WitnessEra.APOSTOLIC_FATHERS, 0.95),
    "Barnabas": (WitnessEra.APOSTOLIC_FATHERS, 0.93),
    "Hermas": (WitnessEra.APOSTOLIC_FATHERS, 0.90),

    # Ante-Nicene (pre-325)
    "Justin_Martyr": (WitnessEra.ANTE_NICENE, 0.95),
    "Irenaeus": (WitnessEra.ANTE_NICENE, 0.96),
    "Clement_Alexandria": (WitnessEra.ANTE_NICENE, 0.94),
    "Origen": (WitnessEra.ANTE_NICENE, 0.93),  # Slightly lower due to later condemnations
    "Tertullian": (WitnessEra.ANTE_NICENE, 0.90),  # Later Montanism
    "Cyprian": (WitnessEra.ANTE_NICENE, 0.95),
    "Hippolytus": (WitnessEra.ANTE_NICENE, 0.94),
    "Methodius": (WitnessEra.ANTE_NICENE, 0.93),

    # Nicene (325-451)
    "Athanasius": (WitnessEra.NICENE, 0.98),
    "Basil": (WitnessEra.NICENE, 0.98),
    "Gregory_Nazianzen": (WitnessEra.NICENE, 0.98),
    "Gregory_Nyssa": (WitnessEra.NICENE, 0.97),
    "Chrysostom": (WitnessEra.NICENE, 0.98),
    "Cyril_Alexandria": (WitnessEra.NICENE, 0.97),
    "Ambrose": (WitnessEra.NICENE, 0.96),
    "Augustine": (WitnessEra.NICENE, 0.95),  # Slightly cautious in Orthodox usage
    "Jerome": (WitnessEra.NICENE, 0.95),
    "Ephrem": (WitnessEra.NICENE, 0.96),
    "Hilary": (WitnessEra.NICENE, 0.94),
    "Epiphanius": (WitnessEra.NICENE, 0.93),

    # Post-Chalcedonian (451-787)
    "Leo_Great": (WitnessEra.POST_CHALCEDONIAN, 0.95),
    "Maximus_Confessor": (WitnessEra.POST_CHALCEDONIAN, 0.97),
    "John_Damascus": (WitnessEra.POST_CHALCEDONIAN, 0.97),
    "Dionysius_Areopagite": (WitnessEra.POST_CHALCEDONIAN, 0.93),
    "Romanos_Melodist": (WitnessEra.POST_CHALCEDONIAN, 0.92),
    "Andrew_Crete": (WitnessEra.POST_CHALCEDONIAN, 0.91),

    # Byzantine (787-1453)
    "Theodore_Studite": (WitnessEra.BYZANTINE, 0.90),
    "Photios": (WitnessEra.BYZANTINE, 0.92),
    "Symeon_New_Theologian": (WitnessEra.BYZANTINE, 0.93),
    "Gregory_Palamas": (WitnessEra.BYZANTINE, 0.95),
    "Nicholas_Cabasilas": (WitnessEra.BYZANTINE, 0.91),
    "Theophylact": (WitnessEra.BYZANTINE, 0.90),
}


# =============================================================================
# LITURGICAL CALENDAR WEIGHTS
# =============================================================================

LITURGICAL_WEIGHTS: Dict[LiturgicalSeason, float] = {
    LiturgicalSeason.PASCHA: 2.0,
    LiturgicalSeason.BRIGHT_WEEK: 1.9,
    LiturgicalSeason.HOLY_WEEK: 1.85,
    LiturgicalSeason.PENTECOSTARION: 1.5,
    LiturgicalSeason.GREAT_LENT: 1.4,
    LiturgicalSeason.TRIODION: 1.3,
    LiturgicalSeason.THEOPHANY: 1.6,
    LiturgicalSeason.NATIVITY: 1.5,
    LiturgicalSeason.TRANSFIGURATION: 1.5,
    LiturgicalSeason.DORMITION: 1.4,
    LiturgicalSeason.EXALTATION_CROSS: 1.45,
    LiturgicalSeason.ORDINARY: 1.0,
}


# =============================================================================
# CONCILIAR DEFINITIONS - What must not be contradicted
# =============================================================================

@dataclass(frozen=True)
class ConciliarDefinition:
    """A dogmatic definition from an ecumenical council."""
    council: ConciliarAuthority
    year: int
    topic: str
    affirmation: str
    anathema: str
    key_terms: FrozenSet[str]


CONCILIAR_DEFINITIONS: List[ConciliarDefinition] = [
    ConciliarDefinition(
        council=ConciliarAuthority.NICAEA_I,
        year=325,
        topic="Divinity of Christ",
        affirmation="The Son is homoousios (of one essence) with the Father",
        anathema="Arius: The Son is a creature, there was when he was not",
        key_terms=frozenset(["homoousios", "consubstantial", "true God", "begotten not made"]),
    ),
    ConciliarDefinition(
        council=ConciliarAuthority.CONSTANTINOPLE_I,
        year=381,
        topic="Divinity of Holy Spirit",
        affirmation="The Spirit is Lord, Giver of Life, proceeding from the Father",
        anathema="Macedonius: The Spirit is a creature, inferior to Father and Son",
        key_terms=frozenset(["Lord", "Giver of Life", "proceeds", "worshipped", "glorified"]),
    ),
    ConciliarDefinition(
        council=ConciliarAuthority.EPHESUS,
        year=431,
        topic="Unity of Christ's Person",
        affirmation="Mary is Theotokos because Christ is one person, divine and human",
        anathema="Nestorius: Mary bore only the human Christ, two persons",
        key_terms=frozenset(["Theotokos", "one person", "hypostatic union"]),
    ),
    ConciliarDefinition(
        council=ConciliarAuthority.CHALCEDON,
        year=451,
        topic="Two Natures of Christ",
        affirmation="Christ is in two natures, unconfused, unchangeable, undivided, inseparable",
        anathema="Eutyches: One nature after the union (Monophysitism)",
        key_terms=frozenset(["two natures", "unconfused", "unchangeable", "undivided", "inseparable"]),
    ),
    ConciliarDefinition(
        council=ConciliarAuthority.CONSTANTINOPLE_II,
        year=553,
        topic="Christological Clarification",
        affirmation="One of the Trinity suffered in the flesh",
        anathema="Theodore of Mopsuestia, Theodoret's writings against Cyril",
        key_terms=frozenset(["one of Trinity", "suffered in flesh", "enhypostasia"]),
    ),
    ConciliarDefinition(
        council=ConciliarAuthority.CONSTANTINOPLE_III,
        year=681,
        topic="Two Wills of Christ",
        affirmation="Christ has two wills, divine and human, in harmony",
        anathema="Monothelitism: Christ has only one will",
        key_terms=frozenset(["two wills", "dyothelitism", "human will", "divine will"]),
    ),
    ConciliarDefinition(
        council=ConciliarAuthority.NICAEA_II,
        year=787,
        topic="Veneration of Icons",
        affirmation="Icons may be venerated; honor passes to the prototype",
        anathema="Iconoclasm: Icon veneration is idolatry",
        key_terms=frozenset(["icon", "image", "veneration", "prototype", "circumscription"]),
    ),
]


# =============================================================================
# RESULT DATACLASS - Enhanced
# =============================================================================

@dataclass
class ConstraintResult:
    """
    Result of a constraint evaluation.

    Enhanced with:
    - Theological reasoning chain
    - Patristic citation support
    - Conciliar reference
    - Remedy suggestions
    """
    passed: bool
    constraint_type: ConstraintType
    violation_severity: Optional[ConstraintViolationSeverity] = None
    confidence_modifier: float = 1.0
    reason: str = ""
    evidence: List[str] = field(default_factory=list)
    recoverable: bool = True
    theological_reasoning: List[str] = field(default_factory=list)
    patristic_citations: List[str] = field(default_factory=list)
    conciliar_reference: Optional[ConciliarAuthority] = None
    remedy_suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "passed": self.passed,
            "constraint_type": self.constraint_type.value,
            "violation_severity": self.violation_severity.value if self.violation_severity else None,
            "confidence_modifier": self.confidence_modifier,
            "reason": self.reason,
            "evidence": self.evidence,
            "recoverable": self.recoverable,
            "theological_reasoning": self.theological_reasoning,
            "patristic_citations": self.patristic_citations,
            "conciliar_reference": self.conciliar_reference.value if self.conciliar_reference else None,
            "remedy_suggestions": self.remedy_suggestions,
        }

    @property
    def severity_score(self) -> float:
        """Numerical severity score for sorting/comparison."""
        severity_scores = {
            ConstraintViolationSeverity.IMPOSSIBLE: 0.0,
            ConstraintViolationSeverity.HERETICAL: 0.05,
            ConstraintViolationSeverity.CRITICAL: 0.25,
            ConstraintViolationSeverity.SOFT: 0.75,
            ConstraintViolationSeverity.WARNING: 0.9,
            ConstraintViolationSeverity.NEUTRAL: 1.0,
            ConstraintViolationSeverity.BOOST: 1.15,
            ConstraintViolationSeverity.APOSTOLIC_BOOST: 1.3,
            ConstraintViolationSeverity.PASCHAL_BOOST: 1.5,
        }
        if self.violation_severity:
            return severity_scores.get(self.violation_severity, 1.0)
        return 1.0 if self.passed else 0.5


# =============================================================================
# SCOPE/MAGNITUDE ANALYZER - Enhanced
# =============================================================================

class ScopeMagnitudeAnalyzer:
    """
    Analyzes scope and magnitude for typological escalation validation.

    Enhanced with:
    - 8-level scope hierarchy
    - Agent-action-patient analysis
    - Temporal scope (momentary → eternal)
    - Spatial scope (local → cosmic)
    - Ontological scope (creature → Creator)
    """

    # Keywords indicating scope levels - expanded
    SCOPE_INDICATORS: Dict[Scope, List[str]] = {
        Scope.ETERNAL: [
            "eternal", "everlasting", "forever", "ages of ages",
            "before all ages", "world without end", "immortal",
            "αἰώνιος", "עולם"  # Greek and Hebrew
        ],
        Scope.COSMIC: [
            "creation", "universe", "all things", "heaven and earth",
            "cosmos", "heavens", "foundations", "all that is",
            "visible and invisible", "κόσμος", "κτίσις"
        ],
        Scope.UNIVERSAL: [
            "all nations", "humanity", "mankind", "world", "every",
            "all people", "gentiles", "all flesh", "whosoever",
            "ends of the earth", "πᾶς", "כל"
        ],
        Scope.INTERNATIONAL: [
            "nations", "peoples", "kingdoms", "empires", "tongues",
            "tribes", "many nations", "gentile", "ἔθνη", "גוים"
        ],
        Scope.NATIONAL: [
            "israel", "people", "nation", "tribe", "house of",
            "children of", "assembly", "congregation", "ἐκκλησία", "עם"
        ],
        Scope.LOCAL: [
            "city", "village", "place", "region", "land", "country",
            "territory", "dwelling", "πόλις", "עיר"
        ],
        Scope.FAMILIAL: [
            "family", "household", "father's house", "descendants",
            "offspring", "seed", "generations", "οἶκος", "בית"
        ],
        Scope.INDIVIDUAL: [
            "man", "person", "individual", "one", "single",
            "servant", "prophet", "king", "ἄνθρωπος", "איש"
        ],
    }

    # Agent significance rankings - expanded
    AGENT_SIGNIFICANCE: Dict[str, int] = {
        # Divine
        "god": 100, "lord": 100, "yahweh": 100, "elohim": 100,
        "the almighty": 100, "i am": 100,
        # Christological
        "christ": 98, "messiah": 98, "son of god": 98, "word": 97,
        "lamb of god": 97, "son of man": 96, "lord jesus": 98,
        # Pneumatological
        "spirit": 95, "holy spirit": 95, "spirit of god": 95,
        "comforter": 94, "paraclete": 94,
        # Angelic
        "seraph": 80, "cherub": 80, "angel": 75, "archangel": 78,
        "michael": 78, "gabriel": 78,
        # Human - sacred offices
        "prophet": 65, "apostle": 68, "high priest": 60,
        "king": 55, "priest": 55, "judge": 50,
        # Human - patriarchs and leaders
        "moses": 65, "abraham": 63, "david": 60, "elijah": 62,
        "patriarch": 55, "elder": 45,
        # Human - general
        "servant": 35, "disciple": 40, "believer": 35,
        "man": 25, "person": 25, "woman": 25,
    }

    # Action reversibility markers
    ETERNAL_ACTIONS = frozenset([
        "eternal", "forever", "everlasting", "perpetual", "αἰώνιος",
        "never", "always", "abides", "remains", "immortal"
    ])

    LASTING_ACTIONS = frozenset([
        "covenant", "promise", "inheritance", "salvation", "redemption",
        "justified", "sanctified", "adopted", "sealed"
    ])

    TEMPORARY_ACTIONS = frozenset([
        "day", "moment", "time", "season", "year", "generation",
        "until", "temporary", "passing", "shadow"
    ])

    def analyze_scope(
        self,
        element: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Scope:
        """
        Determine the scope of a typological element.

        Uses hierarchical scanning from most expansive to least.
        """
        text = self._extract_text(element, context).lower()

        # Check for scope indicators from most expansive to least
        scope_order = [
            Scope.ETERNAL, Scope.COSMIC, Scope.UNIVERSAL,
            Scope.INTERNATIONAL, Scope.NATIONAL, Scope.LOCAL,
            Scope.FAMILIAL, Scope.INDIVIDUAL
        ]

        for scope in scope_order:
            indicators = self.SCOPE_INDICATORS[scope]
            if any(indicator.lower() in text for indicator in indicators):
                return scope

        # Default to INDIVIDUAL if no indicators found
        return Scope.INDIVIDUAL

    def calculate_magnitude(
        self,
        element: Dict[str, Any]
    ) -> float:
        """
        Calculate the magnitude of a typological element (0-100 scale).

        Factors:
        - Agent significance (who is acting)
        - Action reversibility (temporary vs eternal)
        - Effect breadth (narrow vs comprehensive)
        - Ontological weight (creature vs Creator involvement)
        """
        text = element.get("text", element.get("description", "")).lower()

        # Agent significance (0-100)
        agent_score = 25  # Default
        for agent, score in self.AGENT_SIGNIFICANCE.items():
            if agent in text:
                agent_score = max(agent_score, score)

        # Action reversibility (0-1)
        if any(kw in text for kw in self.ETERNAL_ACTIONS):
            reversibility = 1.0
        elif any(kw in text for kw in self.LASTING_ACTIONS):
            reversibility = 0.75
        elif any(kw in text for kw in self.TEMPORARY_ACTIONS):
            reversibility = 0.35
        else:
            reversibility = 0.5

        # Effect breadth (0-1)
        comprehensive = ["all", "every", "complete", "full", "whole", "πᾶς"]
        broad = ["many", "much", "great", "abundant", "πολύς"]
        narrow = ["one", "single", "only", "alone", "εἷς"]

        if any(kw in text for kw in comprehensive):
            breadth = 1.0
        elif any(kw in text for kw in broad):
            breadth = 0.7
        elif any(kw in text for kw in narrow):
            breadth = 0.3
        else:
            breadth = 0.5

        # Ontological weight (0-1) - divine involvement
        divine_markers = ["god", "lord", "spirit", "christ", "θεός", "κύριος"]
        if any(dm in text for dm in divine_markers):
            ontological = 1.0
        else:
            ontological = 0.5

        # Combine factors with weighted average
        magnitude = (
            agent_score * 0.35 +
            (reversibility * 100) * 0.25 +
            (breadth * 100) * 0.20 +
            (ontological * 100) * 0.20
        )

        return min(100.0, max(0.0, magnitude))

    def analyze_fulfillment_completeness(
        self,
        type_elem: Dict[str, Any],
        antitype_elem: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """
        Analyze how completely the antitype fulfills the type.

        Returns:
            Tuple of (completeness score 0-1, list of fulfillment details)
        """
        type_text = self._extract_text(type_elem, {}).lower()
        antitype_text = self._extract_text(antitype_elem, {}).lower()

        # Extract key concepts from type
        type_concepts = self._extract_theological_concepts(type_text)
        antitype_concepts = self._extract_theological_concepts(antitype_text)

        details = []

        if not type_concepts:
            return 0.5, ["Unable to extract concepts from type"]

        # Calculate overlap
        overlap = type_concepts & antitype_concepts
        missing = type_concepts - antitype_concepts
        additional = antitype_concepts - type_concepts

        completeness = len(overlap) / len(type_concepts) if type_concepts else 0.5

        if overlap:
            details.append(f"Fulfilled concepts: {', '.join(list(overlap)[:5])}")
        if missing:
            details.append(f"Unfulfilled concepts: {', '.join(list(missing)[:3])}")
        if additional:
            details.append(f"Escalated concepts: {', '.join(list(additional)[:3])}")
            # Bonus for escalation
            completeness = min(1.0, completeness + 0.1 * min(len(additional), 5))

        return completeness, details

    def _extract_text(
        self,
        element: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Extract text from element or context."""
        if isinstance(element, str):
            return element
        return element.get("text", element.get("description", ""))

    def _extract_theological_concepts(self, text: str) -> Set[str]:
        """Extract key theological concepts from text."""
        concepts = {
            # Sacrifice
            "sacrifice", "blood", "lamb", "altar", "offering", "atonement",
            "propitiation", "θυσία", "αἷμα",
            # Covenant
            "covenant", "promise", "blessing", "inheritance", "oath",
            "διαθήκη", "ברית",
            # Salvation
            "salvation", "redemption", "deliverance", "freedom", "ransom",
            "σωτηρία", "ישועה",
            # Kingship
            "king", "kingdom", "throne", "crown", "reign", "scepter",
            "βασιλεία", "מלך",
            # Priesthood
            "priest", "temple", "worship", "holy", "sanctuary", "offering",
            "ἱερεύς", "כהן",
            # Prophecy
            "prophet", "word", "message", "revelation", "oracle",
            "προφήτης", "נביא",
            # Death/Resurrection
            "death", "resurrection", "life", "eternal", "raise", "grave",
            "θάνατος", "ἀνάστασις",
            # Sin/Forgiveness
            "sin", "forgiveness", "cleansing", "purification", "wash",
            "ἁμαρτία", "חטא",
            # Baptismal
            "water", "baptism", "birth", "new", "immersion", "flood",
            "βάπτισμα", "מים",
            # Eucharistic
            "bread", "body", "wine", "cup", "meal", "feast",
            "ἄρτος", "לחם",
            # Pastoral
            "shepherd", "sheep", "flock", "pasture", "feed",
            "ποιμήν", "רעה",
            # Filial
            "son", "father", "spirit", "servant", "heir",
            "υἱός", "בן",
        }

        found = set()
        text_lower = text.lower()
        for concept in concepts:
            if concept in text_lower:
                found.add(concept)

        return found


# =============================================================================
# SEMANTIC COHERENCE CHECKER - Enhanced
# =============================================================================

class SemanticCoherenceChecker:
    """
    Checks semantic coherence between prophetic promises and fulfillments.

    Enhanced with:
    - Contradiction detection with resolution patterns
    - Extension pattern recognition
    - Semantic entailment scoring
    - Prophetic genre-specific analysis
    """

    # Semantic contradiction pairs - expanded
    CONTRADICTION_PAIRS: List[Tuple[str, str]] = [
        ("life", "death"),
        ("blessing", "curse"),
        ("victory", "defeat"),
        ("salvation", "destruction"),
        ("peace", "war"),
        ("unity", "division"),
        ("freedom", "bondage"),
        ("light", "darkness"),
        ("truth", "lie"),
        ("love", "hate"),
        ("exaltation", "humiliation"),
        ("righteousness", "wickedness"),
        ("mercy", "judgment"),  # Note: can coexist in theology
        ("gathering", "scattering"),
        ("healing", "affliction"),
        ("abundance", "famine"),
        ("presence", "absence"),
        ("covenant", "abandonment"),
    ]

    # Extension patterns (promise → fulfillment expansion)
    EXTENSION_PATTERNS: List[Tuple[str, str]] = [
        ("land", "kingdom"),
        ("nation", "all nations"),
        ("temporal", "eternal"),
        ("physical", "spiritual"),
        ("local", "universal"),
        ("shadow", "reality"),
        ("partial", "complete"),
        ("earthly", "heavenly"),
        ("letter", "spirit"),
        ("flesh", "spirit"),
        ("servant", "son"),
        ("prophet", "christ"),
        ("priest", "high priest"),
        ("king", "king of kings"),
        ("temple", "body"),
        ("sacrifice", "self-offering"),
        ("lamb", "lamb of god"),
        ("moses", "greater than moses"),
        ("david", "son of david"),
        ("adam", "last adam"),
    ]

    # Resolution patterns that resolve apparent contradictions
    RESOLUTION_PATTERNS: List[str] = [
        "through", "by means of", "resulting in", "leading to",
        "in order that", "so that", "that", "for",
        "but", "yet", "however", "nevertheless",
        "transformed", "become", "made", "accomplished"
    ]

    def extract_promise_components(
        self,
        semantics: Dict[str, Any]
    ) -> Set[str]:
        """Extract promise components from semantic analysis."""
        components = set()

        text = semantics.get("text", "").lower()
        themes = semantics.get("themes", [])
        keywords = semantics.get("keywords", [])

        # Add themes and keywords
        components.update(str(t).lower() for t in themes)
        components.update(str(k).lower() for k in keywords)

        # Extract action-object patterns
        promise_markers = [
            "will", "shall", "promise", "give", "make", "establish",
            "swear", "covenant", "surely", "behold"
        ]
        for marker in promise_markers:
            if marker in text:
                components.add(f"promise_{marker}")

        return components

    def extract_fulfillment_claims(
        self,
        semantics: Dict[str, Any]
    ) -> Set[str]:
        """Extract fulfillment claims from semantic analysis."""
        claims = set()

        text = semantics.get("text", "").lower()
        themes = semantics.get("themes", [])
        keywords = semantics.get("keywords", [])

        # Add themes and keywords
        claims.update(str(t).lower() for t in themes)
        claims.update(str(k).lower() for k in keywords)

        # Extract fulfillment markers
        fulfillment_markers = [
            "fulfilled", "completed", "accomplished", "finished",
            "came to pass", "written", "spoken", "prophesied"
        ]
        for marker in fulfillment_markers:
            if marker in text:
                claims.add(f"fulfillment_{marker}")

        return claims

    def detect_contradictions(
        self,
        promise_semantics: Dict[str, Any],
        fulfillment_semantics: Dict[str, Any]
    ) -> List[Tuple[str, str, bool]]:
        """
        Detect semantic contradictions between promise and fulfillment.

        Returns:
            List of (positive_term, negative_term, has_resolution) tuples
        """
        contradictions = []

        promise_text = promise_semantics.get("text", "").lower()
        fulfillment_text = fulfillment_semantics.get("text", "").lower()

        # Check contradiction pairs
        for pos, neg in self.CONTRADICTION_PAIRS:
            if pos in promise_text and neg in fulfillment_text:
                # Check if there's a resolution context
                has_resolution = any(
                    m in fulfillment_text for m in self.RESOLUTION_PATTERNS
                )
                contradictions.append((pos, neg, has_resolution))

            # Also check reverse
            if neg in promise_text and pos in fulfillment_text:
                has_resolution = any(
                    m in fulfillment_text for m in self.RESOLUTION_PATTERNS
                )
                # Reversal might be positive (e.g., death → life)
                contradictions.append((neg, pos, has_resolution))

        return contradictions

    def detect_extensions(
        self,
        promise_semantics: Dict[str, Any],
        fulfillment_semantics: Dict[str, Any]
    ) -> List[Tuple[str, str]]:
        """
        Detect where fulfillment exceeds/extends promise.

        Returns:
            List of (limited_term, expanded_term) tuples
        """
        extensions = []

        promise_text = promise_semantics.get("text", "").lower()
        fulfillment_text = fulfillment_semantics.get("text", "").lower()

        # Check extension patterns
        for limited, expanded in self.EXTENSION_PATTERNS:
            if limited in promise_text and expanded in fulfillment_text:
                extensions.append((limited, expanded))

        return extensions

    def check_entailment(
        self,
        promise_semantics: Dict[str, Any],
        fulfillment_semantics: Dict[str, Any]
    ) -> Tuple[bool, float, List[str]]:
        """
        Check if fulfillment semantically entails promise.

        Returns:
            Tuple of (entails, confidence, reasoning_chain)
        """
        reasoning = []

        promise_components = self.extract_promise_components(promise_semantics)
        fulfillment_claims = self.extract_fulfillment_claims(fulfillment_semantics)

        if not promise_components:
            return True, 0.5, ["Unable to extract promise components"]

        # Calculate overlap
        overlap = promise_components & fulfillment_claims
        coverage = len(overlap) / len(promise_components)
        reasoning.append(f"Semantic coverage: {coverage:.2%}")

        # Check for contradictions
        contradictions = self.detect_contradictions(
            promise_semantics, fulfillment_semantics
        )

        unresolved = [c for c in contradictions if not c[2]]
        resolved = [c for c in contradictions if c[2]]

        if unresolved:
            reasoning.append(f"Unresolved contradictions: {len(unresolved)}")
            return False, 0.2, reasoning

        if resolved:
            reasoning.append(f"Resolved tensions: {len(resolved)} (dialectical fulfillment)")

        # Check for extensions (positive)
        extensions = self.detect_extensions(promise_semantics, fulfillment_semantics)

        if extensions:
            reasoning.append(f"Typological extensions: {len(extensions)}")

        # Calculate final score
        if coverage >= 0.5 and extensions:
            score = min(1.0, coverage + 0.15 * len(extensions))
            return True, score, reasoning

        if coverage >= 0.3:
            return True, coverage, reasoning

        return False, coverage, reasoning


# =============================================================================
# TRINITARIAN GRAMMAR VALIDATOR
# =============================================================================

class TrinitarianGrammarValidator:
    """
    Validates proper Trinitarian language according to conciliar definitions.

    Detects:
    - Subordinationism (Son/Spirit inferior)
    - Modalism (persons confused)
    - Tritheism (three gods)
    - Arianism (Son as creature)
    - Macedonianism (Spirit as creature)
    """

    # Arian/subordinationist markers
    SUBORDINATIONIST_PATTERNS = [
        "created the son", "son was created", "made the son",
        "lesser god", "inferior to the father", "not truly god",
        "there was when he was not", "before the son existed",
        "spirit is a creature", "spirit was made"
    ]

    # Modalist markers
    MODALIST_PATTERNS = [
        "father became son", "son is the father", "father died",
        "no distinction", "same person", "modes of",
        "father suffered", "one person three names"
    ]

    # Proper Trinitarian affirmations
    ORTHODOX_AFFIRMATIONS = [
        "homoousios", "consubstantial", "of one essence",
        "three persons", "three hypostases", "one nature",
        "begotten not made", "proceeds from the father",
        "co-eternal", "co-equal", "trinity", "triune"
    ]

    def validate_trinitarian_language(
        self,
        text: str,
        context: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], List[str]]:
        """
        Validate Trinitarian language in text.

        Returns:
            Tuple of (is_orthodox, heresy_type if detected, evidence)
        """
        text_lower = text.lower()
        evidence = []

        # Check for subordinationism
        for pattern in self.SUBORDINATIONIST_PATTERNS:
            if pattern in text_lower:
                evidence.append(f"Subordinationist pattern: '{pattern}'")
                return False, "subordinationism", evidence

        # Check for modalism
        for pattern in self.MODALIST_PATTERNS:
            if pattern in text_lower:
                evidence.append(f"Modalist pattern: '{pattern}'")
                return False, "modalism", evidence

        # Check for orthodox affirmations (positive)
        for affirmation in self.ORTHODOX_AFFIRMATIONS:
            if affirmation in text_lower:
                evidence.append(f"Orthodox affirmation: '{affirmation}'")

        return True, None, evidence


# =============================================================================
# THEOSIS TRAJECTORY VALIDATOR
# =============================================================================

class TheosisTrajectoryValidator:
    """
    Validates deification themes according to Orthodox soteriology.

    Key principle (Athanasius): "God became man that man might become god"
    Trajectory: Creation → Fall → Purification → Illumination → Deification

    Validates:
    - Proper sequence of stages
    - No confusion of nature (creature becomes God by nature - pantheism)
    - Participation in divine energies, not essence
    """

    # Stage markers
    STAGE_MARKERS: Dict[TheosisStage, List[str]] = {
        TheosisStage.CREATION: [
            "image", "likeness", "created", "formed", "made",
            "εἰκών", "ὁμοίωσις"
        ],
        TheosisStage.FALL: [
            "sin", "fall", "corruption", "death", "darkened",
            "separated", "alienated", "ἁμαρτία", "φθορά"
        ],
        TheosisStage.PURIFICATION: [
            "cleanse", "purify", "repent", "confess", "wash",
            "κάθαρσις", "μετάνοια"
        ],
        TheosisStage.ILLUMINATION: [
            "enlighten", "illuminate", "light", "knowledge", "wisdom",
            "φωτισμός", "γνῶσις"
        ],
        TheosisStage.DEIFICATION: [
            "deification", "theosis", "divinization", "partake",
            "divine nature", "θέωσις", "θεοποίησις"
        ],
        TheosisStage.GLORIFICATION: [
            "glory", "glorified", "resurrection", "incorruption",
            "immortality", "δόξα", "ἀφθαρσία"
        ],
    }

    # Pantheist markers (heretical)
    PANTHEIST_PATTERNS = [
        "become god by nature", "absorbed into god", "lose identity",
        "divine essence", "identical with god", "no distinction"
    ]

    # Orthodox qualifiers
    ORTHODOX_QUALIFIERS = [
        "by grace", "participation", "energies", "not essence",
        "remain creature", "κατὰ χάριν", "μέθεξις"
    ]

    def detect_theosis_stage(self, text: str) -> Optional[TheosisStage]:
        """Detect which theosis stage is referenced."""
        text_lower = text.lower()

        for stage, markers in self.STAGE_MARKERS.items():
            if any(marker.lower() in text_lower for marker in markers):
                return stage

        return None

    def validate_theosis_language(
        self,
        text: str,
        context: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate theosis language is properly Orthodox.

        Returns:
            Tuple of (is_orthodox, evidence)
        """
        text_lower = text.lower()
        evidence = []

        # Check for pantheist patterns (heretical)
        for pattern in self.PANTHEIST_PATTERNS:
            if pattern in text_lower:
                evidence.append(f"Pantheist pattern detected: '{pattern}'")
                return False, evidence

        # Check for orthodox qualifiers (positive)
        has_qualifier = any(
            qual in text_lower for qual in self.ORTHODOX_QUALIFIERS
        )

        # Detect stage
        stage = self.detect_theosis_stage(text)
        if stage:
            evidence.append(f"Theosis stage: {stage.value}")

        if has_qualifier:
            evidence.append("Orthodox qualifier present (by grace, participation)")

        return True, evidence


# =============================================================================
# SACRAMENTAL TYPOLOGY VALIDATOR
# =============================================================================

class SacramentalTypologyValidator:
    """
    Validates sacramental typological connections.

    Key types:
    - BAPTISMAL: Flood, Red Sea, Jordan crossing → Christian Baptism
    - EUCHARISTIC: Melchizedek's bread/wine, Passover, Manna → Eucharist
    - CHRISMATION: Anointing of kings/priests → Gift of Spirit
    """

    # Baptismal type markers
    BAPTISMAL_TYPES: Dict[str, str] = {
        "flood": "Noah's Flood prefigures Baptism (1 Pet 3:20-21)",
        "red sea": "Red Sea crossing prefigures Baptism (1 Cor 10:1-2)",
        "jordan": "Jordan crossing prefigures Baptism (Josh 3)",
        "naaman": "Naaman's cleansing prefigures Baptism (2 Kgs 5)",
        "circumcision": "Circumcision as sign prefigures Baptism (Col 2:11-12)",
        "mikvah": "Ritual immersion prefigures Baptism",
    }

    # Eucharistic type markers
    EUCHARISTIC_TYPES: Dict[str, str] = {
        "melchizedek": "Melchizedek's bread/wine prefigures Eucharist (Gen 14, Heb 7)",
        "passover": "Passover Lamb prefigures Eucharist (Ex 12, 1 Cor 5:7)",
        "manna": "Manna prefigures Eucharist (Ex 16, John 6)",
        "showbread": "Showbread prefigures Eucharist (Lev 24:5-9)",
        "isaac": "Isaac's near-sacrifice prefigures Christ's sacrifice (Gen 22)",
        "todah": "Todah thanksgiving sacrifice prefigures Eucharist",
    }

    # Chrismation type markers
    CHRISMATION_TYPES: Dict[str, str] = {
        "anointing": "Anointing of kings/priests prefigures Chrismation",
        "holy oil": "Sacred anointing oil prefigures Chrism (Ex 30:22-33)",
        "elijah elisha": "Elijah's mantle transfer prefigures Spirit's gift",
        "pentecost": "First Pentecost prefigures Church's Pentecost",
    }

    def identify_sacramental_type(
        self,
        source_text: str,
        target_text: str
    ) -> Optional[Tuple[SacramentalType, str]]:
        """
        Identify sacramental typology in a cross-reference.

        Returns:
            Tuple of (SacramentalType, patristic_support) or None
        """
        source_lower = source_text.lower()

        # Check baptismal types
        for marker, support in self.BAPTISMAL_TYPES.items():
            if marker in source_lower:
                return SacramentalType.BAPTISMAL, support

        # Check eucharistic types
        for marker, support in self.EUCHARISTIC_TYPES.items():
            if marker in source_lower:
                return SacramentalType.EUCHARISTIC, support

        # Check chrismation types
        for marker, support in self.CHRISMATION_TYPES.items():
            if marker in source_lower:
                return SacramentalType.CHRISMATION, support

        return None

    def validate_sacramental_connection(
        self,
        source_text: str,
        target_text: str,
        claimed_type: SacramentalType
    ) -> Tuple[bool, float, List[str]]:
        """
        Validate a claimed sacramental typological connection.

        Returns:
            Tuple of (is_valid, confidence, evidence)
        """
        evidence = []
        detected = self.identify_sacramental_type(source_text, target_text)

        if detected is None:
            evidence.append("No sacramental markers detected in source")
            return False, 0.3, evidence

        detected_type, support = detected
        evidence.append(f"Detected type: {detected_type.value}")
        evidence.append(f"Patristic support: {support}")

        if detected_type == claimed_type:
            return True, 0.9, evidence

        evidence.append(f"Type mismatch: claimed {claimed_type.value}, detected {detected_type.value}")
        return False, 0.5, evidence


# =============================================================================
# ESCHATOLOGICAL COHERENCE VALIDATOR
# =============================================================================

class EschatologicalCoherenceValidator:
    """
    Validates already/not-yet eschatological tension.

    Key principle: Christ's first coming inaugurates the Kingdom;
    His second coming consummates it. Both "already" and "not yet" are true.

    Validates:
    - Proper recognition of inaugurated eschatology
    - No over-realized eschatology (everything now)
    - No postponed eschatology (nothing now)
    """

    # Inaugurated (already) markers
    INAUGURATED_MARKERS = [
        "has come", "is at hand", "fulfilled", "accomplished",
        "in christ", "new creation", "born again", "ἤγγικεν"
    ]

    # Consummated (not yet) markers
    CONSUMMATED_MARKERS = [
        "will come", "shall be", "not yet", "await",
        "hope", "return", "second coming", "parousia"
    ]

    # Over-realized markers (potentially problematic)
    OVER_REALIZED_MARKERS = [
        "already resurrected", "no future judgment",
        "kingdom fully present", "no more death now"
    ]

    def analyze_eschatological_tension(
        self,
        text: str
    ) -> Tuple[EschatologicalTension, List[str]]:
        """
        Analyze the eschatological perspective in text.

        Returns:
            Tuple of (EschatologicalTension, evidence)
        """
        text_lower = text.lower()
        evidence = []

        has_inaugurated = any(m in text_lower for m in self.INAUGURATED_MARKERS)
        has_consummated = any(m in text_lower for m in self.CONSUMMATED_MARKERS)
        has_over_realized = any(m in text_lower for m in self.OVER_REALIZED_MARKERS)

        if has_inaugurated:
            evidence.append("Contains inaugurated (already) markers")
        if has_consummated:
            evidence.append("Contains consummated (not yet) markers")
        if has_over_realized:
            evidence.append("WARNING: Contains over-realized markers")

        # Determine tension type
        if has_inaugurated and has_consummated:
            return EschatologicalTension.PROGRESSIVE, evidence
        elif has_inaugurated and not has_consummated:
            if has_over_realized:
                return EschatologicalTension.REALIZED, evidence
            return EschatologicalTension.INAUGURATED, evidence
        elif has_consummated and not has_inaugurated:
            return EschatologicalTension.CONSUMMATED, evidence
        else:
            return EschatologicalTension.PROGRESSIVE, evidence

    def validate_eschatological_coherence(
        self,
        source_tension: EschatologicalTension,
        target_tension: EschatologicalTension
    ) -> Tuple[bool, float, str]:
        """
        Validate that eschatological tensions are coherent.

        Returns:
            Tuple of (is_coherent, confidence, reason)
        """
        # Over-realized is always problematic
        if source_tension == EschatologicalTension.REALIZED:
            return False, 0.4, "Source shows over-realized eschatology"
        if target_tension == EschatologicalTension.REALIZED:
            return False, 0.4, "Target shows over-realized eschatology"

        # Progressive tension is ideal
        if source_tension == EschatologicalTension.PROGRESSIVE:
            return True, 0.95, "Source maintains proper already/not-yet tension"
        if target_tension == EschatologicalTension.PROGRESSIVE:
            return True, 0.95, "Target maintains proper already/not-yet tension"

        # Inaugurated → Consummated is proper trajectory
        if (source_tension == EschatologicalTension.INAUGURATED and
            target_tension == EschatologicalTension.CONSUMMATED):
            return True, 0.9, "Proper inaugurated → consummated trajectory"

        # Same tension is acceptable
        if source_tension == target_tension:
            return True, 0.8, f"Both have {source_tension.value} perspective"

        return True, 0.7, "Eschatological tensions are compatible"


# =============================================================================
# MAIN VALIDATOR CLASS - Enhanced with 12 Constraints
# =============================================================================

class TheologicalConstraintValidator:
    """
    Validates cross-references against patristic theological constraints.

    Implements "covert theological governance" by encoding Church Father
    principles as algorithmic rules without explicit attribution.

    ENHANCED with 12 core constraints and comprehensive theological validation.
    """

    # OT books (for chronological validation)
    OT_BOOKS = set(BOOK_ORDER[:39]) if len(BOOK_ORDER) >= 39 else set(BOOK_ORDER)

    # NT books
    NT_BOOKS = set(BOOK_ORDER[39:]) if len(BOOK_ORDER) >= 39 else set()

    # Constraint applicability by connection type - expanded
    CONSTRAINT_APPLICABILITY: Dict[str, List[ConstraintType]] = {
        "typological": [
            ConstraintType.TYPOLOGICAL_ESCALATION,
            ConstraintType.CHRONOLOGICAL_PRIORITY,
            ConstraintType.LITURGICAL_AMPLIFICATION,
            ConstraintType.FOURFOLD_FOUNDATION,
            ConstraintType.SACRAMENTAL_TYPOLOGY,
            ConstraintType.THEOSIS_TRAJECTORY,
        ],
        "prophetic": [
            ConstraintType.PROPHETIC_COHERENCE,
            ConstraintType.CHRONOLOGICAL_PRIORITY,
            ConstraintType.CHRISTOLOGICAL_WARRANT,
            ConstraintType.ESCHATOLOGICAL_COHERENCE,
        ],
        "verbal": [
            ConstraintType.LITURGICAL_AMPLIFICATION,
            ConstraintType.CANONICAL_PRIORITY,
        ],
        "thematic": [
            ConstraintType.LITURGICAL_AMPLIFICATION,
            ConstraintType.THEOSIS_TRAJECTORY,
        ],
        "conceptual": [
            ConstraintType.LITURGICAL_AMPLIFICATION,
            ConstraintType.TRINITARIAN_GRAMMAR,
            ConstraintType.CONCILIAR_ALIGNMENT,
        ],
        "historical": [
            ConstraintType.CHRONOLOGICAL_PRIORITY,
        ],
        "narrative": [
            ConstraintType.CHRONOLOGICAL_PRIORITY,
        ],
        "genealogical": [
            ConstraintType.CHRONOLOGICAL_PRIORITY,
        ],
        "geographical": [],
        "liturgical": [
            ConstraintType.LITURGICAL_AMPLIFICATION,
            ConstraintType.SACRAMENTAL_TYPOLOGY,
        ],
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the validator with comprehensive sub-validators.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

        # Initialize sub-validators
        self.scope_analyzer = ScopeMagnitudeAnalyzer()
        self.coherence_checker = SemanticCoherenceChecker()
        self.trinitarian_validator = TrinitarianGrammarValidator()
        self.theosis_validator = TheosisTrajectoryValidator()
        self.sacramental_validator = SacramentalTypologyValidator()
        self.eschatology_validator = EschatologicalCoherenceValidator()

        # Configuration with defaults - all 12 constraints
        self.enable_escalation = self.config.get("enable_escalation_validation", True)
        self.enable_prophetic = self.config.get("enable_prophetic_coherence", True)
        self.enable_chronological = self.config.get("enable_chronological_priority", True)
        self.enable_warrant = self.config.get("enable_christological_warrant", True)
        self.enable_liturgical = self.config.get("enable_liturgical_amplification", True)
        self.enable_fourfold = self.config.get("enable_fourfold_foundation", True)
        self.enable_trinitarian = self.config.get("enable_trinitarian_grammar", True)
        self.enable_theosis = self.config.get("enable_theosis_trajectory", True)
        self.enable_conciliar = self.config.get("enable_conciliar_alignment", True)
        self.enable_canonical = self.config.get("enable_canonical_priority", True)
        self.enable_sacramental = self.config.get("enable_sacramental_typology", True)
        self.enable_eschatological = self.config.get("enable_eschatological_coherence", True)

        # Thresholds
        self.min_patristic_witnesses = self.config.get("minimum_patristic_witnesses", 2)
        self.escalation_critical_threshold = self.config.get("escalation_critical_threshold", 1.0)
        self.escalation_boost_threshold = self.config.get("escalation_boost_threshold", 1.5)
        self.liturgical_boost_factor = self.config.get("liturgical_boost_factor", 1.1)
        self.apostolic_boost_factor = self.config.get("apostolic_boost_factor", 1.3)
        self.patristic_boost_factor = self.config.get("patristic_boost_factor", 1.1)
        self.paschal_boost_factor = self.config.get("paschal_boost_factor", 1.5)

        logger.info(
            f"TheologicalConstraintValidator initialized with 12 constraints enabled"
        )

    # =========================================================================
    # CONSTRAINT 1: CHRONOLOGICAL PRIORITY
    # =========================================================================

    def validate_chronological_priority(
        self,
        type_ref: str,
        antitype_ref: str,
        canon_order: Optional[List[str]] = None
    ) -> ConstraintResult:
        """
        Validate that type chronologically precedes antitype.

        This is a HARD constraint with no exceptions. Types must come
        before antitypes in canonical order.
        """
        if not self.enable_chronological:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.CHRONOLOGICAL_PRIORITY,
                reason="Chronological validation disabled",
            )

        order = canon_order or BOOK_ORDER

        # Extract book codes
        type_book = self._extract_book_code(type_ref)
        antitype_book = self._extract_book_code(antitype_ref)

        if not type_book or not antitype_book:
            return ConstraintResult(
                passed=False,
                constraint_type=ConstraintType.CHRONOLOGICAL_PRIORITY,
                violation_severity=ConstraintViolationSeverity.WARNING,
                confidence_modifier=0.9,
                reason=f"Could not parse book codes: {type_ref}, {antitype_ref}",
                recoverable=True,
            )

        # Get positions in canonical order
        try:
            type_pos = order.index(type_book.upper())
        except ValueError:
            type_pos = -1

        try:
            antitype_pos = order.index(antitype_book.upper())
        except ValueError:
            antitype_pos = -1

        if type_pos == -1 or antitype_pos == -1:
            return ConstraintResult(
                passed=False,
                constraint_type=ConstraintType.CHRONOLOGICAL_PRIORITY,
                violation_severity=ConstraintViolationSeverity.WARNING,
                confidence_modifier=0.9,
                reason=f"Book not in canonical order: {type_book} or {antitype_book}",
                recoverable=True,
            )

        # Type must precede antitype
        if type_pos >= antitype_pos:
            logger.warning(
                f"Chronological violation: {type_ref} >= {antitype_ref}"
            )
            return ConstraintResult(
                passed=False,
                constraint_type=ConstraintType.CHRONOLOGICAL_PRIORITY,
                violation_severity=ConstraintViolationSeverity.IMPOSSIBLE,
                confidence_modifier=0.0,
                reason=f"Type ({type_ref}) does not precede antitype ({antitype_ref})",
                evidence=[f"{type_book}@{type_pos} >= {antitype_book}@{antitype_pos}"],
                recoverable=False,
                theological_reasoning=[
                    "Typology requires temporal priority of the type",
                    "The shadow must precede the substance",
                    "Heilsgeschichte (salvation history) flows forward"
                ],
            )

        return ConstraintResult(
            passed=True,
            constraint_type=ConstraintType.CHRONOLOGICAL_PRIORITY,
            confidence_modifier=1.0,
            reason=f"Type ({type_ref}) correctly precedes antitype ({antitype_ref})",
            evidence=[f"{type_book}@{type_pos} < {antitype_book}@{antitype_pos}"],
        )

    # =========================================================================
    # CONSTRAINT 2: TYPOLOGICAL ESCALATION
    # =========================================================================

    def validate_typological_escalation(
        self,
        type_element: Dict[str, Any],
        antitype_element: Dict[str, Any],
        type_context: Dict[str, Any],
        antitype_context: Dict[str, Any]
    ) -> ConstraintResult:
        """
        Validate that antitype exceeds type in scope and magnitude.

        Universal patristic principle: the antitype must be greater
        than the type it fulfills.
        """
        if not self.enable_escalation:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.TYPOLOGICAL_ESCALATION,
                reason="Escalation validation disabled",
            )

        # Analyze scope
        type_scope = self.scope_analyzer.analyze_scope(type_element, type_context)
        antitype_scope = self.scope_analyzer.analyze_scope(antitype_element, antitype_context)

        # Analyze magnitude
        type_magnitude = self.scope_analyzer.calculate_magnitude(type_element)
        antitype_magnitude = self.scope_analyzer.calculate_magnitude(antitype_element)

        # Analyze fulfillment completeness
        completeness, completeness_details = self.scope_analyzer.analyze_fulfillment_completeness(
            type_element, antitype_element
        )

        # Calculate escalation ratio
        escalation_ratio = antitype_magnitude / max(type_magnitude, 1.0)

        # Scope comparison (eternal > cosmic > universal > ... > individual)
        scope_order = [
            Scope.INDIVIDUAL, Scope.FAMILIAL, Scope.LOCAL, Scope.NATIONAL,
            Scope.INTERNATIONAL, Scope.UNIVERSAL, Scope.COSMIC, Scope.ETERNAL
        ]
        type_scope_idx = scope_order.index(type_scope)
        antitype_scope_idx = scope_order.index(antitype_scope)
        scope_escalation = antitype_scope_idx >= type_scope_idx

        evidence = [
            f"Type scope: {type_scope.value}, magnitude: {type_magnitude:.1f}",
            f"Antitype scope: {antitype_scope.value}, magnitude: {antitype_magnitude:.1f}",
            f"Escalation ratio: {escalation_ratio:.2f}",
            f"Completeness: {completeness:.2f}",
        ]
        evidence.extend(completeness_details)

        theological_reasoning = [
            "Antitype fulfillment must exceed the type (Heb 8:6 - 'better covenant')",
            f"Scope escalation: {type_scope.value} → {antitype_scope.value}",
            f"Magnitude escalation: {type_magnitude:.1f} → {antitype_magnitude:.1f}",
        ]

        # Determine result
        if escalation_ratio < self.escalation_critical_threshold and not scope_escalation:
            return ConstraintResult(
                passed=False,
                constraint_type=ConstraintType.TYPOLOGICAL_ESCALATION,
                violation_severity=ConstraintViolationSeverity.CRITICAL,
                confidence_modifier=0.3,
                reason="Antitype does not exceed type in scope or magnitude",
                evidence=evidence,
                recoverable=True,
                theological_reasoning=theological_reasoning,
                remedy_suggestions=[
                    "Consider if antitype has cosmic/eternal dimensions not captured",
                    "Verify the type-antitype relationship direction",
                ],
            )

        if escalation_ratio < self.escalation_critical_threshold:
            return ConstraintResult(
                passed=False,
                constraint_type=ConstraintType.TYPOLOGICAL_ESCALATION,
                violation_severity=ConstraintViolationSeverity.SOFT,
                confidence_modifier=0.7,
                reason="Antitype magnitude is below critical threshold",
                evidence=evidence,
                recoverable=True,
                theological_reasoning=theological_reasoning,
            )

        if escalation_ratio >= self.escalation_boost_threshold and scope_escalation:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.TYPOLOGICAL_ESCALATION,
                violation_severity=ConstraintViolationSeverity.BOOST,
                confidence_modifier=1.2,
                reason="Strong typological escalation - antitype significantly exceeds type",
                evidence=evidence,
                theological_reasoning=theological_reasoning,
            )

        return ConstraintResult(
            passed=True,
            constraint_type=ConstraintType.TYPOLOGICAL_ESCALATION,
            confidence_modifier=1.0,
            reason="Antitype adequately exceeds type",
            evidence=evidence,
            theological_reasoning=theological_reasoning,
        )

    # =========================================================================
    # CONSTRAINT 3: PROPHETIC COHERENCE
    # =========================================================================

    def validate_prophetic_coherence(
        self,
        promise_verse: str,
        fulfillment_verse: str,
        promise_semantics: Dict[str, Any],
        fulfillment_semantics: Dict[str, Any]
    ) -> ConstraintResult:
        """
        Validate that fulfillment extends/completes promise without contradiction.

        Patristic principle: prophecy fulfillment must extend the promise,
        never contradict it.
        """
        if not self.enable_prophetic:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.PROPHETIC_COHERENCE,
                reason="Prophetic coherence validation disabled",
            )

        # Check entailment with full reasoning
        entails, confidence, reasoning = self.coherence_checker.check_entailment(
            promise_semantics, fulfillment_semantics
        )

        # Check for contradictions
        contradictions = self.coherence_checker.detect_contradictions(
            promise_semantics, fulfillment_semantics
        )

        # Check for extensions
        extensions = self.coherence_checker.detect_extensions(
            promise_semantics, fulfillment_semantics
        )

        evidence = reasoning.copy()
        if extensions:
            evidence.extend([f"Extension: {l} → {e}" for l, e in extensions])

        unresolved_contradictions = [c for c in contradictions if not c[2]]

        if unresolved_contradictions:
            evidence.extend([
                f"Contradiction: {c[0]} vs {c[1]} (unresolved)"
                for c in unresolved_contradictions
            ])
            return ConstraintResult(
                passed=False,
                constraint_type=ConstraintType.PROPHETIC_COHERENCE,
                violation_severity=ConstraintViolationSeverity.CRITICAL,
                confidence_modifier=0.2,
                reason=f"Semantic contradictions between {promise_verse} and {fulfillment_verse}",
                evidence=evidence,
                recoverable=True,
                theological_reasoning=[
                    "Prophecy fulfillment must preserve promise semantics",
                    "Contradictions indicate misidentified fulfillment",
                ],
            )

        if not entails and not extensions:
            return ConstraintResult(
                passed=False,
                constraint_type=ConstraintType.PROPHETIC_COHERENCE,
                violation_severity=ConstraintViolationSeverity.SOFT,
                confidence_modifier=0.7,
                reason="Fulfillment does not clearly entail or extend promise",
                evidence=evidence,
                recoverable=True,
            )

        if extensions:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.PROPHETIC_COHERENCE,
                violation_severity=ConstraintViolationSeverity.BOOST,
                confidence_modifier=1.15,
                reason="Fulfillment extends promise - typological escalation present",
                evidence=evidence,
                theological_reasoning=[
                    "Proper fulfillment exceeds promise (Eph 3:20 - 'exceedingly abundantly')",
                ],
            )

        return ConstraintResult(
            passed=True,
            constraint_type=ConstraintType.PROPHETIC_COHERENCE,
            confidence_modifier=1.0,
            reason="No contradictions detected in prophetic fulfillment",
            evidence=evidence,
        )

    # =========================================================================
    # CONSTRAINT 4: CHRISTOLOGICAL WARRANT
    # =========================================================================

    def validate_christological_warrant(
        self,
        ot_verse: str,
        christological_claim: str,
        nt_quotations: List[str],
        patristic_witnesses: List[str]
    ) -> ConstraintResult:
        """
        Validate that christological OT reading has apostolic/patristic warrant.

        Patristic principle: Christological OT readings require either
        apostolic use (NT quotation) OR patristic consensus.
        """
        if not self.enable_warrant:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.CHRISTOLOGICAL_WARRANT,
                reason="Christological warrant validation disabled",
            )

        evidence = []
        patristic_citations = []

        # Check for apostolic (NT) warrant
        has_apostolic = len(nt_quotations) > 0
        if has_apostolic:
            evidence.append(f"Apostolic warrant: {len(nt_quotations)} NT quotation(s)")
            evidence.extend([f"  - {q}" for q in nt_quotations[:3]])

        # Analyze patristic witnesses with weighting
        weighted_score = 0.0
        major_witnesses = []

        for witness in patristic_witnesses:
            # Look up father classification
            for father_name, (era, weight) in FATHER_CLASSIFICATION.items():
                if father_name.lower() in witness.lower():
                    weighted_score += weight
                    major_witnesses.append(f"{father_name} ({era.value}, w={weight:.2f})")
                    patristic_citations.append(witness)
                    break

        has_patristic_consensus = weighted_score >= self.min_patristic_witnesses

        if major_witnesses:
            evidence.append(f"Patristic support: {len(major_witnesses)} witness(es), weighted score: {weighted_score:.2f}")
            evidence.extend([f"  - {w}" for w in major_witnesses[:5]])

        # Determine result
        if has_apostolic:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.CHRISTOLOGICAL_WARRANT,
                violation_severity=ConstraintViolationSeverity.APOSTOLIC_BOOST,
                confidence_modifier=self.apostolic_boost_factor,
                reason=f"Apostolic warrant for christological reading of {ot_verse}",
                evidence=evidence,
                patristic_citations=patristic_citations,
                theological_reasoning=[
                    "Apostolic interpretation carries highest authority",
                    "NT authors read OT christologically under inspiration",
                ],
            )

        if has_patristic_consensus:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.CHRISTOLOGICAL_WARRANT,
                violation_severity=ConstraintViolationSeverity.BOOST,
                confidence_modifier=self.patristic_boost_factor,
                reason=f"Patristic consensus for christological reading of {ot_verse}",
                evidence=evidence,
                patristic_citations=patristic_citations,
                theological_reasoning=[
                    "Consensus patrum establishes interpretive tradition",
                    f"Weighted score {weighted_score:.2f} exceeds threshold",
                ],
            )

        if patristic_witnesses:
            evidence.append(f"Partial support: {len(patristic_witnesses)} witness(es)")
            return ConstraintResult(
                passed=False,
                constraint_type=ConstraintType.CHRISTOLOGICAL_WARRANT,
                violation_severity=ConstraintViolationSeverity.WARNING,
                confidence_modifier=0.9,
                reason="Christological reading has some support but lacks consensus",
                evidence=evidence,
                recoverable=True,
                remedy_suggestions=[
                    "Search for additional patristic witnesses",
                    "Check for NT allusions (not just quotations)",
                ],
            )

        # No warrant at all
        return ConstraintResult(
            passed=False,
            constraint_type=ConstraintType.CHRISTOLOGICAL_WARRANT,
            violation_severity=ConstraintViolationSeverity.CRITICAL,
            confidence_modifier=0.3,
            reason=f"Novel christological reading of {ot_verse} lacks warrant",
            evidence=["No NT quotations found", "No patristic witnesses found"],
            recoverable=True,
            theological_reasoning=[
                "Novel interpretations without tradition warrant are suspect",
                "Private interpretation risks eisegesis",
            ],
        )

    # =========================================================================
    # CONSTRAINT 5: LITURGICAL AMPLIFICATION
    # =========================================================================

    def validate_liturgical_amplification(
        self,
        verse_ref: str,
        liturgical_contexts: List[str]
    ) -> ConstraintResult:
        """
        Validate and boost connections with liturgical significance.

        Orthodox principle: Liturgical usage amplifies theological weight.
        """
        if not self.enable_liturgical:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.LITURGICAL_AMPLIFICATION,
                reason="Liturgical amplification disabled",
            )

        if not liturgical_contexts:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.LITURGICAL_AMPLIFICATION,
                violation_severity=ConstraintViolationSeverity.NEUTRAL,
                confidence_modifier=1.0,
                reason="No liturgical contexts provided",
            )

        # Calculate boost based on liturgical significance
        max_boost = 1.0
        max_season = None
        evidence = []

        for context in liturgical_contexts:
            context_clean = context.lower().replace(" ", "_").replace("-", "_")

            # Try to match to liturgical season
            for season in LiturgicalSeason:
                if season.value.lower() in context_clean or context_clean in season.value.lower():
                    boost = LITURGICAL_WEIGHTS.get(season, 1.0)
                    if boost > max_boost:
                        max_boost = boost
                        max_season = season
                    evidence.append(f"{context} → {season.value}: boost {boost}")
                    break
            else:
                evidence.append(f"{context}: no specific season match")

        if max_season == LiturgicalSeason.PASCHA:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.LITURGICAL_AMPLIFICATION,
                violation_severity=ConstraintViolationSeverity.PASCHAL_BOOST,
                confidence_modifier=self.paschal_boost_factor,
                reason=f"Paschal liturgical significance for {verse_ref}",
                evidence=evidence,
                theological_reasoning=[
                    "Pascha is the Feast of Feasts",
                    "Paschal usage indicates central theological importance",
                ],
            )

        if max_boost > 1.0:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.LITURGICAL_AMPLIFICATION,
                violation_severity=ConstraintViolationSeverity.BOOST,
                confidence_modifier=min(max_boost, 1.5),
                reason=f"Liturgical amplification for {verse_ref}",
                evidence=evidence,
            )

        return ConstraintResult(
            passed=True,
            constraint_type=ConstraintType.LITURGICAL_AMPLIFICATION,
            confidence_modifier=1.0,
            reason="Liturgical contexts recognized but no significant boost",
            evidence=evidence,
        )

    # =========================================================================
    # CONSTRAINT 6: FOURFOLD FOUNDATION
    # =========================================================================

    def validate_fourfold_foundation(
        self,
        verse_ref: str,
        literal_analysis: Dict[str, Any],
        allegorical_claim: Dict[str, Any]
    ) -> ConstraintResult:
        """
        Validate that allegorical reading has literal foundation.

        Patristic principle (Origen, Cassian): Allegorical interpretation
        requires established literal sense.
        """
        if not self.enable_fourfold:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.FOURFOLD_FOUNDATION,
                reason="Fourfold foundation validation disabled",
            )

        # Check if literal sense is established
        has_literal = bool(literal_analysis)
        literal_complete = (
            literal_analysis.get("text") or
            literal_analysis.get("historical_context") or
            literal_analysis.get("grammatical_analysis")
        ) if has_literal else False

        # Check if allegorical reading exists
        has_allegorical = bool(allegorical_claim)
        allegorical_type = allegorical_claim.get("type", "allegorical") if has_allegorical else None

        evidence = []
        if has_literal:
            evidence.append("Literal sense established")
            if literal_analysis.get("historical_context"):
                evidence.append("Historical context present")
            if literal_analysis.get("grammatical_analysis"):
                evidence.append("Grammatical analysis present")
        else:
            evidence.append("Literal sense not provided")

        if has_allegorical:
            evidence.append(f"Allegorical reading type: {allegorical_type}")

        # If no allegorical claim, no validation needed
        if not has_allegorical:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.FOURFOLD_FOUNDATION,
                confidence_modifier=1.0,
                reason="No allegorical reading to validate",
            )

        # Allegorical without literal foundation
        if has_allegorical and not literal_complete:
            return ConstraintResult(
                passed=False,
                constraint_type=ConstraintType.FOURFOLD_FOUNDATION,
                violation_severity=ConstraintViolationSeverity.WARNING,
                confidence_modifier=0.9,
                reason="Allegorical reading without established literal foundation",
                evidence=evidence,
                recoverable=True,
                theological_reasoning=[
                    "Cassian: 'Spiritual sense is built on literal'",
                    "Allegory without historia risks fantasy",
                ],
                remedy_suggestions=[
                    "Establish literal/historical sense first",
                    "Provide grammatical analysis of the passage",
                ],
            )

        return ConstraintResult(
            passed=True,
            constraint_type=ConstraintType.FOURFOLD_FOUNDATION,
            confidence_modifier=1.0,
            reason="Allegorical reading properly grounded in literal sense",
            evidence=evidence,
        )

    # =========================================================================
    # CONSTRAINT 7: TRINITARIAN GRAMMAR
    # =========================================================================

    def validate_trinitarian_grammar(
        self,
        text: str,
        context: Dict[str, Any]
    ) -> ConstraintResult:
        """
        Validate proper Trinitarian language patterns.

        Checks against conciliar definitions to ensure no Arian,
        modalist, or other heretical patterns.
        """
        if not self.enable_trinitarian:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.TRINITARIAN_GRAMMAR,
                reason="Trinitarian grammar validation disabled",
            )

        is_orthodox, heresy_type, evidence = self.trinitarian_validator.validate_trinitarian_language(
            text, context
        )

        if not is_orthodox:
            # Map heresy to council
            council_map = {
                "subordinationism": ConciliarAuthority.NICAEA_I,
                "arianism": ConciliarAuthority.NICAEA_I,
                "modalism": ConciliarAuthority.CONSTANTINOPLE_I,
                "macedonianism": ConciliarAuthority.CONSTANTINOPLE_I,
            }

            return ConstraintResult(
                passed=False,
                constraint_type=ConstraintType.TRINITARIAN_GRAMMAR,
                violation_severity=ConstraintViolationSeverity.HERETICAL,
                confidence_modifier=0.05,
                reason=f"Trinitarian language violation: {heresy_type}",
                evidence=evidence,
                recoverable=True,
                conciliar_reference=council_map.get(heresy_type),
                theological_reasoning=[
                    f"Pattern matches {heresy_type} heresy",
                    "Contradicts conciliar Trinitarian definitions",
                ],
                remedy_suggestions=[
                    "Review language for proper Trinitarian grammar",
                    "Consult Nicene-Constantinopolitan Creed formulations",
                ],
            )

        # Check for positive orthodox affirmations
        if evidence:  # Has orthodox affirmations
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.TRINITARIAN_GRAMMAR,
                violation_severity=ConstraintViolationSeverity.BOOST,
                confidence_modifier=1.1,
                reason="Orthodox Trinitarian language present",
                evidence=evidence,
            )

        return ConstraintResult(
            passed=True,
            constraint_type=ConstraintType.TRINITARIAN_GRAMMAR,
            confidence_modifier=1.0,
            reason="No Trinitarian grammar violations detected",
            evidence=evidence,
        )

    # =========================================================================
    # CONSTRAINT 8: THEOSIS TRAJECTORY
    # =========================================================================

    def validate_theosis_trajectory(
        self,
        text: str,
        context: Dict[str, Any]
    ) -> ConstraintResult:
        """
        Validate deification themes according to Orthodox soteriology.

        Key principle: Participation in divine energies, not essence.
        """
        if not self.enable_theosis:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.THEOSIS_TRAJECTORY,
                reason="Theosis trajectory validation disabled",
            )

        is_orthodox, evidence = self.theosis_validator.validate_theosis_language(
            text, context
        )

        stage = self.theosis_validator.detect_theosis_stage(text)
        if stage:
            evidence.append(f"Detected theosis stage: {stage.value}")

        if not is_orthodox:
            return ConstraintResult(
                passed=False,
                constraint_type=ConstraintType.THEOSIS_TRAJECTORY,
                violation_severity=ConstraintViolationSeverity.HERETICAL,
                confidence_modifier=0.1,
                reason="Theosis language contains pantheistic elements",
                evidence=evidence,
                recoverable=True,
                theological_reasoning=[
                    "Theosis is by grace, not by nature",
                    "Creature-Creator distinction must be maintained",
                    "Palamas: energies/essence distinction",
                ],
                remedy_suggestions=[
                    "Add qualifier 'by grace' or 'participation'",
                    "Clarify energies vs essence distinction",
                ],
            )

        if stage in [TheosisStage.DEIFICATION, TheosisStage.GLORIFICATION]:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.THEOSIS_TRAJECTORY,
                violation_severity=ConstraintViolationSeverity.BOOST,
                confidence_modifier=1.15,
                reason=f"Theosis trajectory leads to {stage.value}",
                evidence=evidence,
                theological_reasoning=[
                    "Athanasius: 'God became man that man might become god'",
                    f"Stage {stage.value} indicates soteriological depth",
                ],
            )

        return ConstraintResult(
            passed=True,
            constraint_type=ConstraintType.THEOSIS_TRAJECTORY,
            confidence_modifier=1.0,
            reason="Theosis language is properly Orthodox",
            evidence=evidence,
        )

    # =========================================================================
    # CONSTRAINT 9: CONCILIAR ALIGNMENT
    # =========================================================================

    def validate_conciliar_alignment(
        self,
        text: str,
        context: Dict[str, Any]
    ) -> ConstraintResult:
        """
        Check against ecumenical council definitions.

        Ensures no contradiction of dogmatic definitions from the seven councils.
        """
        if not self.enable_conciliar:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.CONCILIAR_ALIGNMENT,
                reason="Conciliar alignment validation disabled",
            )

        text_lower = text.lower()
        evidence = []
        violated_council = None

        # Check against each conciliar definition
        for definition in CONCILIAR_DEFINITIONS:
            # Check if text contains anathematized content
            anathema_lower = definition.anathema.lower()

            # Simple keyword matching for anathema patterns
            anathema_keywords = [
                word.strip() for word in anathema_lower.split()
                if len(word) > 4
            ]

            matches = sum(1 for kw in anathema_keywords if kw in text_lower)
            if matches >= 3:  # Multiple keyword matches suggest violation
                evidence.append(f"Potential violation of {definition.council.value} ({definition.year})")
                evidence.append(f"Topic: {definition.topic}")
                evidence.append(f"Anathema: {definition.anathema}")
                violated_council = definition.council
                break

            # Check for positive affirmations
            for term in definition.key_terms:
                if term.lower() in text_lower:
                    evidence.append(f"Affirms {definition.council.value} term: '{term}'")

        if violated_council:
            return ConstraintResult(
                passed=False,
                constraint_type=ConstraintType.CONCILIAR_ALIGNMENT,
                violation_severity=ConstraintViolationSeverity.HERETICAL,
                confidence_modifier=0.05,
                reason=f"Potential contradiction of {violated_council.value}",
                evidence=evidence,
                recoverable=True,
                conciliar_reference=violated_council,
                theological_reasoning=[
                    "Ecumenical councils define dogmatic boundaries",
                    "Anathematized positions are rejected by consensus",
                ],
            )

        if evidence:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.CONCILIAR_ALIGNMENT,
                violation_severity=ConstraintViolationSeverity.BOOST,
                confidence_modifier=1.1,
                reason="Text aligns with conciliar definitions",
                evidence=evidence,
            )

        return ConstraintResult(
            passed=True,
            constraint_type=ConstraintType.CONCILIAR_ALIGNMENT,
            confidence_modifier=1.0,
            reason="No conciliar violations detected",
        )

    # =========================================================================
    # CONSTRAINT 10: CANONICAL PRIORITY
    # =========================================================================

    def validate_canonical_priority(
        self,
        source_text: str,
        lxx_reading: Optional[str],
        mt_reading: Optional[str],
        nt_quotation: Optional[str]
    ) -> ConstraintResult:
        """
        Handle LXX/MT divergence with apostolic preference.

        Orthodox principle: When NT quotes OT, apostolic text choice
        takes precedence.
        """
        if not self.enable_canonical:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.CANONICAL_PRIORITY,
                reason="Canonical priority validation disabled",
            )

        evidence = []

        # If NT quotation exists, determine which text it follows
        if nt_quotation:
            nt_lower = nt_quotation.lower()
            lxx_lower = (lxx_reading or "").lower()
            mt_lower = (mt_reading or "").lower()

            # Simple similarity check
            if lxx_reading and mt_reading:
                lxx_words = set(lxx_lower.split())
                mt_words = set(mt_lower.split())
                nt_words = set(nt_lower.split())

                lxx_overlap = len(nt_words & lxx_words) / max(len(nt_words), 1)
                mt_overlap = len(nt_words & mt_words) / max(len(nt_words), 1)

                if lxx_overlap > mt_overlap:
                    evidence.append(f"NT follows LXX (overlap: {lxx_overlap:.2%} vs MT: {mt_overlap:.2%})")
                    return ConstraintResult(
                        passed=True,
                        constraint_type=ConstraintType.CANONICAL_PRIORITY,
                        violation_severity=ConstraintViolationSeverity.BOOST,
                        confidence_modifier=1.15,
                        reason="Apostolic preference for LXX confirmed",
                        evidence=evidence,
                        theological_reasoning=[
                            "NT authors frequently follow LXX",
                            "Apostolic text choice indicates inspired reading",
                        ],
                    )
                elif mt_overlap > lxx_overlap:
                    evidence.append(f"NT follows MT (overlap: {mt_overlap:.2%} vs LXX: {lxx_overlap:.2%})")
                    return ConstraintResult(
                        passed=True,
                        constraint_type=ConstraintType.CANONICAL_PRIORITY,
                        confidence_modifier=1.1,
                        reason="Apostolic preference for MT tradition confirmed",
                        evidence=evidence,
                    )

        # No NT quotation to determine preference
        if lxx_reading and mt_reading and lxx_reading != mt_reading:
            evidence.append("LXX/MT divergence present, no NT quotation to resolve")
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.CANONICAL_PRIORITY,
                violation_severity=ConstraintViolationSeverity.WARNING,
                confidence_modifier=0.95,
                reason="LXX/MT divergence without apostolic resolution",
                evidence=evidence,
                theological_reasoning=[
                    "Both textual traditions have value",
                    "LXX was Scripture of the early Church",
                ],
            )

        return ConstraintResult(
            passed=True,
            constraint_type=ConstraintType.CANONICAL_PRIORITY,
            confidence_modifier=1.0,
            reason="No canonical priority issues detected",
        )

    # =========================================================================
    # CONSTRAINT 11: SACRAMENTAL TYPOLOGY
    # =========================================================================

    def validate_sacramental_typology(
        self,
        source_text: str,
        target_text: str,
        claimed_type: Optional[SacramentalType]
    ) -> ConstraintResult:
        """
        Validate sacramental typological connections.

        Checks baptismal, eucharistic, and other sacramental prefigurations.
        """
        if not self.enable_sacramental:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.SACRAMENTAL_TYPOLOGY,
                reason="Sacramental typology validation disabled",
            )

        detected = self.sacramental_validator.identify_sacramental_type(
            source_text, target_text
        )

        evidence = []

        if detected:
            detected_type, support = detected
            evidence.append(f"Detected sacramental type: {detected_type.value}")
            evidence.append(f"Patristic support: {support}")

            if claimed_type and detected_type == claimed_type:
                return ConstraintResult(
                    passed=True,
                    constraint_type=ConstraintType.SACRAMENTAL_TYPOLOGY,
                    violation_severity=ConstraintViolationSeverity.BOOST,
                    confidence_modifier=1.2,
                    reason=f"Confirmed {detected_type.value} typology",
                    evidence=evidence,
                    theological_reasoning=[
                        f"{detected_type.value} type has patristic warrant",
                        support,
                    ],
                )

            if claimed_type and detected_type != claimed_type:
                evidence.append(f"Mismatch: claimed {claimed_type.value}")
                return ConstraintResult(
                    passed=False,
                    constraint_type=ConstraintType.SACRAMENTAL_TYPOLOGY,
                    violation_severity=ConstraintViolationSeverity.SOFT,
                    confidence_modifier=0.8,
                    reason=f"Sacramental type mismatch",
                    evidence=evidence,
                    recoverable=True,
                )

            # Detected but not claimed
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.SACRAMENTAL_TYPOLOGY,
                violation_severity=ConstraintViolationSeverity.BOOST,
                confidence_modifier=1.15,
                reason=f"Unclaimed {detected_type.value} typology detected",
                evidence=evidence,
            )

        # Nothing detected
        if claimed_type:
            evidence.append(f"Claimed {claimed_type.value} but not detected")
            return ConstraintResult(
                passed=False,
                constraint_type=ConstraintType.SACRAMENTAL_TYPOLOGY,
                violation_severity=ConstraintViolationSeverity.WARNING,
                confidence_modifier=0.9,
                reason="Claimed sacramental typology not confirmed",
                evidence=evidence,
                recoverable=True,
            )

        return ConstraintResult(
            passed=True,
            constraint_type=ConstraintType.SACRAMENTAL_TYPOLOGY,
            confidence_modifier=1.0,
            reason="No sacramental typology to validate",
        )

    # =========================================================================
    # CONSTRAINT 12: ESCHATOLOGICAL COHERENCE
    # =========================================================================

    def validate_eschatological_coherence(
        self,
        source_text: str,
        target_text: str
    ) -> ConstraintResult:
        """
        Validate already/not-yet eschatological tension.

        Ensures proper recognition of inaugurated eschatology.
        """
        if not self.enable_eschatological:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.ESCHATOLOGICAL_COHERENCE,
                reason="Eschatological coherence validation disabled",
            )

        source_tension, source_evidence = self.eschatology_validator.analyze_eschatological_tension(
            source_text
        )
        target_tension, target_evidence = self.eschatology_validator.analyze_eschatological_tension(
            target_text
        )

        is_coherent, confidence, reason = self.eschatology_validator.validate_eschatological_coherence(
            source_tension, target_tension
        )

        evidence = [
            f"Source: {source_tension.value}",
            f"Target: {target_tension.value}",
        ]
        evidence.extend(source_evidence)
        evidence.extend(target_evidence)

        if not is_coherent:
            return ConstraintResult(
                passed=False,
                constraint_type=ConstraintType.ESCHATOLOGICAL_COHERENCE,
                violation_severity=ConstraintViolationSeverity.WARNING,
                confidence_modifier=confidence,
                reason=reason,
                evidence=evidence,
                recoverable=True,
                theological_reasoning=[
                    "Proper eschatology maintains already/not-yet tension",
                    "Over-realized eschatology is problematic (2 Tim 2:18)",
                ],
            )

        if source_tension == EschatologicalTension.PROGRESSIVE:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.ESCHATOLOGICAL_COHERENCE,
                violation_severity=ConstraintViolationSeverity.BOOST,
                confidence_modifier=1.1,
                reason="Proper already/not-yet eschatological tension",
                evidence=evidence,
            )

        return ConstraintResult(
            passed=True,
            constraint_type=ConstraintType.ESCHATOLOGICAL_COHERENCE,
            confidence_modifier=confidence,
            reason=reason,
            evidence=evidence,
        )

    # =========================================================================
    # MAIN VALIDATION METHODS
    # =========================================================================

    async def validate_all_constraints(
        self,
        candidate: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[ConstraintResult]:
        """
        Run all applicable constraints for a candidate.

        Args:
            candidate: Cross-reference candidate
            context: Pipeline context

        Returns:
            List of all constraint evaluation results
        """
        results = []

        connection_type = candidate.get("connection_type", "thematic")
        source_verse = candidate.get("source_verse", candidate.get("source_ref", ""))
        target_verse = candidate.get("target_verse", candidate.get("target_ref", ""))
        source_text = candidate.get("source_text", "")
        target_text = candidate.get("target_text", "")

        # Get applicable constraints for this connection type
        applicable = self.CONSTRAINT_APPLICABILITY.get(connection_type, [])

        # Always check chronological for typological/prophetic
        if connection_type in ["typological", "prophetic"]:
            result = self.validate_chronological_priority(source_verse, target_verse)
            results.append(result)

            # If chronological fails with IMPOSSIBLE, short-circuit
            if (not result.passed and
                result.violation_severity == ConstraintViolationSeverity.IMPOSSIBLE):
                return results

        # Typological escalation
        if ConstraintType.TYPOLOGICAL_ESCALATION in applicable:
            type_elem = context.get("type_element", {"text": source_text})
            antitype_elem = context.get("antitype_element", {"text": target_text})
            result = self.validate_typological_escalation(
                type_elem, antitype_elem,
                context.get("type_context", {}),
                context.get("antitype_context", {})
            )
            results.append(result)

        # Prophetic coherence
        if ConstraintType.PROPHETIC_COHERENCE in applicable:
            result = self.validate_prophetic_coherence(
                source_verse, target_verse,
                context.get("promise_semantics", {"text": source_text}),
                context.get("fulfillment_semantics", {"text": target_text})
            )
            results.append(result)

        # Christological warrant
        if ConstraintType.CHRISTOLOGICAL_WARRANT in applicable:
            result = self.validate_christological_warrant(
                source_verse,
                context.get("christological_claim", ""),
                context.get("nt_quotations", []),
                context.get("patristic_witnesses", [])
            )
            results.append(result)

        # Liturgical amplification
        if ConstraintType.LITURGICAL_AMPLIFICATION in applicable:
            result = self.validate_liturgical_amplification(
                source_verse,
                context.get("liturgical_contexts", [])
            )
            results.append(result)

        # Fourfold foundation
        if ConstraintType.FOURFOLD_FOUNDATION in applicable:
            result = self.validate_fourfold_foundation(
                source_verse,
                context.get("literal_analysis", {}),
                context.get("allegorical_claim", {})
            )
            results.append(result)

        # Trinitarian grammar
        if ConstraintType.TRINITARIAN_GRAMMAR in applicable:
            combined_text = f"{source_text} {target_text}"
            result = self.validate_trinitarian_grammar(combined_text, context)
            results.append(result)

        # Theosis trajectory
        if ConstraintType.THEOSIS_TRAJECTORY in applicable:
            combined_text = f"{source_text} {target_text}"
            result = self.validate_theosis_trajectory(combined_text, context)
            results.append(result)

        # Conciliar alignment
        if ConstraintType.CONCILIAR_ALIGNMENT in applicable:
            combined_text = f"{source_text} {target_text}"
            result = self.validate_conciliar_alignment(combined_text, context)
            results.append(result)

        # Canonical priority
        if ConstraintType.CANONICAL_PRIORITY in applicable:
            result = self.validate_canonical_priority(
                source_text,
                context.get("lxx_reading"),
                context.get("mt_reading"),
                context.get("nt_quotation")
            )
            results.append(result)

        # Sacramental typology
        if ConstraintType.SACRAMENTAL_TYPOLOGY in applicable:
            result = self.validate_sacramental_typology(
                source_text, target_text,
                context.get("sacramental_type")
            )
            results.append(result)

        # Eschatological coherence
        if ConstraintType.ESCHATOLOGICAL_COHERENCE in applicable:
            result = self.validate_eschatological_coherence(source_text, target_text)
            results.append(result)

        return results

    def calculate_composite_modifier(
        self,
        results: List[ConstraintResult]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate composite confidence modifier from all constraint results.

        IMPOSSIBLE/HERETICAL → return near-zero immediately
        Otherwise, weighted product of modifiers

        Returns:
            Tuple of (composite_modifier, breakdown_dict)
        """
        if not results:
            return 1.0, {"no_constraints": True}

        breakdown = {
            "total_constraints": len(results),
            "passed": 0,
            "failed": 0,
            "boosts": 0,
            "by_type": {},
        }

        # Check for absolute failures first
        for result in results:
            if not result.passed:
                if result.violation_severity in [
                    ConstraintViolationSeverity.IMPOSSIBLE,
                    ConstraintViolationSeverity.HERETICAL
                ]:
                    logger.warning(
                        f"{result.violation_severity.value} violation in "
                        f"{result.constraint_type.value}: {result.reason}"
                    )
                    breakdown["fatal_violation"] = result.constraint_type.value
                    return result.confidence_modifier, breakdown

        # Calculate weighted product
        composite = 1.0
        for result in results:
            composite *= result.confidence_modifier

            # Track breakdown
            breakdown["by_type"][result.constraint_type.value] = {
                "passed": result.passed,
                "modifier": result.confidence_modifier,
                "severity": result.violation_severity.value if result.violation_severity else None,
            }

            if result.passed:
                breakdown["passed"] += 1
                if result.violation_severity in [
                    ConstraintViolationSeverity.BOOST,
                    ConstraintViolationSeverity.APOSTOLIC_BOOST,
                    ConstraintViolationSeverity.PASCHAL_BOOST
                ]:
                    breakdown["boosts"] += 1
            else:
                breakdown["failed"] += 1

        # Apply floor and ceiling
        composite = max(0.1, min(2.0, composite))
        breakdown["composite"] = composite

        logger.debug(f"Composite modifier: {composite:.3f} from {len(results)} constraints")
        return composite, breakdown

    def _extract_book_code(self, verse_ref: str) -> Optional[str]:
        """Extract book code from verse reference."""
        if not verse_ref:
            return None

        # Handle formats: "GEN.1.1", "GEN 1:1", "Genesis 1:1"
        parts = re.split(r'[.\s:]+', verse_ref)
        if parts:
            return parts[0].upper()[:3]
        return None
