"""
Theological Test Framework

Framework for testing theological assertions with confidence levels,
patristic authority weighting, and Orthodox tradition validation.
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any
from datetime import datetime


class TheologicalConfidence(Enum):
    """
    Confidence levels for theological assertions, mapped to Orthodox tradition authority.
    Higher confidence requires higher pass rates.
    """
    DOGMATIC = "dogmatic"          # Ecumenical council defined (7 councils)
    CONSENSUS = "consensus"        # Universal patristic agreement
    MAJORITY = "majority"          # Most Fathers agree (>80%)
    TRADITIONAL = "traditional"    # Well-established interpretation
    SCHOLARLY = "scholarly"        # Academic consensus
    EXPLORATORY = "exploratory"    # Novel or minority view

    @property
    def required_pass_rate(self) -> float:
        """Minimum pass rate for this confidence level."""
        return {
            TheologicalConfidence.DOGMATIC: 1.0,      # Must always pass
            TheologicalConfidence.CONSENSUS: 0.98,    # Near-perfect
            TheologicalConfidence.MAJORITY: 0.95,     # Very high
            TheologicalConfidence.TRADITIONAL: 0.90,  # High
            TheologicalConfidence.SCHOLARLY: 0.85,    # Good
            TheologicalConfidence.EXPLORATORY: 0.75,  # Acceptable
        }[self]

    @property
    def description(self) -> str:
        """Description of this confidence level."""
        return {
            TheologicalConfidence.DOGMATIC: "Defined by Ecumenical Councils (Nicaea, Constantinople, etc.)",
            TheologicalConfidence.CONSENSUS: "Universal agreement among Church Fathers",
            TheologicalConfidence.MAJORITY: "Majority patristic opinion (>80% agreement)",
            TheologicalConfidence.TRADITIONAL: "Well-established traditional interpretation",
            TheologicalConfidence.SCHOLARLY: "Modern Orthodox academic consensus",
            TheologicalConfidence.EXPLORATORY: "Novel interpretation or minority view",
        }[self]


class TheologicalCategory(Enum):
    """Categories of theological assertions for validation."""
    CHRISTOLOGICAL = "christological"    # Christ's person, natures, work
    TRINITARIAN = "trinitarian"          # Father, Son, Holy Spirit relations
    SOTERIOLOGICAL = "soteriological"    # Salvation, redemption, grace
    ESCHATOLOGICAL = "eschatological"    # Last things, judgment, resurrection
    ECCLESIOLOGICAL = "ecclesiological"  # Church, sacraments, tradition
    TYPOLOGICAL = "typological"          # OT types, NT antitypes
    MARIOLOGICAL = "mariological"        # Theotokos, Ever-Virgin
    PROPHETIC = "prophetic"              # Prophecy and fulfillment
    LITURGICAL = "liturgical"            # Worship, hymnography
    MORAL = "moral"                      # Ethics, virtue, sin

    @property
    def weight(self) -> float:
        """Importance weight for this category."""
        return {
            TheologicalCategory.CHRISTOLOGICAL: 1.0,
            TheologicalCategory.TRINITARIAN: 1.0,
            TheologicalCategory.SOTERIOLOGICAL: 0.95,
            TheologicalCategory.ESCHATOLOGICAL: 0.90,
            TheologicalCategory.ECCLESIOLOGICAL: 0.90,
            TheologicalCategory.TYPOLOGICAL: 0.85,
            TheologicalCategory.MARIOLOGICAL: 0.85,
            TheologicalCategory.PROPHETIC: 0.80,
            TheologicalCategory.LITURGICAL: 0.75,
            TheologicalCategory.MORAL: 0.75,
        }[self]


class PatristicAuthority(Enum):
    """Authority weighting for Church Fathers and theological sources."""
    ECUMENICAL_FATHER = "ecumenical_father"  # Fathers of Ecumenical Councils
    GREAT_FATHER = "great_father"            # Great Doctors (Basil, Gregory, etc.)
    MAJOR_FATHER = "major_father"            # Major theologians
    MINOR_FATHER = "minor_father"            # Lesser-known Fathers
    DISPUTED = "disputed"                     # Disputed or heterodox sources

    @property
    def weight(self) -> float:
        """Authority weight for this patristic source."""
        return {
            PatristicAuthority.ECUMENICAL_FATHER: 1.0,
            PatristicAuthority.GREAT_FATHER: 0.9,
            PatristicAuthority.MAJOR_FATHER: 0.7,
            PatristicAuthority.MINOR_FATHER: 0.4,
            PatristicAuthority.DISPUTED: 0.2,
        }[self]


# Canonical mapping of Church Fathers to authority levels
PATRISTIC_AUTHORITY_MAP: Dict[str, PatristicAuthority] = {
    # Ecumenical Fathers (present at councils, universally recognized)
    "Athanasius": PatristicAuthority.ECUMENICAL_FATHER,
    "Basil the Great": PatristicAuthority.ECUMENICAL_FATHER,
    "Gregory of Nazianzus": PatristicAuthority.ECUMENICAL_FATHER,
    "Gregory of Nyssa": PatristicAuthority.ECUMENICAL_FATHER,
    "John Chrysostom": PatristicAuthority.ECUMENICAL_FATHER,
    "Cyril of Alexandria": PatristicAuthority.ECUMENICAL_FATHER,
    "Maximus the Confessor": PatristicAuthority.ECUMENICAL_FATHER,
    "John of Damascus": PatristicAuthority.ECUMENICAL_FATHER,

    # Great Fathers (universally revered, major theological contributions)
    "Ignatius of Antioch": PatristicAuthority.GREAT_FATHER,
    "Irenaeus of Lyons": PatristicAuthority.GREAT_FATHER,
    "Clement of Alexandria": PatristicAuthority.GREAT_FATHER,
    "Origen": PatristicAuthority.GREAT_FATHER,  # Despite some disputed teachings
    "Augustine of Hippo": PatristicAuthority.GREAT_FATHER,
    "Ambrose of Milan": PatristicAuthority.GREAT_FATHER,
    "Jerome": PatristicAuthority.GREAT_FATHER,
    "Ephrem the Syrian": PatristicAuthority.GREAT_FATHER,

    # Major Fathers (significant but less universal authority)
    "Justin Martyr": PatristicAuthority.MAJOR_FATHER,
    "Tertullian": PatristicAuthority.MAJOR_FATHER,
    "Cyprian of Carthage": PatristicAuthority.MAJOR_FATHER,
    "Hilary of Poitiers": PatristicAuthority.MAJOR_FATHER,
    "Cyril of Jerusalem": PatristicAuthority.MAJOR_FATHER,
    "Theodore of Mopsuestia": PatristicAuthority.MAJOR_FATHER,

    # Minor Fathers (limited influence or local significance)
    "Didymus the Blind": PatristicAuthority.MINOR_FATHER,
    "Theodoret of Cyrus": PatristicAuthority.MINOR_FATHER,

    # Disputed (heterodox or problematic teachings)
    "Apollinaris": PatristicAuthority.DISPUTED,
    "Nestorius": PatristicAuthority.DISPUTED,
}


@dataclass
class PatristicWitness:
    """A patristic witness supporting a theological assertion."""
    father_name: str
    authority: PatristicAuthority
    quote: str
    source_work: str
    confidence: float = 1.0  # 0-1, how directly the quote supports the assertion

    @property
    def weighted_confidence(self) -> float:
        """Confidence weighted by patristic authority."""
        return self.confidence * self.authority.weight


@dataclass
class TheologicalTestCase:
    """
    Structured theological test case with confidence levels and patristic support.
    """
    name: str
    category: TheologicalCategory
    confidence: TheologicalConfidence
    description: str

    # Verse connections
    source_verse: str
    expected_connections: List[str]

    # Theological assertions
    expected_connection_types: List[str] = field(default_factory=list)
    min_confidence: float = 0.7

    # Patristic support
    patristic_witnesses: List[PatristicWitness] = field(default_factory=list)

    # Validation criteria
    require_all_connections: bool = False  # All expected_connections must be found
    allow_additional_connections: bool = True  # Can find more than expected

    # Metadata
    ecumenical_council_support: Optional[str] = None  # e.g., "Nicaea I (325)"
    liturgical_support: Optional[str] = None  # e.g., "Great Friday Vespers"
    notes: str = ""

    def __post_init__(self):
        """Validate test case configuration."""
        if self.confidence == TheologicalConfidence.DOGMATIC:
            if not self.ecumenical_council_support:
                raise ValueError(f"DOGMATIC test case '{self.name}' requires ecumenical_council_support")

        if self.confidence in {TheologicalConfidence.DOGMATIC, TheologicalConfidence.CONSENSUS}:
            if not self.patristic_witnesses:
                raise ValueError(f"Test case '{self.name}' at {self.confidence.value} level requires patristic_witnesses")

    @property
    def patristic_consensus_score(self) -> float:
        """Calculate patristic consensus score based on weighted witnesses."""
        if not self.patristic_witnesses:
            return 0.0

        total_weight = sum(w.weighted_confidence for w in self.patristic_witnesses)
        max_weight = len(self.patristic_witnesses) * 1.0  # Max if all ECUMENICAL_FATHER at 1.0
        return total_weight / max_weight if max_weight > 0 else 0.0

    @property
    def required_pass_rate(self) -> float:
        """Get required pass rate for this test case."""
        return self.confidence.required_pass_rate


# =============================================================================
# CANONICAL TEST CASES
# =============================================================================

CANONICAL_TEST_CASES = [
    # CHRISTOLOGICAL - Highest confidence
    TheologicalTestCase(
        name="genesis_logos_connection",
        category=TheologicalCategory.CHRISTOLOGICAL,
        confidence=TheologicalConfidence.CONSENSUS,
        description="Genesis creation 'In the beginning' connects to Johannine Logos theology",
        source_verse="GEN.1.1",
        expected_connections=["JHN.1.1", "JHN.1.3", "COL.1.16", "HEB.1.2"],
        expected_connection_types=["typological", "thematic", "conceptual"],
        min_confidence=0.85,
        patristic_witnesses=[
            PatristicWitness(
                father_name="Basil the Great",
                authority=PatristicAuthority.ECUMENICAL_FATHER,
                quote="'In the beginning' signifies the co-eternal Son through whom all was made",
                source_work="Hexaemeron, Homily 1",
                confidence=0.95
            ),
            PatristicWitness(
                father_name="John Chrysostom",
                authority=PatristicAuthority.ECUMENICAL_FATHER,
                quote="The Word present in Genesis is the same Logos incarnate in John",
                source_work="Homilies on Genesis",
                confidence=0.90
            ),
        ],
        liturgical_support="Byzantine Liturgy, Paschal readings",
        notes="Foundational Christological reading of OT creation"
    ),

    TheologicalTestCase(
        name="virgin_birth_prophecy",
        category=TheologicalCategory.CHRISTOLOGICAL,
        confidence=TheologicalConfidence.DOGMATIC,
        description="Isaiah 7:14 virgin birth prophecy fulfilled in Matthew 1:23",
        source_verse="ISA.7.14",
        expected_connections=["MAT.1.23", "LUK.1.27", "LUK.1.34-35"],
        expected_connection_types=["prophetic", "typological"],
        min_confidence=0.95,
        patristic_witnesses=[
            PatristicWitness(
                father_name="Justin Martyr",
                authority=PatristicAuthority.MAJOR_FATHER,
                quote="The virgin shall conceive refers to Mary, mother of Christ",
                source_work="Dialogue with Trypho, 84",
                confidence=1.0
            ),
            PatristicWitness(
                father_name="Irenaeus of Lyons",
                authority=PatristicAuthority.GREAT_FATHER,
                quote="Isaiah prophesied the virgin birth of Emmanuel",
                source_work="Against Heresies, III.21.4",
                confidence=1.0
            ),
        ],
        ecumenical_council_support="Ephesus (431) - Theotokos defined",
        liturgical_support="Feast of Annunciation (March 25)",
        require_all_connections=True,
        notes="Core Mariological and Christological prophecy"
    ),

    TheologicalTestCase(
        name="isaac_christ_sacrifice",
        category=TheologicalCategory.TYPOLOGICAL,
        confidence=TheologicalConfidence.CONSENSUS,
        description="Isaac's near-sacrifice (Akedah) as type of Christ's crucifixion",
        source_verse="GEN.22.2",
        expected_connections=["JHN.3.16", "ROM.8.32", "HEB.11.17-19", "JHN.19.17"],
        expected_connection_types=["typological", "thematic"],
        min_confidence=0.80,
        patristic_witnesses=[
            PatristicWitness(
                father_name="Origen",
                authority=PatristicAuthority.GREAT_FATHER,
                quote="Isaac carrying wood prefigures Christ bearing the Cross",
                source_work="Homilies on Genesis, VIII",
                confidence=0.90
            ),
            PatristicWitness(
                father_name="Cyril of Alexandria",
                authority=PatristicAuthority.ECUMENICAL_FATHER,
                quote="Abraham's only beloved son is type of God's only Son",
                source_work="Glaphyra on Genesis",
                confidence=0.95
            ),
        ],
        liturgical_support="Great Friday readings",
        notes="Classic patristic typology, universally recognized"
    ),

    TheologicalTestCase(
        name="passover_lamb_christ",
        category=TheologicalCategory.TYPOLOGICAL,
        confidence=TheologicalConfidence.CONSENSUS,
        description="Passover lamb as type of Christ the Lamb of God",
        source_verse="EXO.12.3",
        expected_connections=["JHN.1.29", "1CO.5.7", "1PE.1.19", "REV.5.6"],
        expected_connection_types=["typological", "prophetic"],
        min_confidence=0.90,
        patristic_witnesses=[
            PatristicWitness(
                father_name="John Chrysostom",
                authority=PatristicAuthority.ECUMENICAL_FATHER,
                quote="The Passover lamb was a clear type of Christ our Passover",
                source_work="Homilies on 1 Corinthians",
                confidence=1.0
            ),
            PatristicWitness(
                father_name="Athanasius",
                authority=PatristicAuthority.ECUMENICAL_FATHER,
                quote="Christ is the true Passover Lamb slain for us",
                source_work="Festal Letters",
                confidence=1.0
            ),
        ],
        liturgical_support="Paschal Liturgy, Holy Week",
        require_all_connections=True,
        notes="Central to Paschal theology"
    ),

    # SOTERIOLOGICAL
    TheologicalTestCase(
        name="serpent_lifted_up",
        category=TheologicalCategory.SOTERIOLOGICAL,
        confidence=TheologicalConfidence.CONSENSUS,
        description="Bronze serpent lifted up as type of crucifixion for salvation",
        source_verse="NUM.21.9",
        expected_connections=["JHN.3.14-15", "JHN.12.32"],
        expected_connection_types=["typological"],
        min_confidence=0.85,
        patristic_witnesses=[
            PatristicWitness(
                father_name="Augustine of Hippo",
                authority=PatristicAuthority.GREAT_FATHER,
                quote="As Moses lifted the serpent, so must the Son of Man be lifted up",
                source_work="Tractates on John",
                confidence=0.95
            ),
        ],
        liturgical_support="Feast of the Exaltation of the Cross (September 14)",
        notes="Direct typology stated by Christ himself"
    ),

    TheologicalTestCase(
        name="adam_christ_typology",
        category=TheologicalCategory.SOTERIOLOGICAL,
        confidence=TheologicalConfidence.CONSENSUS,
        description="Adam as type of Christ (first Adam/last Adam)",
        source_verse="GEN.2.7",
        expected_connections=["ROM.5.14", "1CO.15.45", "1CO.15.22"],
        expected_connection_types=["typological", "thematic"],
        min_confidence=0.85,
        patristic_witnesses=[
            PatristicWitness(
                father_name="Irenaeus of Lyons",
                authority=PatristicAuthority.GREAT_FATHER,
                quote="Christ recapitulated in himself the first Adam",
                source_work="Against Heresies, V.21.1",
                confidence=1.0
            ),
        ],
        notes="Recapitulation theology, foundational to Orthodox soteriology"
    ),

    # ESCHATOLOGICAL
    TheologicalTestCase(
        name="daniel_son_of_man",
        category=TheologicalCategory.ESCHATOLOGICAL,
        confidence=TheologicalConfidence.TRADITIONAL,
        description="Daniel's Son of Man vision connects to Christ's self-designation",
        source_verse="DAN.7.13",
        expected_connections=["MAT.24.30", "MAT.26.64", "REV.1.7", "REV.14.14"],
        expected_connection_types=["prophetic", "thematic"],
        min_confidence=0.80,
        patristic_witnesses=[
            PatristicWitness(
                father_name="Cyril of Alexandria",
                authority=PatristicAuthority.ECUMENICAL_FATHER,
                quote="The Son of Man coming on clouds is Christ's Second Coming",
                source_work="Commentary on Daniel",
                confidence=0.90
            ),
        ],
        notes="Eschatological prophecy central to Christ's identity"
    ),

    # TRINITARIAN
    TheologicalTestCase(
        name="plural_elohim_trinity",
        category=TheologicalCategory.TRINITARIAN,
        confidence=TheologicalConfidence.TRADITIONAL,
        description="Plural 'Let us make man' hints at Trinitarian nature of God",
        source_verse="GEN.1.26",
        expected_connections=["GEN.3.22", "GEN.11.7", "ISA.6.8"],
        expected_connection_types=["thematic", "conceptual"],
        min_confidence=0.65,
        patristic_witnesses=[
            PatristicWitness(
                father_name="Basil the Great",
                authority=PatristicAuthority.ECUMENICAL_FATHER,
                quote="'Let us make' reveals the plurality of Persons in the One Godhead",
                source_work="On the Holy Spirit",
                confidence=0.75
            ),
        ],
        notes="Debated interpretation, but patristically supported"
    ),

    # PROPHETIC
    TheologicalTestCase(
        name="bethlehem_birthplace",
        category=TheologicalCategory.PROPHETIC,
        confidence=TheologicalConfidence.DOGMATIC,
        description="Micah 5:2 prophecy of Messiah born in Bethlehem",
        source_verse="MIC.5.2",
        expected_connections=["MAT.2.6", "LUK.2.4", "JHN.7.42"],
        expected_connection_types=["prophetic"],
        min_confidence=0.95,
        patristic_witnesses=[
            PatristicWitness(
                father_name="Justin Martyr",
                authority=PatristicAuthority.MAJOR_FATHER,
                quote="Micah foretold Christ's birth in Bethlehem of Judea",
                source_work="Dialogue with Trypho, 78",
                confidence=1.0
            ),
        ],
        ecumenical_council_support="Implicit in Nicene Creed ('incarnate')",
        liturgical_support="Nativity Liturgy (December 25)",
        require_all_connections=True,
        notes="Precise geographical prophecy"
    ),

    TheologicalTestCase(
        name="suffering_servant",
        category=TheologicalCategory.PROPHETIC,
        confidence=TheologicalConfidence.CONSENSUS,
        description="Isaiah 53 Suffering Servant prophecy fulfilled in Christ's Passion",
        source_verse="ISA.53.5",
        expected_connections=["MAT.8.17", "MAT.27.12", "1PE.2.24", "ACT.8.32-35"],
        expected_connection_types=["prophetic", "typological"],
        min_confidence=0.90,
        patristic_witnesses=[
            PatristicWitness(
                father_name="Athanasius",
                authority=PatristicAuthority.ECUMENICAL_FATHER,
                quote="Isaiah saw the sufferings of the Savior and foretold them",
                source_work="On the Incarnation",
                confidence=1.0
            ),
            PatristicWitness(
                father_name="John Chrysostom",
                authority=PatristicAuthority.ECUMENICAL_FATHER,
                quote="The Suffering Servant is none other than Christ crucified",
                source_work="Homilies on Isaiah",
                confidence=1.0
            ),
        ],
        liturgical_support="Great Friday readings, Royal Hours",
        require_all_connections=True,
        notes="Most detailed messianic prophecy in OT"
    ),

    # MARIOLOGICAL
    TheologicalTestCase(
        name="ark_covenant_theotokos",
        category=TheologicalCategory.MARIOLOGICAL,
        confidence=TheologicalConfidence.TRADITIONAL,
        description="Ark of the Covenant as type of Mary the Theotokos",
        source_verse="EXO.25.10",
        expected_connections=["LUK.1.35", "REV.11.19", "2SA.6.14"],
        expected_connection_types=["typological"],
        min_confidence=0.75,
        patristic_witnesses=[
            PatristicWitness(
                father_name="Athanasius",
                authority=PatristicAuthority.ECUMENICAL_FATHER,
                quote="Mary is the new Ark containing the Word of God",
                source_work="Homily on the Annunciation",
                confidence=0.85
            ),
        ],
        liturgical_support="Akathist Hymn, Marian feasts",
        notes="Developed Marian typology in Eastern tradition"
    ),

    # LITURGICAL
    TheologicalTestCase(
        name="manna_eucharist",
        category=TheologicalCategory.LITURGICAL,
        confidence=TheologicalConfidence.CONSENSUS,
        description="Manna in wilderness as type of Eucharist",
        source_verse="EXO.16.15",
        expected_connections=["JHN.6.31-35", "JHN.6.48-51", "1CO.10.3"],
        expected_connection_types=["typological", "liturgical"],
        min_confidence=0.85,
        patristic_witnesses=[
            PatristicWitness(
                father_name="Cyril of Alexandria",
                authority=PatristicAuthority.ECUMENICAL_FATHER,
                quote="The manna prefigured the living Bread from heaven",
                source_work="Commentary on John",
                confidence=0.95
            ),
        ],
        liturgical_support="Divine Liturgy prayers, Holy Thursday",
        notes="Eucharistic typology, liturgically central"
    ),

    # EXPLORATORY (lower confidence, scholarly debate)
    TheologicalTestCase(
        name="melchizedek_priesthood",
        category=TheologicalCategory.TYPOLOGICAL,
        confidence=TheologicalConfidence.SCHOLARLY,
        description="Melchizedek as type of Christ's eternal priesthood",
        source_verse="GEN.14.18",
        expected_connections=["PSA.110.4", "HEB.7.1-3", "HEB.7.17"],
        expected_connection_types=["typological"],
        min_confidence=0.70,
        patristic_witnesses=[
            PatristicWitness(
                father_name="Jerome",
                authority=PatristicAuthority.GREAT_FATHER,
                quote="Melchizedek's priesthood foreshadows Christ's eternal priesthood",
                source_work="Letter to Evangelus",
                confidence=0.80
            ),
        ],
        notes="Hebrews makes this connection explicit"
    ),
]


def get_canonical_test_by_name(name: str) -> Optional[TheologicalTestCase]:
    """Retrieve canonical test case by name."""
    for test_case in CANONICAL_TEST_CASES:
        if test_case.name == name:
            return test_case
    return None


def get_canonical_tests_by_category(category: TheologicalCategory) -> List[TheologicalTestCase]:
    """Retrieve all canonical tests for a specific category."""
    return [tc for tc in CANONICAL_TEST_CASES if tc.category == category]


def get_canonical_tests_by_confidence(confidence: TheologicalConfidence) -> List[TheologicalTestCase]:
    """Retrieve all canonical tests at a specific confidence level."""
    return [tc for tc in CANONICAL_TEST_CASES if tc.confidence == confidence]


def calculate_theological_score(
    found_connections: Set[str],
    expected_connections: List[str],
    patristic_consensus: float,
    category_weight: float = 1.0
) -> float:
    """
    Calculate theological validation score.

    Args:
        found_connections: Set of discovered verse connections
        expected_connections: List of expected connections
        patristic_consensus: Patristic consensus score (0-1)
        category_weight: Weight for theological category

    Returns:
        Theological score (0-1)
    """
    # Connection accuracy
    if not expected_connections:
        connection_accuracy = 1.0
    else:
        matches = len(found_connections & set(expected_connections))
        connection_accuracy = matches / len(expected_connections)

    # Weighted combination
    theological_score = (
        connection_accuracy * 0.6 +  # 60% connection accuracy
        patristic_consensus * 0.4     # 40% patristic support
    ) * category_weight

    return min(1.0, theological_score)
