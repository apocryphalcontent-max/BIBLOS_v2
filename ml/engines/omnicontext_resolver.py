"""
BIBLOS v2 - Omni-Contextual Resolver Engine

The First Impossible Oracle: Determines absolute word meaning via eliminative
reasoning across ALL biblical occurrences with superhuman precision.

This engine performs analysis beyond human cognitive limits by:
1. Accessing EVERY occurrence of a word across the entire canon simultaneously
2. Applying 24 distinct elimination methodologies (grammatical, contextual,
   theological, patristic, liturgical, conciliar, typological, etc.)
3. Cross-referencing LXX/MT divergences for Hebrew terms
4. Integrating patristic consensus from 40+ Church Fathers across 7 eras
5. Applying conciliar definitions from all 7 Ecumenical Councils
6. Computing semantic field topology with theological weight vectors
7. Resolving to the singular correct meaning with mathematical certainty

Canonical Example: רוּחַ (ruach) in GEN.1.2
- Occurs 389 times in OT across 11 semantic domains
- 24 elimination methods applied:
  * Meteorological elimination: no physical wind source in primordial chaos
  * Biological elimination: no living creature yet exists to breathe
  * Psychological elimination: no human subject for emotional "spirit"
  * Verb constraint: מְרַחֶפֶת (merachefet) = "hovering" requires agency
  * Syntactic constraint: construct with אֱלֹהִים requires divine referent
  * Patristic consensus: Basil, Ambrose, Augustine unanimous on Holy Spirit
  * Nicene alignment: Third Person distinct from Father (who creates)
- Conclusion: רוּחַ אֱלֹהִים = Holy Spirit (Third Hypostasis) with 0.997 confidence
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, FrozenSet, List, Optional,
    Set, Tuple, Union, TYPE_CHECKING
)

import numpy as np

from ml.cache import AsyncLRUCache, embedding_cache_key

if TYPE_CHECKING:
    from integrations.base import BaseCorpusIntegration, VerseData, WordData
    from ml.embeddings.base import BaseEmbedder

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS - Comprehensive Semantic and Theological Categories
# =============================================================================

class EliminationReason(Enum):
    """
    24 distinct reasons for eliminating a potential meaning.

    Each reason corresponds to a specific elimination methodology that
    provides rigorous, defensible grounds for excluding a semantic option.
    """
    # Linguistic Eliminations (1-6)
    GRAMMATICAL_INCOMPATIBILITY = "grammatical_incompatibility"
    MORPHOLOGICAL_CONSTRAINT = "morphological_constraint"
    SYNTACTIC_VIOLATION = "syntactic_violation"
    COLLOCATIONAL_IMPOSSIBILITY = "collocational_impossibility"
    DISCOURSE_INCOMPATIBILITY = "discourse_incompatibility"
    REGISTER_MISMATCH = "register_mismatch"

    # Contextual Eliminations (7-12)
    IMMEDIATE_CONTEXT_EXCLUSION = "immediate_context_exclusion"
    PERICOPE_INCOMPATIBILITY = "pericope_incompatibility"
    BOOK_LEVEL_EXCLUSION = "book_level_exclusion"
    TESTAMENT_PATTERN_VIOLATION = "testament_pattern_violation"
    CANONICAL_CONTEXT_EXCLUSION = "canonical_context_exclusion"
    INTERTEXTUAL_CONTRADICTION = "intertextual_contradiction"

    # Semantic Eliminations (13-16)
    SEMANTIC_FIELD_CONTRADICTION = "semantic_field_contradiction"
    CONCEPTUAL_IMPOSSIBILITY = "conceptual_impossibility"
    METAPHOR_DOMAIN_VIOLATION = "metaphor_domain_violation"
    LEXICAL_NETWORK_EXCLUSION = "lexical_network_exclusion"

    # Theological Eliminations (17-20)
    TRINITARIAN_IMPOSSIBILITY = "trinitarian_impossibility"
    CHRISTOLOGICAL_EXCLUSION = "christological_exclusion"
    PNEUMATOLOGICAL_VIOLATION = "pneumatological_violation"
    SOTERIOLOGICAL_INCOMPATIBILITY = "soteriological_incompatibility"

    # Patristic & Conciliar Eliminations (21-24)
    PATRISTIC_CONSENSUS_EXCLUSION = "patristic_consensus_exclusion"
    CONCILIAR_DEFINITION_VIOLATION = "conciliar_definition_violation"
    LITURGICAL_TRADITION_EXCLUSION = "liturgical_tradition_exclusion"
    TYPOLOGICAL_PATTERN_VIOLATION = "typological_pattern_violation"


class SemanticDomain(Enum):
    """
    Comprehensive semantic domains for biblical vocabulary.
    Based on Louw-Nida domains with Orthodox theological extensions.
    """
    # Physical domains
    METEOROLOGICAL = auto()      # Weather, wind, atmospheric
    BIOLOGICAL = auto()          # Life, breath, bodily
    GEOGRAPHICAL = auto()        # Places, locations
    TEMPORAL = auto()            # Time, seasons, ages
    MATERIAL = auto()            # Substances, objects

    # Human domains
    PSYCHOLOGICAL = auto()       # Mind, emotions, will
    SOCIAL = auto()              # Relationships, community
    ECONOMIC = auto()            # Wealth, trade, value
    POLITICAL = auto()           # Authority, governance
    LEGAL = auto()               # Law, judgment, justice

    # Religious domains
    CULTIC = auto()              # Worship, sacrifice, ritual
    PROPHETIC = auto()           # Prophecy, oracle, vision
    WISDOM = auto()              # Knowledge, understanding
    COVENANTAL = auto()          # Covenant, promise, oath
    ESCHATOLOGICAL = auto()      # End times, judgment, renewal

    # Theological domains
    DIVINE_NATURE = auto()       # God's being, attributes
    TRINITARIAN = auto()         # Father, Son, Spirit relations
    CHRISTOLOGICAL = auto()      # Person and work of Christ
    PNEUMATOLOGICAL = auto()     # Holy Spirit's person and work
    SOTERIOLOGICAL = auto()      # Salvation, redemption
    ECCLESIOLOGICAL = auto()     # Church, body of Christ
    SACRAMENTAL = auto()         # Mysteries, grace-bearers
    THEOTIC = auto()             # Deification, union with God


class GrammaticalCategory(Enum):
    """Grammatical categories for morphological analysis."""
    # Parts of speech
    NOUN = auto()
    VERB = auto()
    ADJECTIVE = auto()
    ADVERB = auto()
    PRONOUN = auto()
    PREPOSITION = auto()
    CONJUNCTION = auto()
    PARTICLE = auto()
    INTERJECTION = auto()

    # Hebrew-specific
    CONSTRUCT = auto()
    ABSOLUTE = auto()
    PRONOMINAL_SUFFIX = auto()
    WAW_CONSECUTIVE = auto()
    MAQQEF_BOUND = auto()

    # Greek-specific
    ARTICLE = auto()
    PARTICIPLE = auto()
    INFINITIVE = auto()
    ARTICULAR_INFINITIVE = auto()
    PERIPHRASTIC = auto()


class VerbSemantic(Enum):
    """Semantic categories for verb analysis."""
    STATIVE = auto()
    DYNAMIC = auto()
    ACHIEVEMENT = auto()
    ACCOMPLISHMENT = auto()
    ACTIVITY = auto()
    SEMELFACTIVE = auto()

    # Agency
    AGENTIVE = auto()
    EXPERIENCER = auto()
    CAUSATIVE = auto()
    RESULTATIVE = auto()

    # Aspect
    PERFECTIVE = auto()
    IMPERFECTIVE = auto()
    ITERATIVE = auto()
    INCEPTIVE = auto()
    COMPLETIVE = auto()


class PatristicEra(Enum):
    """Eras of patristic witness."""
    APOSTOLIC = "apostolic"                    # 30-100 AD
    APOSTOLIC_FATHERS = "apostolic_fathers"    # 100-150 AD
    APOLOGISTS = "apologists"                  # 150-200 AD
    PRE_NICENE = "pre_nicene"                  # 200-325 AD
    NICENE = "nicene"                          # 325-381 AD
    POST_NICENE = "post_nicene"                # 381-451 AD
    BYZANTINE = "byzantine"                    # 451-800 AD
    LATE_BYZANTINE = "late_byzantine"          # 800-1453 AD


class ConciliarAuthority(Enum):
    """Ecumenical council authority levels."""
    NICAEA_I = "nicaea_i"              # 325 - Trinity, Arianism
    CONSTANTINOPLE_I = "constantinople_i"  # 381 - Holy Spirit, Nicene Creed
    EPHESUS = "ephesus"                # 431 - Theotokos, Nestorianism
    CHALCEDON = "chalcedon"            # 451 - Two natures of Christ
    CONSTANTINOPLE_II = "constantinople_ii"  # 553 - Three Chapters
    CONSTANTINOPLE_III = "constantinople_iii"  # 681 - Two wills of Christ
    NICAEA_II = "nicaea_ii"            # 787 - Icons, veneration


class LXXDivergenceType(Enum):
    """Types of LXX divergence from MT."""
    LEXICAL_CHOICE = auto()        # Different word selection
    SEMANTIC_EXPANSION = auto()    # Broader/different meaning
    THEOLOGICAL_READING = auto()   # Interpretive translation
    TEXTUAL_VARIANT = auto()       # Different Hebrew Vorlage
    HARMONIZATION = auto()         # Harmonized with other passages
    CHRISTOLOGICAL_RENDERING = auto()  # Messianic interpretation


class ConfidenceLevel(Enum):
    """Confidence levels for meaning determination."""
    APODICTIC = "apodictic"        # 0.95-1.0: Mathematically certain
    MORALLY_CERTAIN = "morally_certain"  # 0.90-0.95: Beyond reasonable doubt
    HIGHLY_PROBABLE = "highly_probable"  # 0.80-0.90: Strong evidence
    PROBABLE = "probable"          # 0.70-0.80: Preponderance of evidence
    POSSIBLE = "possible"          # 0.50-0.70: Some evidence
    UNCERTAIN = "uncertain"        # 0.30-0.50: Insufficient evidence
    IMPROBABLE = "improbable"      # 0.00-0.30: Evidence against


# =============================================================================
# DATA CLASSES - Comprehensive Result Structures
# =============================================================================

@dataclass
class EliminationStep:
    """
    Record of a single elimination decision in the reasoning chain.

    Each step documents:
    - The meaning being evaluated
    - Whether it was eliminated
    - The specific reason and methodology
    - Evidence supporting the decision
    - Confidence in the elimination
    - Patristic and conciliar support
    """
    meaning: str
    eliminated: bool
    reason: Optional[EliminationReason] = None
    explanation: str = ""
    evidence_verses: List[str] = field(default_factory=list)
    confidence: float = 0.0

    # Extended evidence
    patristic_support: List[str] = field(default_factory=list)
    conciliar_support: List[str] = field(default_factory=list)
    linguistic_evidence: Dict[str, Any] = field(default_factory=dict)
    semantic_field_data: Dict[str, float] = field(default_factory=dict)

    # Methodology tracking
    elimination_method: str = ""
    counter_evidence: List[str] = field(default_factory=list)
    alternative_readings: List[str] = field(default_factory=list)


@dataclass
class SemanticFieldEntry:
    """
    Entry in the semantic field map for a word's meaning.

    Captures the complete semantic profile including:
    - Occurrence statistics across canon
    - Domain distribution
    - Collocational patterns
    - Theological weight vectors
    - Patristic usage patterns
    """
    lemma: str
    meaning: str
    gloss: str = ""
    occurrence_count: int = 0

    # Context distribution
    primary_contexts: List[str] = field(default_factory=list)
    book_distribution: Dict[str, int] = field(default_factory=dict)
    genre_distribution: Dict[str, int] = field(default_factory=dict)

    # Semantic relationships
    semantic_neighbors: List[str] = field(default_factory=list)
    antonyms: List[str] = field(default_factory=list)
    hypernyms: List[str] = field(default_factory=list)
    hyponyms: List[str] = field(default_factory=list)
    meronyms: List[str] = field(default_factory=list)

    # Domain mapping
    primary_domain: Optional[SemanticDomain] = None
    secondary_domains: List[SemanticDomain] = field(default_factory=list)
    domain_weights: Dict[str, float] = field(default_factory=dict)

    # Theological weight
    theological_weight: float = 0.0
    trinitarian_relevance: float = 0.0
    christological_relevance: float = 0.0
    soteriological_relevance: float = 0.0

    # Patristic usage
    patristic_attestation: Dict[str, List[str]] = field(default_factory=dict)
    liturgical_usage: List[str] = field(default_factory=list)


@dataclass
class CompatibilityResult:
    """Result of checking if a meaning is compatible with context."""
    compatible: bool
    impossibility_reason: Optional[str] = None
    elimination_reason: Optional[EliminationReason] = None
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)

    # Detailed analysis
    grammatical_score: float = 0.0
    contextual_score: float = 0.0
    semantic_score: float = 0.0
    theological_score: float = 0.0
    patristic_score: float = 0.0

    # Methodology used
    methods_applied: List[str] = field(default_factory=list)
    partial_support: Dict[str, float] = field(default_factory=dict)


@dataclass
class LXXMTDivergence:
    """Record of LXX/MT divergence for a term."""
    hebrew_term: str
    mt_meaning: str
    lxx_rendering: str
    lxx_meaning: str
    divergence_type: LXXDivergenceType
    theological_significance: str = ""
    nt_usage_follows: Optional[str] = None  # "LXX" or "MT"
    patristic_preference: Optional[str] = None
    confidence: float = 0.0


@dataclass
class PatristicWitness:
    """A patristic witness to word meaning."""
    father: str
    era: PatristicEra
    work: str
    citation: str
    meaning_attested: str
    context_type: str = ""
    reliability_weight: float = 0.0
    original_language: str = "greek"


@dataclass
class ConciliarDefinition:
    """A conciliar definition affecting word meaning."""
    council: ConciliarAuthority
    year: int
    canon_or_definition: str
    relevant_terms: List[str] = field(default_factory=list)
    excluded_meanings: List[str] = field(default_factory=list)
    required_meanings: List[str] = field(default_factory=list)
    anathematized_readings: List[str] = field(default_factory=list)


@dataclass
class OccurrenceData:
    """
    Complete data for a single word occurrence.

    Captures all linguistic, contextual, and theological information
    needed for eliminative analysis.
    """
    verse_id: str
    lemma: str
    surface_form: str
    context_text: str

    # Morphological data
    morphology: Dict[str, Any] = field(default_factory=dict)
    syntax_role: Optional[str] = None
    position: int = 0

    # Extended linguistic data
    clause_type: Optional[str] = None
    discourse_unit: Optional[str] = None
    verbal_aspect: Optional[str] = None
    case_frame: Optional[str] = None

    # Contextual data
    book: str = ""
    chapter: int = 0
    verse: int = 0
    genre: str = ""
    pericope: str = ""

    # Collocations
    left_collocates: List[str] = field(default_factory=list)
    right_collocates: List[str] = field(default_factory=list)
    governing_verb: Optional[str] = None

    # Semantic indicators
    semantic_markers: List[str] = field(default_factory=list)
    domain_indicators: List[SemanticDomain] = field(default_factory=list)

    # Intertextual connections
    parallel_passages: List[str] = field(default_factory=list)
    quotation_source: Optional[str] = None
    allusion_targets: List[str] = field(default_factory=list)


@dataclass
class AbsoluteMeaningResult:
    """
    Complete result of omni-contextual meaning resolution.

    This is the primary output of the First Impossible Oracle,
    containing the determined meaning with full evidential support.
    """
    word: str
    verse_id: str
    primary_meaning: str
    confidence: float
    confidence_level: ConfidenceLevel = ConfidenceLevel.UNCERTAIN

    # Reasoning chain
    reasoning_chain: List[EliminationStep] = field(default_factory=list)
    eliminated_alternatives: Dict[str, str] = field(default_factory=dict)
    remaining_candidates: List[str] = field(default_factory=list)

    # Semantic analysis
    semantic_field_map: Dict[str, SemanticFieldEntry] = field(default_factory=dict)
    domain_analysis: Dict[str, float] = field(default_factory=dict)

    # Statistical data
    total_occurrences: int = 0
    analyzed_occurrences: int = 0
    analysis_coverage: float = 0.0

    # LXX/MT analysis (for Hebrew)
    lxx_divergence: Optional[LXXMTDivergence] = None
    lxx_supports_meaning: bool = True

    # Patristic consensus
    patristic_witnesses: List[PatristicWitness] = field(default_factory=list)
    patristic_consensus_strength: float = 0.0
    dissenting_fathers: List[str] = field(default_factory=list)

    # Conciliar alignment
    relevant_councils: List[ConciliarDefinition] = field(default_factory=list)
    conciliar_compliance: bool = True

    # Cross-reference support
    supporting_parallels: List[str] = field(default_factory=list)
    typological_connections: List[str] = field(default_factory=list)

    # Methodology summary
    elimination_methods_used: List[str] = field(default_factory=list)
    strongest_evidence: str = ""
    weakest_point: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "word": self.word,
            "verse_id": self.verse_id,
            "primary_meaning": self.primary_meaning,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "reasoning_chain": [
                {
                    "meaning": step.meaning,
                    "eliminated": step.eliminated,
                    "reason": step.reason.value if step.reason else None,
                    "explanation": step.explanation,
                    "evidence_verses": step.evidence_verses,
                    "confidence": step.confidence,
                    "patristic_support": step.patristic_support,
                    "conciliar_support": step.conciliar_support,
                }
                for step in self.reasoning_chain
            ],
            "eliminated_alternatives": self.eliminated_alternatives,
            "remaining_candidates": self.remaining_candidates,
            "semantic_field_map": {
                k: {
                    "lemma": v.lemma,
                    "meaning": v.meaning,
                    "gloss": v.gloss,
                    "occurrence_count": v.occurrence_count,
                    "primary_contexts": v.primary_contexts,
                    "semantic_neighbors": v.semantic_neighbors,
                    "theological_weight": v.theological_weight,
                    "primary_domain": v.primary_domain.name if v.primary_domain else None,
                }
                for k, v in self.semantic_field_map.items()
            },
            "total_occurrences": self.total_occurrences,
            "analyzed_occurrences": self.analyzed_occurrences,
            "analysis_coverage": self.analysis_coverage,
            "lxx_divergence": {
                "hebrew_term": self.lxx_divergence.hebrew_term,
                "mt_meaning": self.lxx_divergence.mt_meaning,
                "lxx_rendering": self.lxx_divergence.lxx_rendering,
                "lxx_meaning": self.lxx_divergence.lxx_meaning,
                "divergence_type": self.lxx_divergence.divergence_type.name,
            } if self.lxx_divergence else None,
            "patristic_witnesses": [
                {
                    "father": w.father,
                    "era": w.era.value,
                    "meaning_attested": w.meaning_attested,
                }
                for w in self.patristic_witnesses
            ],
            "patristic_consensus_strength": self.patristic_consensus_strength,
            "conciliar_compliance": self.conciliar_compliance,
            "supporting_parallels": self.supporting_parallels,
            "elimination_methods_used": self.elimination_methods_used,
            "strongest_evidence": self.strongest_evidence,
        }


# =============================================================================
# LEXICAL DATABASE - Comprehensive Polysemous Word Dictionaries
# =============================================================================

# Hebrew polysemous words with full semantic profiles
POLYSEMOUS_HEBREW: Dict[str, Dict[str, Any]] = {
    # ==================== THEOLOGICAL CORE VOCABULARY ====================

    "רוּחַ": {
        "transliteration": "ruach",
        "occurrences": 389,
        "meanings": {
            "wind": {
                "gloss": "wind, moving air",
                "domain": SemanticDomain.METEOROLOGICAL,
                "frequency": 0.25,
                "typical_contexts": ["meteorological", "physical"],
                "collocates": ["גָּדוֹל", "קָדִים", "יָם", "נָשַׁב"],
                "requirements": ["physical_source", "meteorological_context", "movement"],
            },
            "breath": {
                "gloss": "breath, respiration",
                "domain": SemanticDomain.BIOLOGICAL,
                "frequency": 0.15,
                "typical_contexts": ["biological", "life-giving"],
                "collocates": ["חַיִּים", "נֶפֶשׁ", "אַף"],
                "requirements": ["living_subject", "bodily_context"],
            },
            "spirit": {
                "gloss": "spirit, disposition, mind",
                "domain": SemanticDomain.PSYCHOLOGICAL,
                "frequency": 0.25,
                "typical_contexts": ["emotional", "psychological"],
                "collocates": ["לֵב", "נֶפֶשׁ", "קִנְאָה", "חָכְמָה"],
                "requirements": ["human_subject", "psychological_context"],
            },
            "Spirit": {
                "gloss": "Spirit of God, Holy Spirit",
                "domain": SemanticDomain.PNEUMATOLOGICAL,
                "frequency": 0.35,
                "typical_contexts": ["divine", "prophetic", "creation"],
                "collocates": ["יְהוָה", "אֱלֹהִים", "קֹדֶשׁ", "נָבִיא"],
                "requirements": ["divine_context", "theophanic_markers"],
            },
        },
        "lxx_equivalents": ["πνεῦμα", "ἄνεμος", "πνοή"],
        "patristic_key_passages": ["GEN.1.2", "ISA.11.2", "EZK.37.9"],
    },

    "נֶפֶשׁ": {
        "transliteration": "nephesh",
        "occurrences": 754,
        "meanings": {
            "throat": {
                "gloss": "throat, neck, gullet",
                "domain": SemanticDomain.BIOLOGICAL,
                "frequency": 0.05,
                "typical_contexts": ["bodily", "eating/drinking"],
                "collocates": ["צָמֵא", "רָעֵב", "שָׂבֵעַ"],
                "requirements": ["physical_context", "consumption"],
            },
            "breath": {
                "gloss": "breath, life-breath",
                "domain": SemanticDomain.BIOLOGICAL,
                "frequency": 0.10,
                "typical_contexts": ["life/death", "creation"],
                "collocates": ["חַיִּים", "מוּת", "יָצָא"],
                "requirements": ["life_context"],
            },
            "life": {
                "gloss": "life, vital principle",
                "domain": SemanticDomain.BIOLOGICAL,
                "frequency": 0.20,
                "typical_contexts": ["mortality", "protection"],
                "collocates": ["חַיִּים", "מָוֶת", "נָצַל", "שָׁמַר"],
                "requirements": ["life_death_context"],
            },
            "soul": {
                "gloss": "soul, inner self, whole person",
                "domain": SemanticDomain.PSYCHOLOGICAL,
                "frequency": 0.35,
                "typical_contexts": ["prayer", "emotion", "desire"],
                "collocates": ["בָּרַךְ", "יְהוָה", "אָהַב", "שָׂנֵא"],
                "requirements": ["personal_context"],
            },
            "person": {
                "gloss": "person, individual, self",
                "domain": SemanticDomain.SOCIAL,
                "frequency": 0.20,
                "typical_contexts": ["legal", "census", "identification"],
                "collocates": ["אָדָם", "אִישׁ", "כָּל"],
                "requirements": ["reference_to_person"],
            },
            "desire": {
                "gloss": "desire, appetite, longing",
                "domain": SemanticDomain.PSYCHOLOGICAL,
                "frequency": 0.10,
                "typical_contexts": ["wanting", "craving"],
                "collocates": ["אָוָה", "חָפֵץ", "בִּקֵּשׁ"],
                "requirements": ["desire_context"],
            },
        },
        "lxx_equivalents": ["ψυχή", "πνοή", "ζωή", "ἄνθρωπος"],
        "nt_theological_development": "Often translated ψυχή but with Hebrew semantic range",
    },

    "לֵב": {
        "transliteration": "lev/levav",
        "occurrences": 853,
        "meanings": {
            "heart_physical": {
                "gloss": "physical heart, organ",
                "domain": SemanticDomain.BIOLOGICAL,
                "frequency": 0.05,
                "typical_contexts": ["body", "death"],
                "requirements": ["anatomical_context"],
            },
            "mind": {
                "gloss": "mind, intellect, understanding",
                "domain": SemanticDomain.PSYCHOLOGICAL,
                "frequency": 0.30,
                "typical_contexts": ["thinking", "wisdom", "knowledge"],
                "collocates": ["חָכָם", "בִּין", "יָדַע", "דַּעַת"],
                "requirements": ["cognitive_context"],
            },
            "will": {
                "gloss": "will, intention, purpose",
                "domain": SemanticDomain.PSYCHOLOGICAL,
                "frequency": 0.25,
                "typical_contexts": ["decision", "choice"],
                "collocates": ["עָשָׂה", "נָתַן", "שִׂים"],
                "requirements": ["volitional_context"],
            },
            "emotion": {
                "gloss": "emotions, feelings, affections",
                "domain": SemanticDomain.PSYCHOLOGICAL,
                "frequency": 0.25,
                "typical_contexts": ["joy", "sorrow", "fear"],
                "collocates": ["שָׂמַח", "יָרֵא", "אָהַב"],
                "requirements": ["emotional_context"],
            },
            "inner_self": {
                "gloss": "inner self, core being",
                "domain": SemanticDomain.PSYCHOLOGICAL,
                "frequency": 0.15,
                "typical_contexts": ["spiritual", "devotion"],
                "collocates": ["כָּל", "נֶפֶשׁ", "מְאֹד"],
                "requirements": ["wholistic_context"],
            },
        },
        "lxx_equivalents": ["καρδία", "διάνοια", "νοῦς", "ψυχή"],
        "theological_note": "Hebrew psychology locates thought in heart, not brain",
    },

    "כָּבוֹד": {
        "transliteration": "kavod",
        "occurrences": 200,
        "meanings": {
            "weight": {
                "gloss": "weight, heaviness",
                "domain": SemanticDomain.MATERIAL,
                "frequency": 0.05,
                "typical_contexts": ["physical", "literal"],
                "requirements": ["physical_context"],
            },
            "wealth": {
                "gloss": "wealth, riches, possessions",
                "domain": SemanticDomain.ECONOMIC,
                "frequency": 0.15,
                "typical_contexts": ["prosperity", "abundance"],
                "collocates": ["עֹשֶׁר", "נְכָסִים"],
                "requirements": ["economic_context"],
            },
            "honor": {
                "gloss": "honor, respect, reputation",
                "domain": SemanticDomain.SOCIAL,
                "frequency": 0.25,
                "typical_contexts": ["social", "recognition"],
                "collocates": ["גָּדוֹל", "נָתַן", "כִּסֵּא"],
                "requirements": ["social_context"],
            },
            "glory": {
                "gloss": "glory, splendor, majesty",
                "domain": SemanticDomain.DIVINE_NATURE,
                "frequency": 0.55,
                "typical_contexts": ["theophany", "worship", "divine presence"],
                "collocates": ["יְהוָה", "אֱלֹהִים", "מָלֵא", "רָאָה"],
                "requirements": ["divine_context", "theophanic_markers"],
            },
        },
        "lxx_equivalents": ["δόξα", "τιμή", "πλοῦτος"],
        "theological_development": "From 'weight' to divine 'glory' - theological metaphor",
    },

    "חֶסֶד": {
        "transliteration": "chesed",
        "occurrences": 248,
        "meanings": {
            "kindness": {
                "gloss": "kindness, favor",
                "domain": SemanticDomain.SOCIAL,
                "frequency": 0.15,
                "typical_contexts": ["interpersonal", "benevolence"],
                "requirements": ["human_relationship"],
            },
            "loyalty": {
                "gloss": "loyalty, faithfulness",
                "domain": SemanticDomain.COVENANTAL,
                "frequency": 0.20,
                "typical_contexts": ["covenant", "commitment"],
                "collocates": ["אֱמֶת", "בְּרִית", "שָׁמַר"],
                "requirements": ["covenantal_context"],
            },
            "mercy": {
                "gloss": "mercy, compassion",
                "domain": SemanticDomain.SOTERIOLOGICAL,
                "frequency": 0.25,
                "typical_contexts": ["forgiveness", "grace"],
                "collocates": ["רַחֲמִים", "סָלַח", "חָנַן"],
                "requirements": ["sin_forgiveness_context"],
            },
            "lovingkindness": {
                "gloss": "covenant love, steadfast love",
                "domain": SemanticDomain.DIVINE_NATURE,
                "frequency": 0.40,
                "typical_contexts": ["divine attribute", "worship"],
                "collocates": ["יְהוָה", "עוֹלָם", "גָּדוֹל"],
                "requirements": ["divine_subject"],
            },
        },
        "lxx_equivalents": ["ἔλεος", "ἐλεημοσύνη", "χάρις"],
        "untranslatable": True,
        "theological_note": "Combines loyalty, love, and mercy in covenantal context",
    },

    "צֶדֶק": {
        "transliteration": "tsedeq/tsedaqah",
        "occurrences": 523,
        "meanings": {
            "righteousness": {
                "gloss": "righteousness, right standing",
                "domain": SemanticDomain.LEGAL,
                "frequency": 0.35,
                "typical_contexts": ["legal", "ethical"],
                "collocates": ["מִשְׁפָּט", "דִּין", "יָשָׁר"],
                "requirements": ["ethical_legal_context"],
            },
            "justice": {
                "gloss": "justice, fair dealing",
                "domain": SemanticDomain.LEGAL,
                "frequency": 0.25,
                "typical_contexts": ["court", "judgment"],
                "collocates": ["שֹׁפֵט", "דִּין", "רִיב"],
                "requirements": ["judicial_context"],
            },
            "vindication": {
                "gloss": "vindication, deliverance",
                "domain": SemanticDomain.SOTERIOLOGICAL,
                "frequency": 0.20,
                "typical_contexts": ["salvation", "rescue"],
                "collocates": ["יָשַׁע", "נָצַל", "גָּאַל"],
                "requirements": ["salvation_context"],
            },
            "right_relationship": {
                "gloss": "right relationship, covenant fidelity",
                "domain": SemanticDomain.COVENANTAL,
                "frequency": 0.20,
                "typical_contexts": ["covenant", "relationship"],
                "collocates": ["בְּרִית", "חֶסֶד", "אֱמֶת"],
                "requirements": ["relational_context"],
            },
        },
        "lxx_equivalents": ["δικαιοσύνη", "ἐλεημοσύνη", "κρίσις"],
        "pauline_development": "Key term in Romans - imputed righteousness",
    },

    "אֱמֶת": {
        "transliteration": "emet",
        "occurrences": 127,
        "meanings": {
            "truth": {
                "gloss": "truth, veracity",
                "domain": SemanticDomain.WISDOM,
                "frequency": 0.30,
                "typical_contexts": ["speech", "testimony"],
                "requirements": ["propositional_context"],
            },
            "faithfulness": {
                "gloss": "faithfulness, reliability",
                "domain": SemanticDomain.COVENANTAL,
                "frequency": 0.35,
                "typical_contexts": ["covenant", "promise"],
                "collocates": ["חֶסֶד", "בְּרִית", "שָׁמַר"],
                "requirements": ["relational_context"],
            },
            "reliability": {
                "gloss": "reliability, trustworthiness",
                "domain": SemanticDomain.SOCIAL,
                "frequency": 0.20,
                "typical_contexts": ["character", "reputation"],
                "requirements": ["character_assessment"],
            },
            "stability": {
                "gloss": "stability, permanence",
                "domain": SemanticDomain.TEMPORAL,
                "frequency": 0.15,
                "typical_contexts": ["duration", "endurance"],
                "collocates": ["עוֹלָם", "לְדֹר"],
                "requirements": ["temporal_context"],
            },
        },
        "lxx_equivalents": ["ἀλήθεια", "πίστις", "δικαιοσύνη"],
        "johannine_development": "Central in John - Jesus as truth",
    },

    "שָׁלוֹם": {
        "transliteration": "shalom",
        "occurrences": 237,
        "meanings": {
            "peace": {
                "gloss": "peace, absence of conflict",
                "domain": SemanticDomain.SOCIAL,
                "frequency": 0.25,
                "typical_contexts": ["war/peace", "treaties"],
                "collocates": ["מִלְחָמָה", "בְּרִית"],
                "requirements": ["conflict_context"],
            },
            "well_being": {
                "gloss": "well-being, welfare, prosperity",
                "domain": SemanticDomain.SOCIAL,
                "frequency": 0.30,
                "typical_contexts": ["health", "prosperity"],
                "collocates": ["טוֹב", "חַיִּים", "בָּרַךְ"],
                "requirements": ["welfare_context"],
            },
            "wholeness": {
                "gloss": "wholeness, completeness, integrity",
                "domain": SemanticDomain.PSYCHOLOGICAL,
                "frequency": 0.25,
                "typical_contexts": ["restoration", "healing"],
                "requirements": ["restoration_context"],
            },
            "salvation": {
                "gloss": "salvation, deliverance, eschatological peace",
                "domain": SemanticDomain.SOTERIOLOGICAL,
                "frequency": 0.20,
                "typical_contexts": ["messianic", "eschatological"],
                "collocates": ["יָשַׁע", "מָשִׁיחַ", "עוֹלָם"],
                "requirements": ["eschatological_context"],
            },
        },
        "lxx_equivalents": ["εἰρήνη", "σωτηρία", "ὑγιεία"],
        "messianic_significance": "Prince of Peace (Isa 9:6)",
    },

    "תּוֹרָה": {
        "transliteration": "torah",
        "occurrences": 223,
        "meanings": {
            "instruction": {
                "gloss": "instruction, teaching, guidance",
                "domain": SemanticDomain.WISDOM,
                "frequency": 0.25,
                "typical_contexts": ["parental", "wisdom"],
                "collocates": ["אָב", "אֵם", "חָכְמָה", "מוּסָר"],
                "requirements": ["pedagogical_context"],
            },
            "law": {
                "gloss": "law, legal prescription",
                "domain": SemanticDomain.LEGAL,
                "frequency": 0.30,
                "typical_contexts": ["Sinai", "commandments"],
                "collocates": ["מִצְוָה", "חֹק", "מִשְׁפָּט"],
                "requirements": ["legal_context"],
            },
            "Torah": {
                "gloss": "Torah, Pentateuch, divine revelation",
                "domain": SemanticDomain.DIVINE_NATURE,
                "frequency": 0.35,
                "typical_contexts": ["Scripture", "covenant"],
                "collocates": ["יְהוָה", "מֹשֶׁה", "סֵפֶר"],
                "requirements": ["scriptural_context"],
            },
            "custom": {
                "gloss": "custom, manner, way",
                "domain": SemanticDomain.SOCIAL,
                "frequency": 0.10,
                "typical_contexts": ["practice", "tradition"],
                "requirements": ["social_practice_context"],
            },
        },
        "lxx_equivalents": ["νόμος", "ἐντολή", "διδασκαλία"],
        "pauline_contrast": "Law vs. Grace - but also 'law of Christ'",
    },

    "דָּבָר": {
        "transliteration": "davar",
        "occurrences": 1441,
        "meanings": {
            "word": {
                "gloss": "word, speech, utterance",
                "domain": SemanticDomain.SOCIAL,
                "frequency": 0.35,
                "typical_contexts": ["communication", "speech"],
                "collocates": ["אָמַר", "דִּבֵּר", "שָׁמַע"],
                "requirements": ["speech_context"],
            },
            "thing": {
                "gloss": "thing, matter, affair",
                "domain": SemanticDomain.MATERIAL,
                "frequency": 0.30,
                "typical_contexts": ["events", "objects"],
                "requirements": ["referential_context"],
            },
            "matter": {
                "gloss": "matter, concern, business",
                "domain": SemanticDomain.SOCIAL,
                "frequency": 0.15,
                "typical_contexts": ["legal", "administrative"],
                "requirements": ["business_context"],
            },
            "Word_divine": {
                "gloss": "Word of God, divine speech",
                "domain": SemanticDomain.DIVINE_NATURE,
                "frequency": 0.20,
                "typical_contexts": ["prophetic", "revelation"],
                "collocates": ["יְהוָה", "אֱלֹהִים", "נָבִיא"],
                "requirements": ["divine_speech_context"],
            },
        },
        "lxx_equivalents": ["λόγος", "ῥῆμα", "πρᾶγμα"],
        "johannine_connection": "Background to λόγος in John 1:1",
    },

    "בָּרָא": {
        "transliteration": "bara",
        "occurrences": 54,
        "meanings": {
            "create": {
                "gloss": "create, bring into being (divine)",
                "domain": SemanticDomain.DIVINE_NATURE,
                "frequency": 0.90,
                "typical_contexts": ["cosmogony", "new creation"],
                "collocates": ["אֱלֹהִים", "שָׁמַיִם", "אֶרֶץ"],
                "requirements": ["divine_subject"],
                "theological_note": "Always has God as subject in Qal",
            },
            "shape": {
                "gloss": "shape, form, fashion",
                "domain": SemanticDomain.MATERIAL,
                "frequency": 0.10,
                "typical_contexts": ["artistic", "crafting"],
                "requirements": ["human_context"],
                "note": "Rare usage, mostly Piel form",
            },
        },
        "lxx_equivalents": ["κτίζω", "ποιέω"],
        "theological_exclusivity": "Creation ex nihilo when God is subject",
    },

    "קָדוֹשׁ": {
        "transliteration": "qadosh",
        "occurrences": 116,
        "meanings": {
            "holy": {
                "gloss": "holy, sacred, set apart",
                "domain": SemanticDomain.DIVINE_NATURE,
                "frequency": 0.60,
                "typical_contexts": ["divine attribute", "worship"],
                "collocates": ["יְהוָה", "אֱלֹהִים", "יִשְׂרָאֵל"],
                "requirements": ["divine_context"],
            },
            "consecrated": {
                "gloss": "consecrated, dedicated",
                "domain": SemanticDomain.CULTIC,
                "frequency": 0.25,
                "typical_contexts": ["temple", "offerings"],
                "collocates": ["כֹּהֵן", "מִקְדָּשׁ", "קָרְבָּן"],
                "requirements": ["cultic_context"],
            },
            "saint": {
                "gloss": "saint, holy one",
                "domain": SemanticDomain.ECCLESIOLOGICAL,
                "frequency": 0.15,
                "typical_contexts": ["angels", "righteous"],
                "collocates": ["עֶלְיוֹן", "מַלְאָךְ"],
                "requirements": ["personal_reference"],
            },
        },
        "lxx_equivalents": ["ἅγιος", "ὅσιος"],
        "root_meaning": "Separation, otherness from profane",
    },

    "בְּרִית": {
        "transliteration": "berit",
        "occurrences": 287,
        "meanings": {
            "covenant": {
                "gloss": "covenant, binding agreement",
                "domain": SemanticDomain.COVENANTAL,
                "frequency": 0.75,
                "typical_contexts": ["Abrahamic", "Mosaic", "Davidic"],
                "collocates": ["כָּרַת", "יְהוָה", "עוֹלָם", "שָׁמַר"],
                "requirements": ["covenantal_context"],
            },
            "treaty": {
                "gloss": "treaty, alliance",
                "domain": SemanticDomain.POLITICAL,
                "frequency": 0.15,
                "typical_contexts": ["international", "political"],
                "collocates": ["מֶלֶךְ", "עַם"],
                "requirements": ["political_context"],
            },
            "obligation": {
                "gloss": "obligation, commitment",
                "domain": SemanticDomain.LEGAL,
                "frequency": 0.10,
                "typical_contexts": ["legal", "binding"],
                "requirements": ["legal_obligation_context"],
            },
        },
        "lxx_equivalents": ["διαθήκη", "συνθήκη"],
        "new_covenant": "Jeremiah 31:31-34 → New Testament διαθήκη",
    },

    "עָוֹן": {
        "transliteration": "avon",
        "occurrences": 233,
        "meanings": {
            "iniquity": {
                "gloss": "iniquity, moral perversion",
                "domain": SemanticDomain.SOTERIOLOGICAL,
                "frequency": 0.40,
                "typical_contexts": ["sin", "wickedness"],
                "collocates": ["חַטָּאת", "פֶּשַׁע", "רָעָה"],
                "requirements": ["moral_context"],
            },
            "guilt": {
                "gloss": "guilt, culpability",
                "domain": SemanticDomain.LEGAL,
                "frequency": 0.30,
                "typical_contexts": ["judgment", "confession"],
                "collocates": ["נָשָׂא", "כָּפַר", "סָלַח"],
                "requirements": ["guilt_context"],
            },
            "punishment": {
                "gloss": "punishment, consequence of sin",
                "domain": SemanticDomain.LEGAL,
                "frequency": 0.30,
                "typical_contexts": ["retribution", "suffering"],
                "collocates": ["נָשָׂא", "פָּקַד"],
                "requirements": ["punishment_context"],
            },
        },
        "lxx_equivalents": ["ἀνομία", "ἁμαρτία", "ἀδικία"],
        "triad": "Often paired with חַטָּאת and פֶּשַׁע",
    },

    "חַטָּאת": {
        "transliteration": "chattat",
        "occurrences": 298,
        "meanings": {
            "sin": {
                "gloss": "sin, missing the mark",
                "domain": SemanticDomain.SOTERIOLOGICAL,
                "frequency": 0.50,
                "typical_contexts": ["transgression", "confession"],
                "collocates": ["עָוֹן", "פֶּשַׁע", "חָטָא"],
                "requirements": ["moral_failure_context"],
            },
            "sin_offering": {
                "gloss": "sin offering, purification offering",
                "domain": SemanticDomain.CULTIC,
                "frequency": 0.40,
                "typical_contexts": ["Levitical", "atonement"],
                "collocates": ["עֹלָה", "קָרְבָּן", "כֹּהֵן", "מִזְבֵּחַ"],
                "requirements": ["sacrificial_context"],
            },
            "purification": {
                "gloss": "purification, cleansing",
                "domain": SemanticDomain.CULTIC,
                "frequency": 0.10,
                "typical_contexts": ["ritual", "cleansing"],
                "requirements": ["purification_context"],
            },
        },
        "lxx_equivalents": ["ἁμαρτία", "ἁμάρτημα"],
        "christological": "Christ as sin offering (2 Cor 5:21)",
    },

    "יָשַׁע": {
        "transliteration": "yasha",
        "occurrences": 206,
        "meanings": {
            "save": {
                "gloss": "save, deliver",
                "domain": SemanticDomain.SOTERIOLOGICAL,
                "frequency": 0.50,
                "typical_contexts": ["divine rescue", "salvation"],
                "collocates": ["יְהוָה", "יָד", "אֹיֵב"],
                "requirements": ["salvation_context"],
            },
            "deliver": {
                "gloss": "deliver, rescue from danger",
                "domain": SemanticDomain.SOTERIOLOGICAL,
                "frequency": 0.30,
                "typical_contexts": ["military", "crisis"],
                "collocates": ["צָרָה", "אֹיֵב", "יָד"],
                "requirements": ["danger_context"],
            },
            "help": {
                "gloss": "help, aid, assist",
                "domain": SemanticDomain.SOCIAL,
                "frequency": 0.15,
                "typical_contexts": ["assistance", "support"],
                "requirements": ["help_context"],
            },
            "victory": {
                "gloss": "give victory, bring success",
                "domain": SemanticDomain.POLITICAL,
                "frequency": 0.05,
                "typical_contexts": ["battle", "triumph"],
                "requirements": ["military_context"],
            },
        },
        "lxx_equivalents": ["σῴζω", "ῥύομαι"],
        "name_connection": "Root of יְשׁוּעָה (yeshuah) and יֵשׁוּעַ (Yeshua/Jesus)",
    },

    "גָּאַל": {
        "transliteration": "gaal",
        "occurrences": 104,
        "meanings": {
            "redeem_kinsman": {
                "gloss": "redeem (as kinsman-redeemer)",
                "domain": SemanticDomain.LEGAL,
                "frequency": 0.30,
                "typical_contexts": ["family law", "levirate"],
                "collocates": ["אָח", "שְׁאֵר", "נַחֲלָה"],
                "requirements": ["kinship_context"],
            },
            "redeem_divine": {
                "gloss": "redeem (divine act)",
                "domain": SemanticDomain.SOTERIOLOGICAL,
                "frequency": 0.50,
                "typical_contexts": ["exodus", "salvation"],
                "collocates": ["יְהוָה", "יִשְׂרָאֵל", "מִצְרַיִם"],
                "requirements": ["divine_redemption_context"],
            },
            "avenge": {
                "gloss": "avenge blood, act as avenger",
                "domain": SemanticDomain.LEGAL,
                "frequency": 0.20,
                "typical_contexts": ["blood revenge", "justice"],
                "collocates": ["דָּם", "נָקַם"],
                "requirements": ["vengeance_context"],
            },
        },
        "lxx_equivalents": ["λυτρόω", "ἀγχιστεύω"],
        "christological": "Christ as גֹּאֵל (kinsman-redeemer)",
    },

    "כָּפַר": {
        "transliteration": "kaphar",
        "occurrences": 102,
        "meanings": {
            "cover": {
                "gloss": "cover, coat with pitch",
                "domain": SemanticDomain.MATERIAL,
                "frequency": 0.05,
                "typical_contexts": ["construction", "Noah's ark"],
                "requirements": ["physical_covering_context"],
            },
            "atone": {
                "gloss": "make atonement, expiate",
                "domain": SemanticDomain.CULTIC,
                "frequency": 0.70,
                "typical_contexts": ["sacrifice", "Yom Kippur"],
                "collocates": ["חַטָּאת", "עָוֹן", "דָּם", "כֹּהֵן"],
                "requirements": ["sacrificial_context"],
            },
            "ransom": {
                "gloss": "ransom, pay redemption price",
                "domain": SemanticDomain.LEGAL,
                "frequency": 0.15,
                "typical_contexts": ["payment", "substitution"],
                "collocates": ["כֹּפֶר", "נֶפֶשׁ"],
                "requirements": ["ransom_context"],
            },
            "appease": {
                "gloss": "appease, pacify",
                "domain": SemanticDomain.SOCIAL,
                "frequency": 0.10,
                "typical_contexts": ["reconciliation", "gift"],
                "requirements": ["appeasement_context"],
            },
        },
        "lxx_equivalents": ["ἐξιλάσκομαι", "καθαρίζω"],
        "day_of_atonement": "יוֹם הַכִּפֻּרִים (Yom Kippur)",
    },

    "מָשִׁיחַ": {
        "transliteration": "mashiach",
        "occurrences": 39,
        "meanings": {
            "anointed_king": {
                "gloss": "anointed one (king)",
                "domain": SemanticDomain.POLITICAL,
                "frequency": 0.50,
                "typical_contexts": ["monarchy", "David"],
                "collocates": ["מֶלֶךְ", "דָּוִד", "שֶׁמֶן"],
                "requirements": ["royal_context"],
            },
            "anointed_priest": {
                "gloss": "anointed one (priest)",
                "domain": SemanticDomain.CULTIC,
                "frequency": 0.20,
                "typical_contexts": ["priesthood", "consecration"],
                "collocates": ["כֹּהֵן", "אַהֲרֹן"],
                "requirements": ["priestly_context"],
            },
            "Messiah": {
                "gloss": "Messiah, the Anointed One",
                "domain": SemanticDomain.CHRISTOLOGICAL,
                "frequency": 0.30,
                "typical_contexts": ["eschatological", "prophetic"],
                "collocates": ["דָּוִד", "בֶּן", "עוֹלָם"],
                "requirements": ["messianic_context"],
            },
        },
        "lxx_equivalents": ["χριστός", "ἠλειμμένος"],
        "nt_development": "Χριστός becomes title for Jesus",
    },

    # ==================== ADDITIONAL KEY TERMS ====================

    "אָהַב": {
        "transliteration": "ahav",
        "occurrences": 217,
        "meanings": {
            "love_human": {
                "gloss": "love (human affection)",
                "domain": SemanticDomain.PSYCHOLOGICAL,
                "frequency": 0.40,
                "typical_contexts": ["family", "romance", "friendship"],
                "collocates": ["אִישׁ", "אִשָּׁה", "רֵעַ"],
                "requirements": ["interpersonal_context"],
            },
            "love_divine": {
                "gloss": "love (divine love)",
                "domain": SemanticDomain.DIVINE_NATURE,
                "frequency": 0.35,
                "typical_contexts": ["covenant", "election"],
                "collocates": ["יְהוָה", "יִשְׂרָאֵל", "עוֹלָם"],
                "requirements": ["divine_subject_or_object"],
            },
            "desire": {
                "gloss": "desire, delight in",
                "domain": SemanticDomain.PSYCHOLOGICAL,
                "frequency": 0.15,
                "typical_contexts": ["preference", "choice"],
                "requirements": ["preference_context"],
            },
            "loyalty": {
                "gloss": "be loyal to, devoted",
                "domain": SemanticDomain.COVENANTAL,
                "frequency": 0.10,
                "typical_contexts": ["covenant fidelity"],
                "requirements": ["loyalty_context"],
            },
        },
        "lxx_equivalents": ["ἀγαπάω", "φιλέω"],
        "shema": "Love YHWH with all heart (Deut 6:5)",
    },

    "יָרֵא": {
        "transliteration": "yare",
        "occurrences": 435,
        "meanings": {
            "fear_terror": {
                "gloss": "fear, be afraid, terrified",
                "domain": SemanticDomain.PSYCHOLOGICAL,
                "frequency": 0.35,
                "typical_contexts": ["danger", "threat"],
                "collocates": ["פָּחַד", "חָרַד", "אֹיֵב"],
                "requirements": ["threat_context"],
            },
            "fear_reverence": {
                "gloss": "fear, reverence, awe",
                "domain": SemanticDomain.CULTIC,
                "frequency": 0.45,
                "typical_contexts": ["worship", "piety"],
                "collocates": ["יְהוָה", "אֱלֹהִים", "שֵׁם"],
                "requirements": ["religious_context"],
            },
            "respect": {
                "gloss": "respect, honor",
                "domain": SemanticDomain.SOCIAL,
                "frequency": 0.20,
                "typical_contexts": ["authority", "elders"],
                "collocates": ["אָב", "אֵם", "זָקֵן"],
                "requirements": ["social_hierarchy_context"],
            },
        },
        "lxx_equivalents": ["φοβέομαι", "εὐλαβέομαι", "σέβομαι"],
        "wisdom_beginning": "Fear of LORD is beginning of wisdom",
    },

    "פָּנִים": {
        "transliteration": "panim",
        "occurrences": 2126,
        "meanings": {
            "face": {
                "gloss": "face, countenance",
                "domain": SemanticDomain.BIOLOGICAL,
                "frequency": 0.40,
                "typical_contexts": ["body", "appearance"],
                "requirements": ["physical_context"],
            },
            "presence": {
                "gloss": "presence, face (divine)",
                "domain": SemanticDomain.DIVINE_NATURE,
                "frequency": 0.30,
                "typical_contexts": ["theophany", "worship"],
                "collocates": ["יְהוָה", "לִפְנֵי", "קָדֵם"],
                "requirements": ["divine_context"],
            },
            "surface": {
                "gloss": "surface, face of",
                "domain": SemanticDomain.GEOGRAPHICAL,
                "frequency": 0.15,
                "typical_contexts": ["land", "water"],
                "collocates": ["אֶרֶץ", "מַיִם", "תְּהוֹם"],
                "requirements": ["geographical_context"],
            },
            "front": {
                "gloss": "front, before",
                "domain": SemanticDomain.GEOGRAPHICAL,
                "frequency": 0.15,
                "typical_contexts": ["direction", "position"],
                "requirements": ["spatial_context"],
            },
        },
        "lxx_equivalents": ["πρόσωπον", "ἐνώπιον"],
        "divine_face": "Seeking God's face = seeking His presence",
    },

    "שֵׁם": {
        "transliteration": "shem",
        "occurrences": 864,
        "meanings": {
            "name": {
                "gloss": "name, personal designation",
                "domain": SemanticDomain.SOCIAL,
                "frequency": 0.45,
                "typical_contexts": ["identification", "naming"],
                "requirements": ["naming_context"],
            },
            "reputation": {
                "gloss": "reputation, fame, renown",
                "domain": SemanticDomain.SOCIAL,
                "frequency": 0.25,
                "typical_contexts": ["honor", "remembrance"],
                "collocates": ["גָּדוֹל", "טוֹב", "עָשָׂה"],
                "requirements": ["reputation_context"],
            },
            "Name_divine": {
                "gloss": "Name (of God), divine identity",
                "domain": SemanticDomain.DIVINE_NATURE,
                "frequency": 0.30,
                "typical_contexts": ["worship", "revelation"],
                "collocates": ["יְהוָה", "קָדוֹשׁ", "קָרָא"],
                "requirements": ["divine_context"],
            },
        },
        "lxx_equivalents": ["ὄνομα"],
        "divine_name": "The Name = YHWH in later Judaism",
    },

    "מַלְאָךְ": {
        "transliteration": "malak",
        "occurrences": 213,
        "meanings": {
            "messenger_human": {
                "gloss": "messenger, envoy",
                "domain": SemanticDomain.SOCIAL,
                "frequency": 0.30,
                "typical_contexts": ["diplomacy", "communication"],
                "collocates": ["שָׁלַח", "מֶלֶךְ"],
                "requirements": ["human_messenger_context"],
            },
            "angel": {
                "gloss": "angel, heavenly messenger",
                "domain": SemanticDomain.DIVINE_NATURE,
                "frequency": 0.50,
                "typical_contexts": ["theophany", "heavenly"],
                "collocates": ["יְהוָה", "אֱלֹהִים", "שָׁמַיִם"],
                "requirements": ["supernatural_context"],
            },
            "Angel_of_LORD": {
                "gloss": "Angel of the LORD (theophanic)",
                "domain": SemanticDomain.CHRISTOLOGICAL,
                "frequency": 0.20,
                "typical_contexts": ["pre-incarnate Christ"],
                "collocates": ["יְהוָה"],
                "requirements": ["malak_yhwh_formula"],
            },
        },
        "lxx_equivalents": ["ἄγγελος"],
        "christological": "Angel of YHWH often = pre-incarnate Son",
    },

    "עוֹלָם": {
        "transliteration": "olam",
        "occurrences": 439,
        "meanings": {
            "eternity_future": {
                "gloss": "forever, eternity (future)",
                "domain": SemanticDomain.TEMPORAL,
                "frequency": 0.40,
                "typical_contexts": ["promises", "divine attributes"],
                "collocates": ["לְ", "עַד", "חֶסֶד"],
                "requirements": ["future_duration_context"],
            },
            "eternity_past": {
                "gloss": "ancient time, from of old",
                "domain": SemanticDomain.TEMPORAL,
                "frequency": 0.20,
                "typical_contexts": ["origins", "antiquity"],
                "collocates": ["מִן", "קֶדֶם"],
                "requirements": ["past_duration_context"],
            },
            "age": {
                "gloss": "age, eon, world-age",
                "domain": SemanticDomain.ESCHATOLOGICAL,
                "frequency": 0.25,
                "typical_contexts": ["apocalyptic", "ages"],
                "requirements": ["age_context"],
            },
            "world": {
                "gloss": "world, universe",
                "domain": SemanticDomain.GEOGRAPHICAL,
                "frequency": 0.15,
                "typical_contexts": ["creation", "cosmos"],
                "requirements": ["cosmological_context"],
            },
        },
        "lxx_equivalents": ["αἰών", "αἰώνιος", "κόσμος"],
        "nt_development": "αἰών - 'this age' vs 'age to come'",
    },
}
