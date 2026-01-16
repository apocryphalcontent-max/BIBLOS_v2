"""
BIBLOS v2 - LXX Christological Extractor

The Third Impossible Oracle: Discovers Christological content uniquely present
in the Septuagint but absent from or muted in the Masoretic Text.

This engine identifies divergences between LXX and MT texts, classifies their
Christological significance, and gathers supporting evidence from manuscripts,
NT quotations, and patristic witnesses.

Critical Principle: Oldest transcriptions are most valid. When manuscript
variants exist, prioritize the earliest attested readings.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class ManuscriptPriority(Enum):
    """
    Textual authority ranking - oldest = most authoritative.
    Century values are negative for BCE, positive for CE.
    """
    DSS = ("Dead Sea Scrolls", -3, 1)           # 3rd c. BCE - 1st c. CE
    OLDEST_LXX = ("Vaticanus/Sinaiticus", 4, 4)  # 4th century CE
    HEXAPLARIC = ("Origen's Hexapla", 3, 3)      # 3rd century CE (fragments)
    MASORETIC = ("Masoretic Text", 7, 10)        # 7th-10th century CE
    VULGATE_PESHITTA = ("Vulgate/Peshitta", 4, 5)  # Confirmation witnesses

    def __init__(self, display_name: str, century_start: int, century_end: int):
        self.display_name = display_name
        self.century_start = century_start
        self.century_end = century_end

    @property
    def reliability_weight(self) -> float:
        """Earlier = higher weight."""
        avg_century = (self.century_start + self.century_end) / 2
        # Normalize: -3 -> 1.0, 10 -> 0.3
        return max(0.3, 1.0 - (avg_century + 3) * 0.05)


class DivergenceType(Enum):
    """Classification of MT-LXX differences."""
    LEXICAL = "lexical"                    # Different word choice (almah -> parthenos)
    SEMANTIC_EXPANSION = "expansion"        # LXX adds meaning
    SEMANTIC_RESTRICTION = "restriction"    # LXX narrows meaning
    GRAMMATICAL = "grammatical"             # Gender, number, case changes
    ADDITION = "addition"                   # LXX adds words/phrases
    OMISSION = "omission"                   # LXX omits MT content
    TRANSLATIONAL = "translational"         # Interpretive rendering
    TEXTUAL_VARIANT = "variant"             # Different Vorlage (Hebrew source)


class ChristologicalCategory(Enum):
    """Theological categories for Christological content."""
    VIRGIN_BIRTH = "virgin_birth"
    INCARNATION = "incarnation"
    PASSION = "passion"
    RESURRECTION = "resurrection"
    DIVINE_NATURE = "divine_nature"
    MESSIANIC_TITLE = "messianic_title"
    PROPHETIC_FULFILLMENT = "prophetic_fulfillment"
    SACRIFICIAL = "sacrificial"
    ROYAL_DAVIDIC = "royal_davidic"
    SOTERIOLOGICAL = "soteriological"
    THEOPHANIC = "theophanic"
    PRIESTLY = "priestly"


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class ManuscriptWitness:
    """Evidence from a specific manuscript."""
    manuscript_id: str              # "4QIsa^a", "Codex Vaticanus"
    manuscript_type: ManuscriptPriority
    date_range: str                 # "150-100 BCE"
    century_numeric: int            # -2 for 2nd century BCE
    reading: str                    # The text in this manuscript
    reading_transliterated: str
    supports_lxx: bool
    supports_mt: bool
    notes: str
    reliability_score: float        # Based on age and completeness

    @property
    def is_oldest_available(self) -> bool:
        return self.century_numeric <= -1  # Pre-Christian


@dataclass
class NTQuotation:
    """New Testament quotation of an OT passage."""
    nt_verse: str
    nt_text_greek: str
    quote_type: str                 # "exact", "adapted", "allusion", "echo"
    follows_lxx: bool
    follows_mt: bool
    verbal_agreement_lxx: float     # 0-1 similarity score
    verbal_agreement_mt: float
    theological_significance: str


@dataclass
class PatristicWitness:
    """Church Father's interpretation."""
    father: str                     # "Chrysostom", "Irenaeus"
    era: str                        # "ante-nicene", "nicene", "post-nicene"
    work: str
    citation: str
    interpretation: str
    text_preference: str            # "lxx", "mt", "both"
    christological_reading: bool


@dataclass
class LXXDivergence:
    """A single divergence between MT and LXX."""
    divergence_id: str
    verse_id: str

    # Textual content
    mt_text_hebrew: str
    mt_text_transliterated: str
    mt_gloss: str
    lxx_text_greek: str
    lxx_text_transliterated: str
    lxx_gloss: str

    # Classification
    divergence_type: DivergenceType
    christological_category: Optional[ChristologicalCategory] = None
    christological_significance: str = ""

    # Manuscript evidence
    manuscript_witnesses: List[ManuscriptWitness] = field(default_factory=list)
    oldest_witness: Optional[ManuscriptWitness] = None
    oldest_supports: str = "unknown"  # "lxx", "mt", "neither", "unique"

    # NT and Patristic support
    nt_quotations: List[NTQuotation] = field(default_factory=list)
    patristic_witnesses: List[PatristicWitness] = field(default_factory=list)

    # Scoring
    divergence_score: float = 0.0       # How significant is the difference
    christological_score: float = 0.0   # Christological importance
    manuscript_confidence: float = 0.0  # Confidence from manuscript priority
    composite_score: float = 0.0        # Combined score

    def compute_composite_score(self) -> float:
        """Weighted combination of evidence."""
        weights = {
            'divergence': 0.20,
            'christological': 0.30,
            'manuscript': 0.25,
            'nt_support': 0.15,
            'patristic': 0.10
        }

        nt_support = len([q for q in self.nt_quotations if q.follows_lxx]) / max(1, len(self.nt_quotations))
        patristic_support = len([p for p in self.patristic_witnesses if p.christological_reading]) / max(1, len(self.patristic_witnesses))

        self.composite_score = (
            weights['divergence'] * self.divergence_score +
            weights['christological'] * self.christological_score +
            weights['manuscript'] * self.manuscript_confidence +
            weights['nt_support'] * nt_support +
            weights['patristic'] * patristic_support
        )
        return self.composite_score


@dataclass
class LXXAnalysisResult:
    """Complete analysis result for a verse."""
    verse_id: str
    mt_verse_id: str                # May differ (Psalm numbering)
    lxx_verse_id: str

    divergences: List[LXXDivergence]
    primary_christological_insight: str
    christological_category: Optional[ChristologicalCategory]

    # Counts
    total_divergence_count: int
    christological_divergence_count: int

    # Aggregate scores
    nt_support_strength: float
    patristic_unanimity: float
    manuscript_priority_score: float
    overall_significance: float

    # Metadata
    analysis_timestamp: str
    cache_hit: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "verse_id": self.verse_id,
            "mt_verse_id": self.mt_verse_id,
            "lxx_verse_id": self.lxx_verse_id,
            "primary_christological_insight": self.primary_christological_insight,
            "christological_category": self.christological_category.value if self.christological_category else None,
            "total_divergence_count": self.total_divergence_count,
            "christological_divergence_count": self.christological_divergence_count,
            "nt_support_strength": self.nt_support_strength,
            "patristic_unanimity": self.patristic_unanimity,
            "manuscript_priority_score": self.manuscript_priority_score,
            "overall_significance": self.overall_significance,
            "analysis_timestamp": self.analysis_timestamp,
            "cache_hit": self.cache_hit,
        }


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LXXExtractorConfig:
    """Configuration for LXX Christological Extractor."""

    # Corpus paths
    lxx_corpus_path: str = "data/corpora/lxx"
    mt_corpus_path: str = "data/corpora/mt"
    catalog_path: str = "data/lxx_christological_catalog.json"

    # Detection thresholds
    alignment_threshold: float = 0.7      # Below this = potential divergence
    min_divergence_score: float = 0.3     # Minimum to report
    christological_threshold: float = 0.5  # Minimum for Christological significance

    # Evidence gathering
    include_manuscripts: bool = True
    include_nt_quotations: bool = True
    include_patristic: bool = True
    max_patristic_witnesses: int = 10

    # Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = 604800       # 1 week

    # Performance
    batch_size: int = 50
    parallel_analysis: bool = True


# =============================================================================
# MAIN ENGINE CLASS
# =============================================================================

class LXXChristologicalExtractor:
    """
    The Third Impossible Oracle: discovers Christological content in LXX.

    Architecture:
    +-------------------------------------------------------------+
    |              LXXChristologicalExtractor                      |
    |                                                              |
    |  +-------------+  +--------------+  +--------------+        |
    |  | TextAligner |  | Divergence   |  | Christology  |        |
    |  |             |  | Detector     |  | Classifier   |        |
    |  | MT <-> LXX  |  |              |  |              |        |
    |  | alignment   |  | Type & score |  | Category     |        |
    |  +------+------+  +------+-------+  +------+-------+        |
    |         |                |                 |                 |
    |         +----------------+-----------------+                 |
    |                          v                                   |
    |              +-------------------------+                     |
    |              |   EvidenceGatherer      |                     |
    |              |                         |                     |
    |              |  - Manuscripts (DSS)    |                     |
    |              |  - NT Quotations        |                     |
    |              |  - Patristic Witness    |                     |
    |              +-----------+-------------+                     |
    |                          |                                   |
    |                          v                                   |
    |              +-------------------------+                     |
    |              |   LXXAnalysisResult     |                     |
    |              +-------------------------+                     |
    +-------------------------------------------------------------+
    """

    # Known Christological verses (pre-cataloged)
    KNOWN_CHRISTOLOGICAL_VERSES: Dict[str, ChristologicalCategory] = {
        "ISA.7.14": ChristologicalCategory.VIRGIN_BIRTH,
        "PSA.40.6": ChristologicalCategory.INCARNATION,
        "GEN.3.15": ChristologicalCategory.SOTERIOLOGICAL,
        "PSA.22.16": ChristologicalCategory.PASSION,
        "ISA.53.8": ChristologicalCategory.PASSION,
        "PSA.16.10": ChristologicalCategory.RESURRECTION,
        "ISA.9.6": ChristologicalCategory.DIVINE_NATURE,
        "MIC.5.2": ChristologicalCategory.DIVINE_NATURE,
        "ZEC.12.10": ChristologicalCategory.PASSION,
        "PSA.110.1": ChristologicalCategory.ROYAL_DAVIDIC,
        "DAN.7.13": ChristologicalCategory.DIVINE_NATURE,
        "ISA.61.1": ChristologicalCategory.MESSIANIC_TITLE,
        "MAL.3.1": ChristologicalCategory.PROPHETIC_FULFILLMENT,
        "PSA.2.7": ChristologicalCategory.DIVINE_NATURE,
        "ISA.11.1": ChristologicalCategory.ROYAL_DAVIDIC,
    }

    # Psalm numbering offset (LXX = MT - 1 for Psalms 10-147)
    PSALM_OFFSET_RANGE = (10, 147)

    # Extended Christological markers with categories and scores
    CHRISTOLOGICAL_MARKERS: Dict[str, Tuple[ChristologicalCategory, float]] = {
        # Virgin Birth / Incarnation
        "παρθένος": (ChristologicalCategory.VIRGIN_BIRTH, 0.95),
        "σῶμα": (ChristologicalCategory.INCARNATION, 0.80),
        # Passion / Suffering
        "ὤρυξαν": (ChristologicalCategory.PASSION, 0.92),
        "ἐξεκέντησαν": (ChristologicalCategory.PASSION, 0.90),
        "ἀμνός": (ChristologicalCategory.SACRIFICIAL, 0.80),
        # Divine Nature
        "θεὸς ἰσχυρός": (ChristologicalCategory.DIVINE_NATURE, 0.90),
        # Messianic Titles
        "χριστός": (ChristologicalCategory.MESSIANIC_TITLE, 0.88),
        "υἱὸς ἀνθρώπου": (ChristologicalCategory.MESSIANIC_TITLE, 0.85),
        # Resurrection
        "ἀνάστασις": (ChristologicalCategory.RESURRECTION, 0.88),
        "διαφθοράν": (ChristologicalCategory.RESURRECTION, 0.80),
        # Soteriological
        "σωτηρία": (ChristologicalCategory.SOTERIOLOGICAL, 0.75),
        "λύτρωσις": (ChristologicalCategory.SOTERIOLOGICAL, 0.78),
    }

    # NT passages preferring LXX over MT
    NT_LXX_PREFERENCES: Dict[str, List[str]] = {
        "ISA.7.14": ["MAT.1.23"],
        "PSA.40.6": ["HEB.10.5"],
        "PSA.22.16": ["JHN.19.37", "REV.1.7"],
        "ISA.53.4": ["MAT.8.17"],
        "PSA.16.10": ["ACT.2.27", "ACT.13.35"],
        "ISA.61.1": ["LUK.4.18"],
        "PSA.110.1": ["MAT.22.44", "MRK.12.36", "LUK.20.42", "ACT.2.34", "HEB.1.13"],
        "DAN.7.13": ["MAT.24.30", "MAT.26.64", "MRK.13.26"],
        "ZEC.12.10": ["JHN.19.37", "REV.1.7"],
        "MIC.5.2": ["MAT.2.6"],
        "MAL.3.1": ["MAT.11.10", "MRK.1.2", "LUK.7.27"],
    }

    def __init__(
        self,
        lxx_client: Any,  # LXXCorpusClient
        mt_client: Any,   # TextFabricIntegration or similar
        neo4j_client: Any,
        redis_client: Any,
        config: Optional[LXXExtractorConfig] = None
    ):
        self.lxx = lxx_client
        self.mt = mt_client
        self.neo4j = neo4j_client
        self.redis = redis_client
        self.config = config or LXXExtractorConfig()
        self.logger = logging.getLogger(__name__)

    async def extract_christological_content(
        self,
        verse_id: str,
        force_recompute: bool = False
    ) -> LXXAnalysisResult:
        """Main entry point for LXX Christological analysis."""

        # Check cache
        if not force_recompute:
            cached = await self._get_cached(verse_id)
            if cached:
                return cached

        # Convert verse numbering if needed
        mt_verse_id, lxx_verse_id = await self._normalize_verse_ids(verse_id)

        # Fetch texts
        mt_data = await self.mt.get_verse(mt_verse_id)
        lxx_data = await self.lxx.get_verse(lxx_verse_id)

        if not mt_data or not lxx_data:
            return self._empty_result(verse_id, mt_verse_id, lxx_verse_id)

        # Align texts
        alignments = await self._align_texts(mt_data, lxx_data)

        # Detect divergences
        divergences = await self._detect_divergences(alignments, verse_id)

        # Gather evidence for each divergence
        for div in divergences:
            if self.config.include_manuscripts:
                div.manuscript_witnesses = await self._gather_manuscripts(verse_id)
            if self.config.include_nt_quotations:
                div.nt_quotations = await self._find_nt_quotations(verse_id)
            if self.config.include_patristic:
                div.patristic_witnesses = await self._gather_patristic(verse_id)

            div.oldest_witness, div.oldest_supports = self._determine_oldest(div.manuscript_witnesses)
            await self._classify_christological(div)
            div.compute_composite_score()

        # Build result
        result = self._build_result(verse_id, mt_verse_id, lxx_verse_id, divergences)

        # Cache and store
        await self._cache_result(result)
        await self._store_to_neo4j(result)

        return result

    async def _normalize_verse_ids(self, verse_id: str) -> Tuple[str, str]:
        """Handle MT/LXX numbering differences (especially Psalms)."""
        parts = verse_id.split(".")
        book = parts[0]

        if book == "PSA" and len(parts) >= 2:
            psalm_num = int(parts[1])
            if self.PSALM_OFFSET_RANGE[0] <= psalm_num <= self.PSALM_OFFSET_RANGE[1]:
                lxx_psalm = psalm_num - 1
                lxx_verse_id = f"PSA.{lxx_psalm}.{'.'.join(parts[2:])}" if len(parts) > 2 else f"PSA.{lxx_psalm}"
                return verse_id, lxx_verse_id

        return verse_id, verse_id

    async def _align_texts(
        self,
        mt_data: Dict,
        lxx_data: Dict
    ) -> List[Tuple[Dict, Optional[Dict], float]]:
        """Align Hebrew and Greek words."""
        mt_words = mt_data.get("words", [])
        lxx_words = lxx_data.get("words", [])

        alignments = []
        for mt_word in mt_words:
            best_match = None
            best_score = 0.0

            for lxx_word in lxx_words:
                score = await self._word_alignment_score(mt_word, lxx_word)
                if score > best_score:
                    best_score = score
                    best_match = lxx_word

            alignments.append((mt_word, best_match, best_score))

        return alignments

    async def _word_alignment_score(
        self,
        mt_word: Dict,
        lxx_word: Dict
    ) -> float:
        """Compute semantic alignment score between Hebrew and Greek words."""
        if not lxx_word:
            return 0.0

        score = 0.0
        weights = {
            'lexicon': 0.50,
            'morphology': 0.25,
            'semantic_field': 0.25
        }

        # 1. Lexicon lookup
        mt_lemma = mt_word.get("lemma", "")
        lxx_lemma = lxx_word.get("lemma", "")

        lexicon_mappings = await self._get_lexicon_mappings(mt_lemma)
        if lxx_lemma in lexicon_mappings:
            score += weights['lexicon'] * lexicon_mappings[lxx_lemma]
        elif self._similar_gloss(mt_word.get("gloss", ""), lxx_word.get("gloss", "")):
            score += weights['lexicon'] * 0.6

        # 2. Morphological compatibility
        mt_morph = mt_word.get("morphology", {})
        lxx_morph = lxx_word.get("morphology", {})
        morph_score = self._morphology_compatibility(mt_morph, lxx_morph)
        score += weights['morphology'] * morph_score

        # 3. Semantic field overlap
        mt_domains = set(mt_word.get("semantic_domains", []))
        lxx_domains = set(lxx_word.get("semantic_domains", []))
        if mt_domains and lxx_domains:
            overlap = len(mt_domains & lxx_domains) / len(mt_domains | lxx_domains)
            score += weights['semantic_field'] * overlap

        return min(1.0, score)

    async def _get_lexicon_mappings(self, hebrew_lemma: str) -> Dict[str, float]:
        """Get Greek translations for Hebrew lemma with confidence scores."""
        cache_key = f"lex_map:{hebrew_lemma}"

        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return cached

        # Query lexicon database
        if self.neo4j:
            mappings = await self.neo4j.query(
                """
                MATCH (h:HebrewLemma {lemma: $lemma})-[t:TRANSLATES_TO]->(g:GreekLemma)
                RETURN g.lemma AS greek, t.confidence AS confidence
                """,
                {"lemma": hebrew_lemma}
            )
            result = {m["greek"]: m["confidence"] for m in mappings}
        else:
            result = {}

        if self.redis:
            await self.redis.set(cache_key, result, ex=86400)

        return result

    def _similar_gloss(self, gloss1: str, gloss2: str) -> bool:
        """Check if glosses are semantically similar."""
        if not gloss1 or not gloss2:
            return False

        g1 = set(gloss1.lower().split())
        g2 = set(gloss2.lower().split())

        if not g1 or not g2:
            return False
        similarity = len(g1 & g2) / len(g1 | g2)
        return similarity >= 0.3

    def _morphology_compatibility(
        self,
        mt_morph: Dict,
        lxx_morph: Dict
    ) -> float:
        """Check morphological feature compatibility."""
        compatible_features = 0
        total_features = 0

        # Number agreement
        if "number" in mt_morph and "number" in lxx_morph:
            total_features += 1
            if mt_morph["number"] == lxx_morph["number"]:
                compatible_features += 1
            elif mt_morph["number"] == "dual" and lxx_morph["number"] == "plural":
                compatible_features += 0.8

        # Person agreement
        if "person" in mt_morph and "person" in lxx_morph:
            total_features += 1
            if mt_morph["person"] == lxx_morph["person"]:
                compatible_features += 1

        # Gender
        if "gender" in mt_morph and "gender" in lxx_morph:
            total_features += 1
            if mt_morph["gender"] == lxx_morph["gender"]:
                compatible_features += 1

        # Verbal stem/voice
        if "stem" in mt_morph or "voice" in lxx_morph:
            total_features += 1
            stem_voice_map = {
                "qal": "active",
                "niphal": "passive",
                "piel": "active",
                "hiphil": "active",
                "hophal": "passive",
                "hithpael": "middle"
            }
            mt_voice = stem_voice_map.get(mt_morph.get("stem", ""), "")
            lxx_voice = lxx_morph.get("voice", "")
            if mt_voice == lxx_voice:
                compatible_features += 1

        return compatible_features / max(1, total_features)

    async def _detect_divergences(
        self,
        alignments: List[Tuple],
        verse_id: str
    ) -> List[LXXDivergence]:
        """Identify significant divergences from alignments."""
        divergences = []

        for mt_word, lxx_word, alignment_score in alignments:
            if alignment_score < self.config.alignment_threshold:
                div_type = await self._classify_divergence_type(mt_word, lxx_word)
                div_score = 1.0 - alignment_score

                if div_score >= self.config.min_divergence_score:
                    divergences.append(LXXDivergence(
                        divergence_id=f"div_{verse_id}_{len(divergences)}",
                        verse_id=verse_id,
                        mt_text_hebrew=mt_word.get("text", ""),
                        mt_text_transliterated=mt_word.get("translit", ""),
                        mt_gloss=mt_word.get("gloss", ""),
                        lxx_text_greek=lxx_word.get("text", "") if lxx_word else "",
                        lxx_text_transliterated=lxx_word.get("translit", "") if lxx_word else "",
                        lxx_gloss=lxx_word.get("gloss", "") if lxx_word else "",
                        divergence_type=div_type,
                        divergence_score=div_score
                    ))

        return divergences

    async def _classify_divergence_type(
        self,
        mt_word: Dict,
        lxx_word: Optional[Dict]
    ) -> DivergenceType:
        """Classify the type of MT-LXX divergence."""
        if not lxx_word:
            return DivergenceType.OMISSION

        mt_text = mt_word.get("text", "")
        lxx_text = lxx_word.get("text", "")

        if not mt_text and lxx_text:
            return DivergenceType.ADDITION

        lexicon_mappings = await self._get_lexicon_mappings(mt_word.get("lemma", ""))
        lxx_lemma = lxx_word.get("lemma", "")

        if lxx_lemma and lxx_lemma not in lexicon_mappings:
            return DivergenceType.TEXTUAL_VARIANT

        mt_gloss = mt_word.get("gloss", "")
        lxx_gloss = lxx_word.get("gloss", "")

        if self._similar_gloss(mt_gloss, lxx_gloss):
            return DivergenceType.GRAMMATICAL
        else:
            mt_domains = set(mt_word.get("semantic_domains", []))
            lxx_domains = set(lxx_word.get("semantic_domains", []))

            if lxx_domains and mt_domains:
                if lxx_domains > mt_domains:
                    return DivergenceType.SEMANTIC_EXPANSION
                elif lxx_domains < mt_domains:
                    return DivergenceType.SEMANTIC_RESTRICTION

            return DivergenceType.LEXICAL

    async def _classify_christological(self, divergence: LXXDivergence):
        """Classify Christological significance."""
        # Check known catalog
        if divergence.verse_id in self.KNOWN_CHRISTOLOGICAL_VERSES:
            divergence.christological_category = self.KNOWN_CHRISTOLOGICAL_VERSES[divergence.verse_id]
            divergence.christological_score = 0.95
            divergence.christological_significance = f"Known Christological verse: {divergence.christological_category.value}"
            return

        # Analyze Greek text for Christological markers
        greek = divergence.lxx_text_greek.lower()

        for marker, (category, score) in self.CHRISTOLOGICAL_MARKERS.items():
            if marker in greek:
                divergence.christological_category = category
                divergence.christological_score = score
                divergence.christological_significance = f"Contains Christological marker: {marker}"
                return

        # Check NT quotation support
        if any(q.follows_lxx for q in divergence.nt_quotations):
            divergence.christological_score = 0.70
            divergence.christological_significance = "NT prefers LXX reading"

    async def _gather_manuscripts(self, verse_id: str) -> List[ManuscriptWitness]:
        """Gather manuscript evidence for a verse."""
        witnesses = []

        if not self.neo4j:
            return witnesses

        # Query DSS evidence
        dss_readings = await self.neo4j.query(
            """
            MATCH (m:Manuscript {type: 'DSS'})-[:CONTAINS]->(r:Reading)
            WHERE r.verse_id = $verse_id
            RETURN m.id AS ms_id, m.name AS name, m.date_range AS date,
                   m.century AS century, r.text AS reading, r.notes AS notes
            """,
            {"verse_id": verse_id}
        )

        for dss in dss_readings:
            witnesses.append(ManuscriptWitness(
                manuscript_id=dss["ms_id"],
                manuscript_type=ManuscriptPriority.DSS,
                date_range=dss["date"],
                century_numeric=dss["century"],
                reading=dss["reading"],
                reading_transliterated=self._transliterate(dss["reading"]),
                supports_lxx=await self._reading_supports_lxx(dss["reading"], verse_id),
                supports_mt=await self._reading_supports_mt(dss["reading"], verse_id),
                notes=dss.get("notes", ""),
                reliability_score=ManuscriptPriority.DSS.reliability_weight
            ))

        # Query early LXX manuscripts
        lxx_readings = await self.neo4j.query(
            """
            MATCH (m:Manuscript)-[:CONTAINS]->(r:Reading)
            WHERE r.verse_id = $verse_id
              AND m.type IN ['uncial', 'majuscule']
              AND m.century <= 5
            RETURN m.id AS ms_id, m.name AS name, m.date_range AS date,
                   m.century AS century, r.text AS reading, r.notes AS notes
            """,
            {"verse_id": verse_id}
        )

        for ms in lxx_readings:
            witnesses.append(ManuscriptWitness(
                manuscript_id=ms["ms_id"],
                manuscript_type=ManuscriptPriority.OLDEST_LXX,
                date_range=ms["date"],
                century_numeric=ms["century"],
                reading=ms["reading"],
                reading_transliterated=self._transliterate(ms["reading"]),
                supports_lxx=True,
                supports_mt=await self._reading_supports_mt(ms["reading"], verse_id),
                notes=ms.get("notes", ""),
                reliability_score=ManuscriptPriority.OLDEST_LXX.reliability_weight
            ))

        return witnesses

    async def _reading_supports_lxx(self, reading: str, verse_id: str) -> bool:
        """Check if a manuscript reading supports the LXX text."""
        lxx_text = await self.lxx.get_verse_text(verse_id)
        if not lxx_text:
            return False

        matcher = SequenceMatcher(None, reading.lower(), lxx_text.lower())
        return matcher.ratio() >= 0.7

    async def _reading_supports_mt(self, reading: str, verse_id: str) -> bool:
        """Check if a manuscript reading supports the MT text."""
        mt_data = await self.mt.get_verse(verse_id)
        if not mt_data:
            return False
        mt_text = mt_data.get("text", "")

        matcher = SequenceMatcher(None, reading.lower(), mt_text.lower())
        return matcher.ratio() >= 0.7

    def _transliterate(self, text: str) -> str:
        """Transliterate Hebrew or Greek to Latin characters."""
        hebrew_map = {
            'א': "'", 'ב': "b", 'ג': "g", 'ד': "d", 'ה': "h",
            'ו': "w", 'ז': "z", 'ח': "h", 'ט': "t", 'י': "y",
            'כ': "k", 'ך': "k", 'ל': "l", 'מ': "m", 'ם': "m",
            'נ': "n", 'ן': "n", 'ס': "s", 'ע': "'", 'פ': "p",
            'ף': "p", 'צ': "ts", 'ץ': "ts", 'ק': "q", 'ר': "r",
            'ש': "sh", 'ת': "t"
        }

        greek_map = {
            'α': "a", 'β': "b", 'γ': "g", 'δ': "d", 'ε': "e",
            'ζ': "z", 'η': "e", 'θ': "th", 'ι': "i", 'κ': "k",
            'λ': "l", 'μ': "m", 'ν': "n", 'ξ': "x", 'ο': "o",
            'π': "p", 'ρ': "r", 'σ': "s", 'ς': "s", 'τ': "t",
            'υ': "y", 'φ': "ph", 'χ': "ch", 'ψ': "ps", 'ω': "o"
        }

        result = []
        for char in text.lower():
            if char in hebrew_map:
                result.append(hebrew_map[char])
            elif char in greek_map:
                result.append(greek_map[char])
            else:
                result.append(char)

        return ''.join(result)

    async def _find_nt_quotations(self, verse_id: str) -> List[NTQuotation]:
        """Find NT quotations of this OT verse."""
        quotations = []

        # Check known preferences first
        if verse_id in self.NT_LXX_PREFERENCES:
            for nt_ref in self.NT_LXX_PREFERENCES[verse_id]:
                quotations.append(NTQuotation(
                    nt_verse=nt_ref,
                    nt_text_greek="",
                    quote_type="quotation",
                    follows_lxx=True,
                    follows_mt=False,
                    verbal_agreement_lxx=0.85,
                    verbal_agreement_mt=0.5,
                    theological_significance=f"NT ({nt_ref}) follows LXX reading"
                ))

        # Query cross-reference database
        if self.neo4j:
            nt_refs = await self.neo4j.query(
                """
                MATCH (ot:Verse {id: $verse_id})<-[q:QUOTES]-(nt:Verse)
                WHERE nt.testament = 'NT'
                RETURN nt.id AS nt_verse, nt.text_greek AS nt_text,
                       q.quote_type AS quote_type, q.verbal_agreement AS agreement
                """,
                {"verse_id": verse_id}
            )

            for ref in nt_refs:
                lxx_text = await self.lxx.get_verse_text(verse_id)
                nt_text = ref.get("nt_text", "")

                lxx_agreement = self._verbal_agreement(nt_text, lxx_text) if lxx_text else 0.0

                quotations.append(NTQuotation(
                    nt_verse=ref["nt_verse"],
                    nt_text_greek=nt_text,
                    quote_type=ref.get("quote_type", "quotation"),
                    follows_lxx=lxx_agreement > 0.6,
                    follows_mt=False,
                    verbal_agreement_lxx=lxx_agreement,
                    verbal_agreement_mt=0.0,
                    theological_significance=""
                ))

        return quotations

    def _verbal_agreement(self, text1: str, text2: str) -> float:
        """Calculate verbal agreement between two texts."""
        if not text1 or not text2:
            return 0.0

        t1 = text1.lower().strip()
        t2 = text2.lower().strip()

        matcher = SequenceMatcher(None, t1, t2)
        return matcher.ratio()

    async def _gather_patristic(self, verse_id: str) -> List[PatristicWitness]:
        """Gather patristic interpretations of this verse."""
        witnesses = []

        if not self.neo4j:
            return witnesses

        patristic_refs = await self.neo4j.query(
            """
            MATCH (v:Verse {id: $verse_id})<-[:INTERPRETS]-(p:PatristicCitation)
            MATCH (p)-[:AUTHORED_BY]->(f:Father)
            RETURN f.name AS father, f.era AS era, p.work AS work,
                   p.citation AS citation, p.interpretation AS interpretation,
                   p.text_preference AS text_pref, p.christological AS christological
            ORDER BY
                CASE f.era
                    WHEN 'apostolic' THEN 1
                    WHEN 'ante-nicene' THEN 2
                    WHEN 'nicene' THEN 3
                    WHEN 'post-nicene' THEN 4
                END
            LIMIT $limit
            """,
            {"verse_id": verse_id, "limit": self.config.max_patristic_witnesses}
        )

        for ref in patristic_refs:
            witnesses.append(PatristicWitness(
                father=ref["father"],
                era=ref["era"],
                work=ref["work"],
                citation=ref["citation"],
                interpretation=ref["interpretation"],
                text_preference=ref.get("text_pref", "lxx"),
                christological_reading=ref.get("christological", False)
            ))

        return witnesses

    def _determine_oldest(
        self,
        witnesses: List[ManuscriptWitness]
    ) -> Tuple[Optional[ManuscriptWitness], str]:
        """Find oldest witness and what it supports."""
        if not witnesses:
            return None, "unknown"

        sorted_witnesses = sorted(witnesses, key=lambda w: w.century_numeric)
        oldest = sorted_witnesses[0]

        if oldest.supports_lxx and not oldest.supports_mt:
            return oldest, "lxx"
        elif oldest.supports_mt and not oldest.supports_lxx:
            return oldest, "mt"
        elif oldest.supports_lxx and oldest.supports_mt:
            return oldest, "both"
        else:
            return oldest, "unique"

    def _build_result(
        self,
        verse_id: str,
        mt_verse_id: str,
        lxx_verse_id: str,
        divergences: List[LXXDivergence]
    ) -> LXXAnalysisResult:
        """Build the final analysis result."""
        christological_divs = [
            d for d in divergences if d.christological_category is not None
        ]

        primary_category = None
        primary_insight = "No significant Christological divergence detected"

        if christological_divs:
            best = max(christological_divs, key=lambda d: d.composite_score)
            primary_category = best.christological_category
            primary_insight = best.christological_significance or \
                f"LXX reading supports {primary_category.value} interpretation"

        # Calculate aggregate scores
        nt_strength = 0.0
        patristic_unanimity = 0.0
        ms_priority = 0.0

        if divergences:
            all_nt = [q for d in divergences for q in d.nt_quotations]
            if all_nt:
                nt_strength = len([q for q in all_nt if q.follows_lxx]) / len(all_nt)

            all_patristic = [p for d in divergences for p in d.patristic_witnesses]
            if all_patristic:
                patristic_unanimity = len([p for p in all_patristic if p.christological_reading]) / len(all_patristic)

            ms_scores = [d.manuscript_confidence for d in divergences if d.manuscript_confidence > 0]
            if ms_scores:
                ms_priority = sum(ms_scores) / len(ms_scores)

        overall = 0.0
        if christological_divs:
            scores = [d.composite_score for d in christological_divs]
            overall = sum(scores) / len(scores)

        return LXXAnalysisResult(
            verse_id=verse_id,
            mt_verse_id=mt_verse_id,
            lxx_verse_id=lxx_verse_id,
            divergences=divergences,
            primary_christological_insight=primary_insight,
            christological_category=primary_category,
            total_divergence_count=len(divergences),
            christological_divergence_count=len(christological_divs),
            nt_support_strength=nt_strength,
            patristic_unanimity=patristic_unanimity,
            manuscript_priority_score=ms_priority,
            overall_significance=overall,
            analysis_timestamp=datetime.utcnow().isoformat()
        )

    def _empty_result(
        self,
        verse_id: str,
        mt_verse_id: str,
        lxx_verse_id: str
    ) -> LXXAnalysisResult:
        """Return empty result when data is missing."""
        return LXXAnalysisResult(
            verse_id=verse_id,
            mt_verse_id=mt_verse_id,
            lxx_verse_id=lxx_verse_id,
            divergences=[],
            primary_christological_insight="No data available",
            christological_category=None,
            total_divergence_count=0,
            christological_divergence_count=0,
            nt_support_strength=0.0,
            patristic_unanimity=0.0,
            manuscript_priority_score=0.0,
            overall_significance=0.0,
            analysis_timestamp=datetime.utcnow().isoformat()
        )

    async def _get_cached(self, verse_id: str) -> Optional[LXXAnalysisResult]:
        """Retrieve cached analysis result."""
        if not self.config.cache_enabled or not self.redis:
            return None

        cache_key = f"lxx_analysis:{verse_id}"
        cached = await self.redis.get(cache_key)

        if cached:
            # Reconstruct from dict
            result = LXXAnalysisResult(
                verse_id=cached["verse_id"],
                mt_verse_id=cached["mt_verse_id"],
                lxx_verse_id=cached["lxx_verse_id"],
                divergences=[],  # Simplified for cache
                primary_christological_insight=cached["primary_christological_insight"],
                christological_category=ChristologicalCategory(cached["christological_category"]) if cached.get("christological_category") else None,
                total_divergence_count=cached["total_divergence_count"],
                christological_divergence_count=cached["christological_divergence_count"],
                nt_support_strength=cached["nt_support_strength"],
                patristic_unanimity=cached["patristic_unanimity"],
                manuscript_priority_score=cached["manuscript_priority_score"],
                overall_significance=cached["overall_significance"],
                analysis_timestamp=cached["analysis_timestamp"],
                cache_hit=True
            )
            return result

        return None

    async def _cache_result(self, result: LXXAnalysisResult):
        """Cache analysis result."""
        if not self.config.cache_enabled or not self.redis:
            return

        cache_key = f"lxx_analysis:{result.verse_id}"
        await self.redis.set(
            cache_key,
            result.to_dict(),
            ex=self.config.cache_ttl_seconds
        )

    async def _store_to_neo4j(self, result: LXXAnalysisResult):
        """Persist analysis result to Neo4j."""
        if not self.neo4j:
            return

        for div in result.divergences:
            await self.neo4j.query(
                """
                MERGE (d:LXXDivergence {divergence_id: $div_id})
                SET d.verse_id = $verse_id,
                    d.divergence_type = $div_type,
                    d.christological_category = $category,
                    d.mt_text = $mt_text,
                    d.lxx_text = $lxx_text,
                    d.composite_score = $score,
                    d.updated_at = datetime()

                WITH d
                MATCH (v:Verse {id: $verse_id})
                MERGE (v)-[r:HAS_LXX_DIVERGENCE]->(d)
                SET r.significance = $score,
                    r.category = $category
                """,
                {
                    "div_id": div.divergence_id,
                    "verse_id": div.verse_id,
                    "div_type": div.divergence_type.value,
                    "category": div.christological_category.value if div.christological_category else None,
                    "mt_text": div.mt_text_hebrew,
                    "lxx_text": div.lxx_text_greek,
                    "score": div.composite_score
                }
            )

    async def batch_extract(
        self,
        verse_ids: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[LXXAnalysisResult]:
        """Process multiple verses with parallelism."""
        results = []

        if self.config.parallel_analysis:
            for i in range(0, len(verse_ids), self.config.batch_size):
                batch = verse_ids[i:i + self.config.batch_size]
                batch_tasks = [
                    self.extract_christological_content(vid)
                    for vid in batch
                ]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                for vid, res in zip(batch, batch_results):
                    if isinstance(res, Exception):
                        self.logger.error(f"Failed to analyze {vid}: {res}")
                    else:
                        results.append(res)

                if progress_callback:
                    progress_callback(len(results), len(verse_ids))
        else:
            for vid in verse_ids:
                try:
                    result = await self.extract_christological_content(vid)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to analyze {vid}: {e}")

                if progress_callback:
                    progress_callback(len(results), len(verse_ids))

        return results

    async def scan_book_for_christological(
        self,
        book_code: str,
        min_score: float = 0.5
    ) -> List[LXXAnalysisResult]:
        """Scan entire book for Christological LXX divergences."""
        if not self.neo4j:
            return []

        verses = await self.neo4j.query(
            """
            MATCH (v:Verse)
            WHERE v.id STARTS WITH $book
            RETURN v.id AS verse_id
            ORDER BY v.id
            """,
            {"book": book_code}
        )

        verse_ids = [v["verse_id"] for v in verses]
        self.logger.info(f"Scanning {len(verse_ids)} verses in {book_code}")

        all_results = await self.batch_extract(verse_ids)

        significant = [
            r for r in all_results
            if r.overall_significance >= min_score
               and r.christological_divergence_count > 0
        ]

        self.logger.info(
            f"Found {len(significant)} verses with significant "
            f"Christological LXX divergences in {book_code}"
        )

        return significant
