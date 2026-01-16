# SESSION 05: LXX CHRISTOLOGICAL EXTRACTOR

## Session Overview

**Objective**: Implement the `LXXChristologicalExtractor` engine that discovers Christological content uniquely present in the Septuagint but absent from or muted in the Masoretic Text. This is the third of the Five Impossible Oracles.

**Prerequisites**:
- Review implementation of parts 1-4 for all 84 canonical books (if book count is relevant)
- Access to parallel LXX and MT corpus data, as well as original transcription data from "C:\Users\Edwin Boston\Desktop\MASTER_LINGUISTIC_CORPUS\RESTRUCTURED_CORPUS\Output\master_corpus.db"
- Access to oldest manuscript transcriptions (sources in `MASTER_LINGUISTIC_CORPUS/RESTRUCTURED_CORPUS`)
- Understanding of Greek/Hebrew textual criticism basics
- Familiarity with messianic prophecy traditions

**Critical Principle**: **Oldest transcriptions are most valid**. When manuscript variants exist, prioritize the earliest attested readings.

---

## Part 1: Understanding the Oracle Concept

### Core Capability
The Septuagint (LXX) was translated by Jewish scholars 200-300 years before Christ. Yet it often renders Hebrew text in ways that appear **prophetically Christological** - readings that the Church Fathers consistently noted pointed to Christ more clearly than the later Masoretic Text.

### Why This Is Significant

1. **Pre-Christian Witness**: LXX translators had no Christian agenda, yet their translation choices often favor Christological readings
2. **NT Preference**: New Testament authors quote the LXX over 300 times, often choosing specifically Christological LXX readings
3. **Patristic Foundation**: Church Fathers heavily relied on LXX as inspired text
4. **Apologetic Value**: LXX divergences that favor Christ predate Christianity
5. **Oldest Transcriptions Priority**: Earliest manuscript evidence carries greatest weight

### Manuscript Priority Hierarchy

```python
class ManuscriptPriority(Enum):
    """
    Textual authority ranking - oldest = most authoritative.
    Century values are negative for BCE, positive for CE.
    """
    DSS = ("Dead Sea Scrolls", -3, 1)           # 3rd c. BCE - 1st c. CE
    OLDEST_LXX = ("Vaticanus/Sinaiticus", 4, 4)  # 4th century CE
    HEXAPLARIC = ("Origen's Hexapla", 3, 3)      # 3rd century CE (fragments)
    MASORETIC = ("Masoretic Text", 7, 10)        # 7th-10th century CE
    VULGATE_PESHITTA = ("Vulgate/Peshitta", 4, 5) # Confirmation witnesses

    def __init__(self, name: str, century_start: int, century_end: int):
        self.display_name = name
        self.century_start = century_start
        self.century_end = century_end

    @property
    def reliability_weight(self) -> float:
        """Earlier = higher weight."""
        avg_century = (self.century_start + self.century_end) / 2
        # Normalize: -3 → 1.0, 10 → 0.3
        return max(0.3, 1.0 - (avg_century + 3) * 0.05)
```

### Canonical Examples

#### Example 1: Isaiah 7:14 - παρθένος vs עַלְמָה
```
MT (Hebrew): הָעַלְמָה (ha'almah) - "the young woman"
LXX (Greek): ἡ παρθένος (hē parthenos) - "the virgin"

Divergence Analysis:
┌────────────────────────────────────────────────────────────────┐
│ Aspect          │ MT Reading        │ LXX Reading              │
├─────────────────┼───────────────────┼──────────────────────────┤
│ Lexeme          │ עַלְמָה (almah)    │ παρθένος (parthenos)     │
│ Semantic Range  │ young woman (age) │ virgin (biological)      │
│ Definiteness    │ definite article  │ definite article         │
│ NT Citation     │ —                 │ MAT.1.23 (verbatim)      │
│ DSS Support     │ 1QIsaᵃ: עלמה      │ —                        │
│ Patristic       │ —                 │ Justin, Irenaeus, Origen │
└────────────────────────────────────────────────────────────────┘

Christological Score: 0.95 (VIRGIN_BIRTH)
```

#### Example 2: Psalm 40:6 (LXX 39:7) - Body Prepared
```
MT: אָזְנַיִם כָּרִיתָ לִּי (ears you have dug/opened for me)
LXX: σῶμα δὲ κατηρτίσω μοι (a body you have prepared for me)

Christological Significance:
- Hebrews 10:5 quotes LXX reading verbatim
- Points directly to Incarnation (body prepared for sacrifice)
- MT "opened ears" = obedience metaphor; LXX = embodiment

Christological Score: 0.92 (INCARNATION)
```

#### Example 3: Psalm 22:16 (LXX 21:17) - Pierced
```
MT: כָּאֲרִי (ka'ari) - "like a lion" [hands and feet]
LXX: ὤρυξαν (ōryxan) - "they pierced" [hands and feet]
DSS: כארו (ka'aru) - "they pierced" (4QPs supports LXX!)

Manuscript Evidence:
- DSS 4QPs^f supports "pierced" reading
- MT "like a lion" creates grammatical difficulty
- LXX describes crucifixion detail centuries before Christ

Christological Score: 0.94 (PASSION)
```

---

## Part 2: Core Data Structures

### File: `ml/engines/lxx_extractor.py`

```python
from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable
from difflib import SequenceMatcher

from db.neo4j_client import Neo4jClient
from db.redis_client import RedisClient
from integrations.lxx_corpus import LXXCorpusClient
from integrations.text_fabric import TextFabricClient
from config import LXXExtractorConfig


class DivergenceType(Enum):
    """Classification of MT-LXX differences."""
    LEXICAL = "lexical"                    # Different word choice (almah → parthenos)
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


@dataclass
class ManuscriptWitness:
    """Evidence from a specific manuscript."""
    manuscript_id: str              # "4QIsaᵃ", "Codex Vaticanus"
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
    christological_category: Optional[ChristologicalCategory]
    christological_significance: str

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
```

---

## Part 3: Main Extractor Implementation

```python
class LXXChristologicalExtractor:
    """
    The third Impossible Oracle: discovers Christological content in LXX.

    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │              LXXChristologicalExtractor                      │
    │                                                              │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
    │  │ TextAligner  │  │ Divergence   │  │ Christology  │       │
    │  │              │  │ Detector     │  │ Classifier   │       │
    │  │ MT ↔ LXX     │  │              │  │              │       │
    │  │ alignment    │  │ Type & score │  │ Category     │       │
    │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
    │         │                 │                 │                │
    │         └─────────────────┼─────────────────┘                │
    │                           ▼                                  │
    │              ┌─────────────────────────┐                    │
    │              │   EvidenceGatherer      │                    │
    │              │                         │                    │
    │              │  - Manuscripts (DSS)    │                    │
    │              │  - NT Quotations        │                    │
    │              │  - Patristic Witness    │                    │
    │              └───────────┬─────────────┘                    │
    │                          │                                  │
    │                          ▼                                  │
    │              ┌─────────────────────────┐                    │
    │              │   LXXAnalysisResult     │                    │
    │              └─────────────────────────┘                    │
    └─────────────────────────────────────────────────────────────┘
    """

    # Known Christological verses (pre-cataloged)
    KNOWN_CHRISTOLOGICAL_VERSES = {
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

    def __init__(
        self,
        lxx_client: LXXCorpusClient,
        mt_client: TextFabricClient,
        neo4j: Neo4jClient,
        redis: RedisClient,
        config: LXXExtractorConfig
    ):
        self.lxx = lxx_client
        self.mt = mt_client
        self.neo4j = neo4j
        self.redis = redis
        self.config = config
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

        # Align texts
        alignments = await self._align_texts(mt_data, lxx_data)

        # Detect divergences
        divergences = await self._detect_divergences(alignments, verse_id)

        # Gather evidence for each divergence
        for div in divergences:
            div.manuscript_witnesses = await self._gather_manuscripts(verse_id)
            div.nt_quotations = await self._find_nt_quotations(verse_id)
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
    ) -> List[Tuple[str, str, float]]:
        """Align Hebrew and Greek words."""
        mt_words = mt_data.get("words", [])
        lxx_words = lxx_data.get("words", [])

        alignments = []
        # Use semantic alignment with morphological awareness
        for i, mt_word in enumerate(mt_words):
            best_match = None
            best_score = 0.0

            for lxx_word in lxx_words:
                score = await self._word_alignment_score(mt_word, lxx_word)
                if score > best_score:
                    best_score = score
                    best_match = lxx_word

            alignments.append((mt_word, best_match, best_score))

        return alignments

    async def _detect_divergences(
        self,
        alignments: List[Tuple],
        verse_id: str
    ) -> List[LXXDivergence]:
        """Identify significant divergences from alignments."""
        divergences = []

        for mt_word, lxx_word, alignment_score in alignments:
            if alignment_score < self.config.alignment_threshold:
                # Potential divergence
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

    async def _classify_christological(self, divergence: LXXDivergence):
        """Classify Christological significance."""
        # Check known catalog
        if divergence.verse_id in self.KNOWN_CHRISTOLOGICAL_VERSES:
            divergence.christological_category = self.KNOWN_CHRISTOLOGICAL_VERSES[divergence.verse_id]
            divergence.christological_score = 0.95
            return

        # Analyze Greek text for Christological markers
        greek = divergence.lxx_text_greek.lower()

        christological_markers = {
            "παρθένος": (ChristologicalCategory.VIRGIN_BIRTH, 0.90),
            "χριστός": (ChristologicalCategory.MESSIANIC_TITLE, 0.85),
            "σῶμα": (ChristologicalCategory.INCARNATION, 0.80),
            "ὤρυξαν": (ChristologicalCategory.PASSION, 0.88),
            "ἐξεκέντησαν": (ChristologicalCategory.PASSION, 0.88),
            "ἀνάστασις": (ChristologicalCategory.RESURRECTION, 0.85),
        }

        for marker, (category, score) in christological_markers.items():
            if marker in greek:
                divergence.christological_category = category
                divergence.christological_score = score
                return

        # Check NT quotation support
        if any(q.follows_lxx for q in divergence.nt_quotations):
            divergence.christological_score = 0.70
            divergence.christological_significance = "NT prefers LXX reading"

    def _determine_oldest(
        self,
        witnesses: List[ManuscriptWitness]
    ) -> Tuple[Optional[ManuscriptWitness], str]:
        """Find oldest witness and what it supports."""
        if not witnesses:
            return None, "unknown"

        # Sort by century (oldest first)
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

    async def _word_alignment_score(
        self,
        mt_word: Dict,
        lxx_word: Dict
    ) -> float:
        """
        Compute semantic alignment score between Hebrew and Greek words.

        Uses multiple signals:
        1. Lexicon-based translation mappings
        2. Morphological compatibility
        3. Semantic field overlap
        """
        if not lxx_word:
            return 0.0

        score = 0.0
        weights = {
            'lexicon': 0.50,
            'morphology': 0.25,
            'semantic_field': 0.25
        }

        # 1. Lexicon lookup - check if LXX word is valid translation
        mt_lemma = mt_word.get("lemma", "")
        lxx_lemma = lxx_word.get("lemma", "")

        lexicon_mappings = await self._get_lexicon_mappings(mt_lemma)
        if lxx_lemma in lexicon_mappings:
            score += weights['lexicon'] * lexicon_mappings[lxx_lemma]
        elif self._similar_gloss(mt_word.get("gloss", ""), lxx_word.get("gloss", "")):
            score += weights['lexicon'] * 0.6  # Partial credit for gloss match

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
        # Check cache first
        cache_key = f"lex_map:{hebrew_lemma}"
        cached = await self.redis.get(cache_key)
        if cached:
            return cached

        # Query lexicon database
        mappings = await self.neo4j.query(
            """
            MATCH (h:HebrewLemma {lemma: $lemma})-[t:TRANSLATES_TO]->(g:GreekLemma)
            RETURN g.lemma AS greek, t.confidence AS confidence
            """,
            {"lemma": hebrew_lemma}
        )

        result = {m["greek"]: m["confidence"] for m in mappings}
        await self.redis.set(cache_key, result, ex=86400)  # Cache 1 day
        return result

    def _similar_gloss(self, gloss1: str, gloss2: str) -> bool:
        """Check if glosses are semantically similar."""
        if not gloss1 or not gloss2:
            return False

        # Normalize
        g1 = set(gloss1.lower().split())
        g2 = set(gloss2.lower().split())

        # Jaccard similarity
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

        # Number agreement (singular, plural, dual)
        if "number" in mt_morph and "number" in lxx_morph:
            total_features += 1
            if mt_morph["number"] == lxx_morph["number"]:
                compatible_features += 1
            elif mt_morph["number"] == "dual" and lxx_morph["number"] == "plural":
                compatible_features += 0.8  # Greek lacks dual

        # Person agreement
        if "person" in mt_morph and "person" in lxx_morph:
            total_features += 1
            if mt_morph["person"] == lxx_morph["person"]:
                compatible_features += 1

        # Gender (with allowance for Greek defaults)
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
        mt_gloss = mt_word.get("gloss", "")
        lxx_gloss = lxx_word.get("gloss", "")

        # Check for addition (LXX has more content)
        if not mt_text and lxx_text:
            return DivergenceType.ADDITION

        # Check lexical difference
        lexicon_mappings = await self._get_lexicon_mappings(mt_word.get("lemma", ""))
        lxx_lemma = lxx_word.get("lemma", "")

        if lxx_lemma and lxx_lemma not in lexicon_mappings:
            # Not a standard translation - could be textual variant
            return DivergenceType.TEXTUAL_VARIANT

        # Check for semantic shift
        if self._similar_gloss(mt_gloss, lxx_gloss):
            # Same general meaning - grammatical difference
            return DivergenceType.GRAMMATICAL
        else:
            # Different semantic content
            mt_domains = set(mt_word.get("semantic_domains", []))
            lxx_domains = set(lxx_word.get("semantic_domains", []))

            if lxx_domains and mt_domains:
                if lxx_domains > mt_domains:
                    return DivergenceType.SEMANTIC_EXPANSION
                elif lxx_domains < mt_domains:
                    return DivergenceType.SEMANTIC_RESTRICTION

            return DivergenceType.LEXICAL

    async def _gather_manuscripts(self, verse_id: str) -> List[ManuscriptWitness]:
        """Gather manuscript evidence for a verse."""
        witnesses = []

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

        # Query early LXX manuscripts (Vaticanus, Sinaiticus)
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

        # Use sequence matching for similarity
        matcher = SequenceMatcher(None, reading.lower(), lxx_text.lower())
        return matcher.ratio() >= 0.7

    async def _reading_supports_mt(self, reading: str, verse_id: str) -> bool:
        """Check if a manuscript reading supports the MT text."""
        mt_text = await self.mt.get_verse_text(verse_id)
        if not mt_text:
            return False

        matcher = SequenceMatcher(None, reading.lower(), mt_text.lower())
        return matcher.ratio() >= 0.7

    def _transliterate(self, text: str) -> str:
        """Transliterate Hebrew or Greek to Latin characters."""
        # Hebrew transliteration map
        hebrew_map = {
            'א': "'", 'ב': "b", 'ג': "g", 'ד': "d", 'ה': "h",
            'ו': "w", 'ז': "z", 'ח': "ḥ", 'ט': "ṭ", 'י': "y",
            'כ': "k", 'ך': "k", 'ל': "l", 'מ': "m", 'ם': "m",
            'נ': "n", 'ן': "n", 'ס': "s", 'ע': "'", 'פ': "p",
            'ף': "p", 'צ': "ṣ", 'ץ': "ṣ", 'ק': "q", 'ר': "r",
            'ש': "š", 'ת': "t"
        }

        # Greek transliteration map
        greek_map = {
            'α': "a", 'β': "b", 'γ': "g", 'δ': "d", 'ε': "e",
            'ζ': "z", 'η': "ē", 'θ': "th", 'ι': "i", 'κ': "k",
            'λ': "l", 'μ': "m", 'ν': "n", 'ξ': "x", 'ο': "o",
            'π': "p", 'ρ': "r", 'σ': "s", 'ς': "s", 'τ': "t",
            'υ': "y", 'φ': "ph", 'χ': "ch", 'ψ': "ps", 'ω': "ō"
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

        # Query cross-reference database for OT→NT connections
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
            # Get LXX and MT text for comparison
            lxx_text = await self.lxx.get_verse_text(verse_id)
            mt_text = await self.mt.get_verse_text(verse_id)
            nt_text = ref["nt_text"]

            # Calculate verbal agreement with LXX and MT
            lxx_agreement = self._verbal_agreement(nt_text, lxx_text) if lxx_text else 0.0
            mt_agreement = self._verbal_agreement(nt_text, mt_text) if mt_text else 0.0

            quotations.append(NTQuotation(
                nt_verse=ref["nt_verse"],
                nt_text_greek=nt_text,
                quote_type=ref.get("quote_type", "quotation"),
                follows_lxx=lxx_agreement > mt_agreement + 0.1,
                follows_mt=mt_agreement > lxx_agreement + 0.1,
                verbal_agreement_lxx=lxx_agreement,
                verbal_agreement_mt=mt_agreement,
                theological_significance=self._assess_theological_significance(
                    verse_id, ref["nt_verse"], lxx_agreement > mt_agreement
                )
            ))

        return quotations

    def _verbal_agreement(self, text1: str, text2: str) -> float:
        """Calculate verbal agreement between two texts."""
        if not text1 or not text2:
            return 0.0

        # Normalize texts
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()

        # Use sequence matcher
        matcher = SequenceMatcher(None, t1, t2)
        return matcher.ratio()

    def _assess_theological_significance(
        self,
        ot_verse: str,
        nt_verse: str,
        prefers_lxx: bool
    ) -> str:
        """Assess theological significance of NT's text preference."""
        significance_notes = []

        if prefers_lxx:
            significance_notes.append("NT follows LXX reading")

            # Check for known significant divergences
            if ot_verse in self.KNOWN_CHRISTOLOGICAL_VERSES:
                category = self.KNOWN_CHRISTOLOGICAL_VERSES[ot_verse]
                significance_notes.append(
                    f"LXX reading supports {category.value} interpretation"
                )

        else:
            significance_notes.append("NT follows MT or independent reading")

        return "; ".join(significance_notes)

    async def _gather_patristic(self, verse_id: str) -> List[PatristicWitness]:
        """Gather patristic interpretations of this verse."""
        witnesses = []

        # Query patristic database
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

        # Determine primary category
        primary_category = None
        primary_insight = "No significant Christological divergence detected"

        if christological_divs:
            # Take highest-scoring
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

        # Overall significance
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

    async def _get_cached(self, verse_id: str) -> Optional[LXXAnalysisResult]:
        """Retrieve cached analysis result."""
        if not self.config.cache_enabled:
            return None

        cache_key = f"lxx_analysis:{verse_id}"
        cached = await self.redis.get(cache_key)

        if cached:
            result = LXXAnalysisResult(**cached)
            result.cache_hit = True
            return result

        return None

    async def _cache_result(self, result: LXXAnalysisResult):
        """Cache analysis result."""
        if not self.config.cache_enabled:
            return

        cache_key = f"lxx_analysis:{result.verse_id}"
        await self.redis.set(
            cache_key,
            result.__dict__,
            ex=self.config.cache_ttl_seconds
        )

    async def _store_to_neo4j(self, result: LXXAnalysisResult):
        """Persist analysis result to Neo4j."""
        # Store each divergence
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
        progress_callback: Optional[callable] = None
    ) -> List[LXXAnalysisResult]:
        """Process multiple verses with parallelism."""
        results = []

        if self.config.parallel_analysis:
            # Process in batches
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
            # Sequential processing
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
        # Get all verses in book
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

        # Batch process
        all_results = await self.batch_extract(verse_ids)

        # Filter to significant Christological divergences
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
```

---

## Part 4: Integration Points

### Neo4j Schema Extension

```cypher
// LXX Divergence nodes and relationships
CREATE (d:LXXDivergence {
    divergence_id: $id,
    verse_id: $verse_id,
    divergence_type: $type,
    christological_category: $category,
    mt_text: $mt_text,
    lxx_text: $lxx_text,
    composite_score: $score
})

MATCH (v:Verse {id: $verse_id})
CREATE (v)-[:HAS_LXX_DIVERGENCE {
    significance: $score,
    category: $category
}]->(d)

// Query high-significance Christological divergences
MATCH (v:Verse)-[r:HAS_LXX_DIVERGENCE]->(d:LXXDivergence)
WHERE d.christological_category IS NOT NULL
  AND d.composite_score >= 0.7
RETURN v.id, d.christological_category, d.composite_score
ORDER BY d.composite_score DESC
```

### Pipeline Integration

```python
# In ml/inference/pipeline.py
async def enrich_with_lxx_analysis(
    self,
    cross_ref: CrossReferenceCandidate
) -> CrossReferenceCandidate:
    """Add LXX Christological data to cross-references."""

    if cross_ref.source_verse.startswith(("GEN", "EXO", "LEV", "NUM", "DEU",
                                           "PSA", "ISA", "JER", "EZE", "DAN")):
        lxx_result = await self.lxx_extractor.extract_christological_content(
            cross_ref.source_verse
        )

        if lxx_result.christological_divergence_count > 0:
            cross_ref.features["lxx_christological"] = True
            cross_ref.features["lxx_category"] = lxx_result.christological_category.value
            cross_ref.features["lxx_significance"] = lxx_result.overall_significance

            # Boost confidence for prophetic connections
            if cross_ref.connection_type == "prophetic":
                cross_ref.confidence *= (1.0 + lxx_result.overall_significance * 0.2)

    return cross_ref
```

---

## Part 5: Testing Specification

```python
class TestLXXExtractor:
    """Tests for LXX Christological Extractor."""

    @pytest.mark.asyncio
    async def test_isaiah_7_14_parthenos(self, extractor):
        """ISA.7.14: almah → parthenos divergence."""
        result = await extractor.extract_christological_content("ISA.7.14")

        assert result.christological_divergence_count >= 1
        assert result.christological_category == ChristologicalCategory.VIRGIN_BIRTH

        # Check divergence details
        div = result.divergences[0]
        assert "παρθένος" in div.lxx_text_greek or "parthenos" in div.lxx_text_transliterated.lower()
        assert div.composite_score >= 0.85

        # NT support
        nt_refs = [q.nt_verse for q in div.nt_quotations]
        assert "MAT.1.23" in nt_refs

    @pytest.mark.asyncio
    async def test_psalm_40_body_prepared(self, extractor):
        """PSA.40.6: ears → body divergence with Psalm numbering."""
        result = await extractor.extract_christological_content("PSA.40.6")

        assert result.lxx_verse_id == "PSA.39.6"  # Numbering conversion
        assert result.christological_category == ChristologicalCategory.INCARNATION

        # Check HEB.10.5 quotation
        has_hebrews_quote = any(
            q.nt_verse == "HEB.10.5" and q.follows_lxx
            for div in result.divergences
            for q in div.nt_quotations
        )
        assert has_hebrews_quote

    @pytest.mark.asyncio
    async def test_psalm_22_16_pierced(self, extractor):
        """PSA.22.16: lion → pierced with DSS support."""
        result = await extractor.extract_christological_content("PSA.22.16")

        assert result.christological_category == ChristologicalCategory.PASSION

        # Check for DSS support
        div = result.divergences[0]
        dss_witnesses = [w for w in div.manuscript_witnesses if "Q" in w.manuscript_id]
        assert len(dss_witnesses) >= 1

        # DSS should support LXX
        if dss_witnesses:
            assert dss_witnesses[0].supports_lxx

    @pytest.mark.asyncio
    async def test_no_christological_divergence(self, extractor):
        """GEN.1.1: No significant Christological divergence."""
        result = await extractor.extract_christological_content("GEN.1.1")

        assert result.christological_divergence_count == 0
        assert result.christological_category is None

    @pytest.mark.asyncio
    async def test_manuscript_priority(self, extractor):
        """Verify oldest manuscript is prioritized."""
        result = await extractor.extract_christological_content("ISA.7.14")

        for div in result.divergences:
            if div.oldest_witness:
                # Oldest should have highest reliability
                other_scores = [w.reliability_score for w in div.manuscript_witnesses if w != div.oldest_witness]
                if other_scores:
                    assert div.oldest_witness.reliability_score >= max(other_scores)
```

---

## Part 6: Configuration

```python
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
    christological_threshold: float = 0.5 # Minimum for Christological significance

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
```

---

## Part 7: Success Criteria

### Functional Requirements
- [ ] Correctly aligns MT and LXX texts
- [ ] Detects lexical divergences with semantic analysis
- [ ] Classifies Christological categories accurately
- [ ] Handles Psalm numbering conversion
- [ ] Gathers manuscript evidence with priority ranking
- [ ] Finds NT quotations and determines LXX vs MT preference

### Theological Accuracy
- [ ] ISA.7.14: παρθένος detected, VIRGIN_BIRTH, score ≥ 0.90
- [ ] PSA.40.6: σῶμα detected, INCARNATION, HEB.10.5 support
- [ ] PSA.22.16: ὤρυξαν detected, PASSION, DSS support noted
- [ ] GEN.3.15: αὐτός detected, SOTERIOLOGICAL

### Performance
- [ ] Single verse: < 1s cached, < 3s cold
- [ ] Book scan: < 2 minutes
- [ ] Manuscript lookup: < 500ms

---

## Part 8: Dependencies

### Depends On
- SESSION 03: OmniContextualResolver (word meaning resolution)
- Corpus integrations (Text-Fabric, LXX sources)

### Depended On By
- SESSION 06: Fractal Typology Engine (LXX readings for types)
- SESSION 07: Prophetic Necessity Prover (LXX as evidence)
- SESSION 11: Pipeline Integration

---

## Session Completion Checklist

```markdown
- [ ] `ml/engines/lxx_extractor.py` implemented
- [ ] `integrations/lxx_corpus.py` created
- [ ] Verse numbering conversion (Psalms)
- [ ] MT-LXX alignment functional
- [ ] Divergence detection and classification
- [ ] Manuscript priority ranking
- [ ] NT quotation detection
- [ ] Neo4j schema extension
- [ ] ISA.7.14 test passing
- [ ] PSA.40.6 test passing
- [ ] Configuration added
```

---

## Part 9: LXX Corpus Client

### File: `integrations/lxx_corpus.py`

```python
"""
LXX Corpus Client for accessing Septuagint texts.

Integrates with:
- MASTER_LINGUISTIC_CORPUS/RESTRUCTURED_CORPUS/Output/master_corpus.db
- Rahlfs-Hanhart LXX morphologically tagged text
- Swete's Old Testament in Greek
"""
from __future__ import annotations
import sqlite3
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import aiofiles
import aiosqlite


@dataclass
class LXXWord:
    """A word from the LXX text."""
    position: int
    text: str                      # Greek text (Unicode)
    lemma: str                     # Dictionary form
    morphology: Dict[str, str]     # POS, case, number, gender, tense, etc.
    gloss: str                     # English translation
    translit: str                  # Transliteration
    semantic_domains: List[str]
    strongs: Optional[str]         # Strong's number if available


@dataclass
class LXXVerse:
    """A verse from the LXX."""
    verse_id: str                  # Standard form: BOOK.CHAPTER.VERSE
    lxx_verse_id: str              # LXX-specific numbering
    text_full: str                 # Full verse text
    words: List[LXXWord]
    variant_readings: List[Dict]   # Manuscript variants
    critical_apparatus: str        # Notes from critical edition


class LXXCorpusClient:
    """
    Client for accessing Septuagint corpus data.

    Primary source: MASTER_LINGUISTIC_CORPUS SQLite database
    Fallback: JSON corpus files
    """

    # Book name mappings (LXX sometimes uses different names)
    LXX_BOOK_NAMES = {
        "GEN": "Genesis",
        "EXO": "Exodus",
        "LEV": "Leviticus",
        "NUM": "Numbers",
        "DEU": "Deuteronomy",
        "JOS": "Joshua",
        "JDG": "Judges",
        "RUT": "Ruth",
        "1SA": "1Kingdoms",      # LXX naming
        "2SA": "2Kingdoms",
        "1KI": "3Kingdoms",
        "2KI": "4Kingdoms",
        "1CH": "1Chronicles",
        "2CH": "2Chronicles",
        "EZR": "Ezra",
        "NEH": "Nehemiah",
        "EST": "Esther",
        "JOB": "Job",
        "PSA": "Psalms",
        "PRO": "Proverbs",
        "ECC": "Ecclesiastes",
        "SNG": "SongOfSongs",
        "ISA": "Isaiah",
        "JER": "Jeremiah",
        "LAM": "Lamentations",
        "EZE": "Ezekiel",
        "DAN": "Daniel",
        "HOS": "Hosea",
        "JOL": "Joel",
        "AMO": "Amos",
        "OBA": "Obadiah",
        "JON": "Jonah",
        "MIC": "Micah",
        "NAH": "Nahum",
        "HAB": "Habakkuk",
        "ZEP": "Zephaniah",
        "HAG": "Haggai",
        "ZEC": "Zechariah",
        "MAL": "Malachi",
    }

    def __init__(
        self,
        corpus_db_path: str,
        json_fallback_path: Optional[str] = None
    ):
        self.db_path = Path(corpus_db_path)
        self.fallback_path = Path(json_fallback_path) if json_fallback_path else None
        self.logger = logging.getLogger(__name__)
        self._connection: Optional[aiosqlite.Connection] = None

    async def connect(self):
        """Initialize database connection."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"LXX corpus not found: {self.db_path}")

        self._connection = await aiosqlite.connect(str(self.db_path))
        self._connection.row_factory = aiosqlite.Row
        self.logger.info(f"Connected to LXX corpus: {self.db_path}")

    async def close(self):
        """Close database connection."""
        if self._connection:
            await self._connection.close()

    async def get_verse(self, verse_id: str) -> Dict[str, Any]:
        """
        Get verse data including words and morphology.

        Args:
            verse_id: Standard verse ID (e.g., "GEN.1.1")

        Returns:
            Dict with verse text, words, and metadata
        """
        if not self._connection:
            await self.connect()

        # Parse verse ID
        parts = verse_id.split(".")
        book = parts[0]
        chapter = int(parts[1]) if len(parts) > 1 else 1
        verse = int(parts[2]) if len(parts) > 2 else 1

        # Query verse text
        query = """
            SELECT
                v.id, v.book, v.chapter, v.verse, v.text,
                v.lxx_book, v.lxx_chapter, v.lxx_verse
            FROM lxx_verses v
            WHERE v.book = ? AND v.chapter = ? AND v.verse = ?
        """
        async with self._connection.execute(query, (book, chapter, verse)) as cursor:
            row = await cursor.fetchone()

        if not row:
            self.logger.warning(f"Verse not found: {verse_id}")
            return {"verse_id": verse_id, "text": "", "words": []}

        # Query words for this verse
        word_query = """
            SELECT
                w.position, w.text, w.lemma, w.morph_code,
                w.gloss, w.translit, w.strongs,
                w.semantic_domain
            FROM lxx_words w
            WHERE w.verse_id = ?
            ORDER BY w.position
        """
        async with self._connection.execute(word_query, (row["id"],)) as cursor:
            word_rows = await cursor.fetchall()

        words = []
        for w in word_rows:
            words.append({
                "position": w["position"],
                "text": w["text"],
                "lemma": w["lemma"],
                "morphology": self._parse_morph_code(w["morph_code"]),
                "gloss": w["gloss"],
                "translit": w["translit"],
                "strongs": w["strongs"],
                "semantic_domains": w["semantic_domain"].split(",") if w["semantic_domain"] else []
            })

        return {
            "verse_id": verse_id,
            "lxx_verse_id": f"{row['lxx_book']}.{row['lxx_chapter']}.{row['lxx_verse']}",
            "text": row["text"],
            "words": words
        }

    async def get_verse_text(self, verse_id: str) -> Optional[str]:
        """Get just the verse text without word-level data."""
        verse_data = await self.get_verse(verse_id)
        return verse_data.get("text")

    async def get_chapter(self, book: str, chapter: int) -> List[Dict]:
        """Get all verses in a chapter."""
        if not self._connection:
            await self.connect()

        query = """
            SELECT v.verse, v.text
            FROM lxx_verses v
            WHERE v.book = ? AND v.chapter = ?
            ORDER BY v.verse
        """
        async with self._connection.execute(query, (book, chapter)) as cursor:
            rows = await cursor.fetchall()

        return [
            {"verse_id": f"{book}.{chapter}.{r['verse']}", "text": r["text"]}
            for r in rows
        ]

    async def search_greek(
        self,
        pattern: str,
        book: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Search for Greek text patterns."""
        if not self._connection:
            await self.connect()

        if book:
            query = """
                SELECT v.book, v.chapter, v.verse, v.text
                FROM lxx_verses v
                WHERE v.text LIKE ? AND v.book = ?
                LIMIT ?
            """
            params = (f"%{pattern}%", book, limit)
        else:
            query = """
                SELECT v.book, v.chapter, v.verse, v.text
                FROM lxx_verses v
                WHERE v.text LIKE ?
                LIMIT ?
            """
            params = (f"%{pattern}%", limit)

        async with self._connection.execute(query, params) as cursor:
            rows = await cursor.fetchall()

        return [
            {
                "verse_id": f"{r['book']}.{r['chapter']}.{r['verse']}",
                "text": r["text"]
            }
            for r in rows
        ]

    async def get_word_occurrences(
        self,
        lemma: str,
        limit: int = 500
    ) -> List[Dict]:
        """Find all occurrences of a Greek lemma."""
        if not self._connection:
            await self.connect()

        query = """
            SELECT
                v.book, v.chapter, v.verse, w.text, w.morph_code
            FROM lxx_words w
            JOIN lxx_verses v ON w.verse_id = v.id
            WHERE w.lemma = ?
            LIMIT ?
        """
        async with self._connection.execute(query, (lemma, limit)) as cursor:
            rows = await cursor.fetchall()

        return [
            {
                "verse_id": f"{r['book']}.{r['chapter']}.{r['verse']}",
                "form": r["text"],
                "morphology": self._parse_morph_code(r["morph_code"])
            }
            for r in rows
        ]

    def _parse_morph_code(self, code: str) -> Dict[str, str]:
        """Parse morphological code into structured dict."""
        if not code:
            return {}

        # Standard Greek morphology code parsing
        # Format varies by corpus, this handles common patterns
        morph = {}

        # Common positional encoding (e.g., Robinson's codes)
        pos_map = {
            'N': 'noun', 'V': 'verb', 'A': 'adjective', 'D': 'adverb',
            'P': 'preposition', 'C': 'conjunction', 'R': 'pronoun',
            'T': 'article', 'X': 'particle', 'I': 'interjection'
        }

        case_map = {'N': 'nominative', 'G': 'genitive', 'D': 'dative', 'A': 'accusative', 'V': 'vocative'}
        number_map = {'S': 'singular', 'P': 'plural', 'D': 'dual'}
        gender_map = {'M': 'masculine', 'F': 'feminine', 'N': 'neuter'}
        tense_map = {'P': 'present', 'I': 'imperfect', 'F': 'future', 'A': 'aorist', 'X': 'perfect', 'Y': 'pluperfect'}
        voice_map = {'A': 'active', 'M': 'middle', 'P': 'passive'}
        mood_map = {'I': 'indicative', 'S': 'subjunctive', 'O': 'optative', 'M': 'imperative', 'N': 'infinitive', 'P': 'participle'}

        if len(code) >= 1 and code[0] in pos_map:
            morph['pos'] = pos_map[code[0]]

        # Parse based on POS
        if morph.get('pos') == 'verb' and len(code) >= 6:
            if code[1] in tense_map:
                morph['tense'] = tense_map[code[1]]
            if code[2] in voice_map:
                morph['voice'] = voice_map[code[2]]
            if code[3] in mood_map:
                morph['mood'] = mood_map[code[3]]
            if code[4] in number_map:
                morph['number'] = number_map[code[4]]
            if code[5] in {'1', '2', '3'}:
                morph['person'] = code[5]

        elif morph.get('pos') in ('noun', 'adjective', 'article', 'pronoun') and len(code) >= 4:
            if code[1] in case_map:
                morph['case'] = case_map[code[1]]
            if code[2] in number_map:
                morph['number'] = number_map[code[2]]
            if code[3] in gender_map:
                morph['gender'] = gender_map[code[3]]

        return morph

    async def get_variants(self, verse_id: str) -> List[Dict]:
        """Get manuscript variants for a verse."""
        if not self._connection:
            await self.connect()

        parts = verse_id.split(".")
        book, chapter, verse = parts[0], int(parts[1]), int(parts[2])

        query = """
            SELECT
                var.manuscript, var.reading, var.notes,
                m.century, m.type
            FROM lxx_variants var
            JOIN lxx_verses v ON var.verse_id = v.id
            LEFT JOIN manuscripts m ON var.manuscript = m.id
            WHERE v.book = ? AND v.chapter = ? AND v.verse = ?
        """
        async with self._connection.execute(query, (book, chapter, verse)) as cursor:
            rows = await cursor.fetchall()

        return [
            {
                "manuscript": r["manuscript"],
                "reading": r["reading"],
                "notes": r["notes"],
                "century": r["century"],
                "type": r["type"]
            }
            for r in rows
        ]
```

---

## Part 10: CLI Commands

### File: `cli/lxx_commands.py`

```python
"""CLI commands for LXX Christological analysis."""
import asyncio
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from ml.engines.lxx_extractor import (
    LXXChristologicalExtractor,
    ChristologicalCategory
)
from config import get_config
from db.neo4j_client import Neo4jClient
from db.redis_client import RedisClient
from integrations.lxx_corpus import LXXCorpusClient
from integrations.text_fabric import TextFabricClient

app = typer.Typer(help="LXX Christological analysis commands")
console = Console()


def get_extractor() -> LXXChristologicalExtractor:
    """Create configured extractor instance."""
    config = get_config()
    return LXXChristologicalExtractor(
        lxx_client=LXXCorpusClient(config.lxx_corpus_path),
        mt_client=TextFabricClient(config.text_fabric_path),
        neo4j=Neo4jClient(config.neo4j_uri),
        redis=RedisClient(config.redis_url),
        config=config.lxx_extractor
    )


@app.command("analyze")
def analyze_verse(
    verse: str = typer.Argument(..., help="Verse ID (e.g., ISA.7.14)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force recompute"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output")
):
    """Analyze a single verse for LXX Christological content."""

    async def _run():
        extractor = get_extractor()
        await extractor.lxx.connect()

        with console.status(f"Analyzing {verse}..."):
            result = await extractor.extract_christological_content(verse, force)

        # Display results
        if result.christological_divergence_count > 0:
            console.print(f"\n[bold green]✓ Christological Content Found[/]")
            console.print(f"  Category: [cyan]{result.christological_category.value}[/]")
            console.print(f"  Insight: {result.primary_christological_insight}")
            console.print(f"  Significance: [yellow]{result.overall_significance:.2f}[/]")

            if verbose:
                table = Table(title="Divergences")
                table.add_column("Type")
                table.add_column("MT (Hebrew)")
                table.add_column("LXX (Greek)")
                table.add_column("Score")

                for div in result.divergences:
                    table.add_row(
                        div.divergence_type.value,
                        div.mt_text_hebrew,
                        div.lxx_text_greek,
                        f"{div.composite_score:.2f}"
                    )
                console.print(table)

                # NT Quotations
                if any(d.nt_quotations for d in result.divergences):
                    console.print("\n[bold]NT Quotations:[/]")
                    for div in result.divergences:
                        for q in div.nt_quotations:
                            pref = "LXX" if q.follows_lxx else "MT" if q.follows_mt else "neither"
                            console.print(f"  • {q.nt_verse} follows {pref}")
        else:
            console.print(f"\n[dim]No significant Christological divergence in {verse}[/]")

        await extractor.lxx.close()

    asyncio.run(_run())


@app.command("scan")
def scan_book(
    book: str = typer.Argument(..., help="Book code (e.g., ISA, PSA)"),
    min_score: float = typer.Option(0.5, "--min-score", "-m", help="Minimum significance"),
    output: str = typer.Option(None, "--output", "-o", help="Output JSON file")
):
    """Scan an entire book for Christological LXX divergences."""

    async def _run():
        extractor = get_extractor()
        await extractor.lxx.connect()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Scanning {book}...", total=None)

            results = await extractor.scan_book_for_christological(book, min_score)

        console.print(f"\n[bold]Found {len(results)} significant verses in {book}[/]")

        table = Table(title=f"Christological LXX Divergences in {book}")
        table.add_column("Verse")
        table.add_column("Category")
        table.add_column("Score")
        table.add_column("NT Support")

        for r in results[:20]:  # Show top 20
            table.add_row(
                r.verse_id,
                r.christological_category.value if r.christological_category else "-",
                f"{r.overall_significance:.2f}",
                "Yes" if r.nt_support_strength > 0.5 else "No"
            )

        console.print(table)

        if output:
            import json
            output_data = [
                {
                    "verse_id": r.verse_id,
                    "category": r.christological_category.value if r.christological_category else None,
                    "significance": r.overall_significance,
                    "nt_support": r.nt_support_strength,
                    "insight": r.primary_christological_insight
                }
                for r in results
            ]
            with open(output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            console.print(f"\n[green]Results saved to {output}[/]")

        await extractor.lxx.close()

    asyncio.run(_run())


@app.command("known")
def list_known():
    """List all known Christological LXX divergences."""
    table = Table(title="Known Christological LXX Divergences")
    table.add_column("Verse", style="cyan")
    table.add_column("Category", style="green")
    table.add_column("Description")

    descriptions = {
        "ISA.7.14": "almah → parthenos (virgin)",
        "PSA.40.6": "ears → body prepared",
        "GEN.3.15": "it/she → he (seed promise)",
        "PSA.22.16": "like a lion → they pierced",
        "ISA.53.8": "Generation declaration",
        "PSA.16.10": "Holy One not see corruption",
        "ISA.9.6": "Wonderful Counselor titles",
        "MIC.5.2": "Origin from eternity",
        "ZEC.12.10": "Look on pierced one",
        "PSA.110.1": "Lord says to my Lord",
        "DAN.7.13": "Son of Man coming",
        "ISA.61.1": "Anointed to preach",
        "MAL.3.1": "Messenger of the covenant",
        "PSA.2.7": "You are my Son",
        "ISA.11.1": "Branch from Jesse",
    }

    for verse, category in LXXChristologicalExtractor.KNOWN_CHRISTOLOGICAL_VERSES.items():
        table.add_row(
            verse,
            category.value,
            descriptions.get(verse, "")
        )

    console.print(table)


@app.command("compare")
def compare_texts(
    verse: str = typer.Argument(..., help="Verse ID to compare")
):
    """Display MT and LXX texts side by side."""

    async def _run():
        extractor = get_extractor()
        await extractor.lxx.connect()
        await extractor.mt.connect()

        mt_data = await extractor.mt.get_verse(verse)
        lxx_data = await extractor.lxx.get_verse(verse)

        console.print(f"\n[bold]Comparison: {verse}[/]\n")

        console.print("[cyan]Masoretic Text (Hebrew):[/]")
        console.print(f"  {mt_data.get('text', 'Not found')}")

        console.print("\n[green]Septuagint (Greek):[/]")
        console.print(f"  {lxx_data.get('text', 'Not found')}")

        # Word-by-word comparison
        if mt_data.get("words") and lxx_data.get("words"):
            console.print("\n[bold]Word Comparison:[/]")
            table = Table()
            table.add_column("MT Word")
            table.add_column("MT Gloss")
            table.add_column("LXX Word")
            table.add_column("LXX Gloss")

            mt_words = mt_data["words"]
            lxx_words = lxx_data["words"]

            for i in range(max(len(mt_words), len(lxx_words))):
                mt_w = mt_words[i] if i < len(mt_words) else {}
                lxx_w = lxx_words[i] if i < len(lxx_words) else {}
                table.add_row(
                    mt_w.get("text", "-"),
                    mt_w.get("gloss", "-"),
                    lxx_w.get("text", "-"),
                    lxx_w.get("gloss", "-")
                )

            console.print(table)

        await extractor.lxx.close()
        await extractor.mt.close()

    asyncio.run(_run())


# Register with main CLI
def register_commands(main_app: typer.Typer):
    """Register LXX commands with main CLI."""
    main_app.add_typer(app, name="lxx", help="LXX Christological analysis")
```

### CLI Usage Examples

```bash
# Analyze specific verse
biblos lxx analyze ISA.7.14 --verbose

# Scan entire book
biblos lxx scan ISA --min-score 0.6 --output isaiah_lxx.json

# List known Christological verses
biblos lxx known

# Compare MT and LXX texts
biblos lxx compare PSA.22.16
```

---

## Part 11: Additional Christological Markers

### Extended Marker Catalog

```python
# In lxx_extractor.py - extended CHRISTOLOGICAL_MARKERS

CHRISTOLOGICAL_MARKERS_EXTENDED = {
    # Virgin Birth / Incarnation
    "παρθένος": (ChristologicalCategory.VIRGIN_BIRTH, 0.95),
    "σῶμα κατηρτίσω": (ChristologicalCategory.INCARNATION, 0.92),
    "ἐν γαστρὶ ἕξει": (ChristologicalCategory.VIRGIN_BIRTH, 0.85),

    # Passion / Suffering
    "ὤρυξαν": (ChristologicalCategory.PASSION, 0.92),
    "ἐξεκέντησαν": (ChristologicalCategory.PASSION, 0.90),
    "ἐταπεινώθη": (ChristologicalCategory.PASSION, 0.75),
    "ἀμνὸς": (ChristologicalCategory.SACRIFICIAL, 0.80),
    "τραυματίας": (ChristologicalCategory.PASSION, 0.78),

    # Divine Nature
    "θεὸς ἰσχυρός": (ChristologicalCategory.DIVINE_NATURE, 0.90),
    "πατὴρ αἰῶνος": (ChristologicalCategory.DIVINE_NATURE, 0.88),
    "κύριος κύριος": (ChristologicalCategory.DIVINE_NATURE, 0.85),
    "ἄρχων εἰρήνης": (ChristologicalCategory.ROYAL_DAVIDIC, 0.82),

    # Messianic Titles
    "χριστός": (ChristologicalCategory.MESSIANIC_TITLE, 0.88),
    "ὁ ἐρχόμενος": (ChristologicalCategory.MESSIANIC_TITLE, 0.80),
    "υἱὸς ἀνθρώπου": (ChristologicalCategory.MESSIANIC_TITLE, 0.85),
    "δοῦλος κυρίου": (ChristologicalCategory.MESSIANIC_TITLE, 0.75),

    # Resurrection
    "ἀνάστασις": (ChristologicalCategory.RESURRECTION, 0.88),
    "οὐκ ἐγκαταλείψεις": (ChristologicalCategory.RESURRECTION, 0.82),
    "διαφθοράν": (ChristologicalCategory.RESURRECTION, 0.80),

    # Prophetic / Soteriological
    "σωτηρία": (ChristologicalCategory.SOTERIOLOGICAL, 0.75),
    "λύτρωσις": (ChristologicalCategory.SOTERIOLOGICAL, 0.78),
    "ἄφεσις": (ChristologicalCategory.SOTERIOLOGICAL, 0.72),
    "ἐξιλασμός": (ChristologicalCategory.SACRIFICIAL, 0.80),

    # Theophanic
    "δόξα κυρίου": (ChristologicalCategory.THEOPHANIC, 0.82),
    "ἄγγελος κυρίου": (ChristologicalCategory.THEOPHANIC, 0.78),
    "λόγος κυρίου": (ChristologicalCategory.THEOPHANIC, 0.80),

    # Royal / Davidic
    "βασιλεύς": (ChristologicalCategory.ROYAL_DAVIDIC, 0.70),
    "θρόνος Δαυίδ": (ChristologicalCategory.ROYAL_DAVIDIC, 0.85),
    "ῥάβδος": (ChristologicalCategory.ROYAL_DAVIDIC, 0.72),
    "ἀνατολή": (ChristologicalCategory.ROYAL_DAVIDIC, 0.75),  # Branch/rising

    # Priestly
    "ἱερεύς": (ChristologicalCategory.PRIESTLY, 0.70),
    "Μελχισέδεκ": (ChristologicalCategory.PRIESTLY, 0.90),
    "θυσία": (ChristologicalCategory.SACRIFICIAL, 0.72),
}

# Known verses with DSS support for LXX reading
DSS_SUPPORTS_LXX = {
    "PSA.22.16": "4QPsᶠ reads כארו (pierced), supporting LXX ὤρυξαν",
    "ISA.53.11": "1QIsaᵃ supports 'he will see light' (φῶς)",
    "DEU.32.43": "4QDeutᵍ has fuller text matching LXX",
    "PSA.145.13": "11QPsᵃ includes verse missing in MT",
}

# NT passages preferring LXX over MT
NT_LXX_PREFERENCES = {
    "ISA.7.14": ["MAT.1.23"],
    "PSA.40.6": ["HEB.10.5"],
    "PSA.22.16": ["JHN.19.37", "REV.1.7"],
    "ISA.53.4": ["MAT.8.17"],
    "ISA.53.12": ["MAR.15.28", "LUK.22.37"],
    "PSA.16.10": ["ACT.2.27", "ACT.13.35"],
    "ISA.61.1": ["LUK.4.18"],
    "PSA.110.1": ["MAT.22.44", "MAR.12.36", "LUK.20.42", "ACT.2.34", "HEB.1.13"],
    "DAN.7.13": ["MAT.24.30", "MAT.26.64", "MAR.13.26", "MAR.14.62"],
    "ZEC.12.10": ["JHN.19.37", "REV.1.7"],
    "MIC.5.2": ["MAT.2.6"],
    "ISA.9.6": ["LUK.1.32-33"],
    "MAL.3.1": ["MAT.11.10", "MAR.1.2", "LUK.7.27"],
}
```

---

## Session Completion Checklist

```markdown
- [ ] `ml/engines/lxx_extractor.py` implemented
  - [ ] LXXChristologicalExtractor class complete
  - [ ] All helper methods implemented
  - [ ] ManuscriptPriority enum functional
  - [ ] Divergence detection working
  - [ ] Christological classification
  - [ ] Composite scoring

- [ ] `integrations/lxx_corpus.py` created
  - [ ] LXXCorpusClient class
  - [ ] Database connection handling
  - [ ] Morphology parsing
  - [ ] Variant retrieval

- [ ] `cli/lxx_commands.py` created
  - [ ] analyze command
  - [ ] scan command
  - [ ] known command
  - [ ] compare command

- [ ] Core functionality
  - [ ] Verse numbering conversion (Psalms)
  - [ ] MT-LXX alignment functional
  - [ ] Divergence detection and classification
  - [ ] Manuscript priority ranking
  - [ ] NT quotation detection
  - [ ] Patristic witness gathering

- [ ] Data integration
  - [ ] Neo4j schema extension
  - [ ] Redis caching
  - [ ] master_corpus.db integration

- [ ] Tests passing
  - [ ] ISA.7.14 test passing
  - [ ] PSA.40.6 test passing
  - [ ] PSA.22.16 test passing
  - [ ] Manuscript priority test
  - [ ] No false positives test (GEN.1.1)

- [ ] Configuration added
  - [ ] LXXExtractorConfig in config.py
  - [ ] Environment variables documented
```

**Next Session**: SESSION 06: Hyper-Fractal Typology Engine
