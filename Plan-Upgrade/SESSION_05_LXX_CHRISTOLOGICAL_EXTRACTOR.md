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
from enum import Enum
from typing import Dict, List, Optional, Tuple
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

**Next Session**: SESSION 06: Hyper-Fractal Typology Engine
