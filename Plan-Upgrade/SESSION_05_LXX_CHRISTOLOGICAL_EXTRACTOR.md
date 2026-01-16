# SESSION 05: LXX CHRISTOLOGICAL EXTRACTOR

## Session Overview

**Objective**: Implement the `LXXChristologicalExtractor` engine that discovers Christological content uniquely present in the Septuagint but absent from or muted in the Masoretic Text. This is the third of the Five Impossible Oracles.

**Estimated Duration**: 1 Claude session (75-90 minutes of focused implementation)

**Prerequisites**:
- Access to parallel LXX and MT corpus data
- Access to oldest manuscript transcriptions (user has sources)
- Understanding of Greek/Hebrew textual criticism basics
- Familiarity with messianic prophecy traditions
- Access to NT quotation apparatus

**Critical Principle**: **Oldest transcriptions are most valid**. When manuscript variants exist, prioritize the earliest attested readings. The user has access to these primary sources.

---

## Part 1: Understanding the Oracle Concept

### Core Capability
The Septuagint (LXX) was translated by Jewish scholars 200-300 years before Christ. Yet it often renders Hebrew text in ways that appear **prophetically Christological** - readings that the Church Fathers consistently noted pointed to Christ more clearly than the later Masoretic Text.

This engine detects and catalogs these divergences systematically.

### Why This Is Significant

1. **Pre-Christian Witness**: LXX translators had no Christian agenda, yet their translation choices often favor Christological readings
2. **NT Preference**: New Testament authors quote the LXX over 300 times, often choosing specifically Christological LXX readings
3. **Patristic Foundation**: Church Fathers heavily relied on LXX as inspired text
4. **Apologetic Value**: LXX divergences that favor Christ predate Christianity, eliminating "Christian tampering" objections
5. **Oldest Transcriptions Priority**: The earliest manuscript evidence carries the greatest weight; later standardizations (like the Masoretic standardization) may reflect theological editing

### Manuscript Priority Hierarchy

The engine should respect this textual authority ranking:
1. **Dead Sea Scrolls (DSS)** - 3rd century BCE to 1st century CE (oldest Hebrew)
2. **Oldest LXX Manuscripts** - Codex Vaticanus, Codex Sinaiticus (4th century CE)
3. **Hexaplaric LXX** - Origen's critical edition preserving older readings
4. **Masoretic Text** - 7th-10th century CE standardization (useful but later)
5. **Vulgate/Peshitta** - For confirmation of early readings

### Canonical Examples

#### Example 1: Isaiah 7:14 - παρθένος vs עַלְמָה
```
MT (Hebrew): הָעַלְמָה (ha'almah) - "the young woman"
LXX (Greek): ἡ παρθένος (hē parthenos) - "the virgin"

Christological Significance:
- LXX specifies "virgin" (biological state)
- NT quotes LXX in MAT.1.23
- Pre-Christian translation supports virgin birth doctrine
```

#### Example 2: Psalm 40:6 (LXX 39:7) - Body Prepared
```
MT: אָזְנַיִם כָּרִיתָ לִּי (ears you have dug/opened for me)
LXX: σῶμα δὲ κατηρτίσω μοι (a body you have prepared for me)

Christological Significance:
- Hebrews 10:5 quotes LXX reading
- Points to Incarnation (body prepared for sacrifice)
- MT reading about "opened ears" lacks this Christological depth
```

#### Example 3: Genesis 3:15 - αὐτός vs הוּא
```
MT: הוּא (hu) - "it" (neuter, referring to seed/offspring collectively)
LXX: αὐτός (autos) - "he" (masculine, specific person)

Christological Significance:
- LXX makes Protoevangelium specifically about a single male individual
- Points to Christ as the one who crushes the serpent
- MT is ambiguous about collective vs individual
```

#### Example 4: Psalm 22:16 (LXX 21:17) - כָּאֲרִי vs ὤρυξαν
```
MT: כָּאֲרִי (ka'ari) - "like a lion" (my hands and feet)
LXX: ὤρυξαν (ōryxan) - "they pierced" (my hands and feet)

Christological Significance:
- LXX describes crucifixion detail
- DSS supports "pierced" reading (כארו)
- MT reading awkward grammatically
```

---

## Part 2: File Creation Specification

### File: `ml/engines/lxx_extractor.py`

**Location**: `ml/engines/` (directory created in Session 03)

**Dependencies to Import**:
- `dataclasses` for result schemas
- `typing` for type hints
- `logging` for analysis logging
- `difflib` for textual comparison
- Access to LXX corpus (Rahlfs, Swete)
- Access to MT corpus (BHSA, Macula-Hebrew)
- Access to NT quotation database
- Greek/Hebrew morphology libraries

**Classes to Define**:

#### 1. `DivergenceType` (Enum)
```python
class DivergenceType(Enum):
    LEXICAL = "lexical"              # Different word choice
    SEMANTIC_EXPANSION = "expansion"  # LXX adds meaning
    SEMANTIC_RESTRICTION = "restriction"  # LXX narrows meaning
    GRAMMATICAL = "grammatical"       # Gender, number, case changes
    ADDITION = "addition"             # LXX adds words/phrases
    OMISSION = "omission"             # LXX omits MT content
    TRANSLATIONAL = "translational"   # Interpretive rendering
    TEXTUAL_VARIANT = "variant"       # Different Vorlage
```

#### 2. `ChristologicalCategory` (Enum)
```python
class ChristologicalCategory(Enum):
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
```

#### 3. `ManuscriptWitness` (Dataclass)
Fields:
- `manuscript_id: str` - Identifier (e.g., "4QIsaa", "Codex Vaticanus")
- `date_range: str` - Estimated date range (e.g., "150-100 BCE")
- `century_numeric: int` - For sorting (-3 = 3rd century BCE)
- `reading: str` - The text as found in this manuscript
- `reading_transliterated: str` - Transliteration if applicable
- `supports_lxx: bool` - Does this witness support LXX reading?
- `supports_mt: bool` - Does this witness support MT reading?
- `notes: str` - Scholarly notes on this witness
- `reliability_score: float` - Based on age and completeness (0-1)

#### 4. `NTQuotation` (Dataclass)
Fields:
- `nt_verse: str` - NT verse that quotes this passage
- `quote_type: str` - "exact", "adapted", "allusion"
- `follows_lxx: bool` - Does NT follow LXX reading?
- `follows_mt: bool` - Does NT follow MT reading?
- `nt_text_greek: str` - Greek text in NT
- `significance: str` - Why this quotation matters

#### 4. `PatristicWitness` (Dataclass)
Fields:
- `father: str` - Name of Church Father
- `work: str` - Source document
- `citation: str` - Reference
- `interpretation: str` - Father's Christological interpretation
- `preference: str` - "lxx", "mt", or "both"

#### 6. `LXXDivergence` (Dataclass)
Fields:
- `verse_id: str` - Verse reference (e.g., "ISA.7.14")
- `mt_text_hebrew: str` - Masoretic Hebrew text
- `mt_text_transliterated: str` - Transliteration
- `mt_gloss: str` - English gloss of MT
- `lxx_text_greek: str` - Septuagint Greek text
- `lxx_text_transliterated: str` - Transliteration
- `lxx_gloss: str` - English gloss of LXX
- `divergence_type: DivergenceType` - Type of difference
- `christological_category: ChristologicalCategory` - Theological category
- `christological_significance: str` - Explanation of Christological import
- `manuscript_witnesses: List[ManuscriptWitness]` - ALL manuscript evidence
- `oldest_witness: ManuscriptWitness` - The oldest available witness
- `oldest_supports: str` - "lxx", "mt", "neither", or "unique"
- `nt_quotations: List[NTQuotation]` - NT usage
- `patristic_witnesses: List[PatristicWitness]` - Father's interpretations
- `confidence: float` - Confidence in Christological significance
- `manuscript_confidence: float` - Confidence from manuscript priority
- `scholarly_notes: str` - Academic discussion points

#### 6. `LXXAnalysisResult` (Dataclass)
Fields:
- `verse_id: str` - Analyzed verse
- `divergences: List[LXXDivergence]` - All detected divergences
- `primary_christological_insight: str` - Main finding
- `total_divergence_count: int` - Number of differences found
- `christological_divergence_count: int` - Theologically significant ones
- `nt_support_strength: float` - How strongly NT supports LXX reading
- `patristic_unanimity: float` - Agreement among Fathers
- `overall_significance: float` - Composite importance score

#### 7. `LXXChristologicalExtractor` (Main Class)

**Constructor**:
- Accept LXX corpus client reference
- Accept MT corpus client reference
- Accept NT quotation database reference
- Accept patristic database reference
- Accept configuration

**Class Attributes**:
```python
# Known significant divergence verses
KNOWN_CHRISTOLOGICAL_VERSES = {
    "ISA.7.14": ChristologicalCategory.VIRGIN_BIRTH,
    "PSA.40.6": ChristologicalCategory.INCARNATION,  # LXX 39:7
    "GEN.3.15": ChristologicalCategory.SOTERIOLOGICAL,
    "PSA.22.16": ChristologicalCategory.PASSION,  # LXX 21:17
    "ISA.53.8": ChristologicalCategory.PASSION,
    "PSA.16.10": ChristologicalCategory.RESURRECTION,  # LXX 15:10
    "ISA.9.6": ChristologicalCategory.DIVINE_NATURE,
    "MIC.5.2": ChristologicalCategory.DIVINE_NATURE,
    "ZEC.12.10": ChristologicalCategory.PASSION,
    "PSA.110.1": ChristologicalCategory.ROYAL_DAVIDIC,  # LXX 109:1
    # ... extensive catalog
}

# NT books with heavy LXX dependence
LXX_DEPENDENT_NT_BOOKS = [
    "MAT", "LUK", "ACT", "ROM", "GAL", "HEB", "1PE"
]
```

**Methods**:

##### `async def extract_christological_content(self, verse_id: str) -> LXXAnalysisResult`
Main entry point:
1. Get MT text for verse
2. Get LXX text for verse (handling numbering differences)
3. Align and compare texts
4. Identify all divergences
5. Classify Christological significance of each
6. Gather NT quotation evidence
7. Gather patristic witness
8. Return complete analysis

##### `async def align_mt_lxx(self, mt_text: str, lxx_text: str) -> List[Tuple[str, str]]`
- Word-by-word alignment
- Handle Hebrew-to-Greek morphological mapping
- Account for translation techniques
- Return aligned word pairs

##### `async def detect_divergences(self, aligned_pairs: List[Tuple]) -> List[LXXDivergence]`
- Compare each aligned pair
- Classify divergence type
- Calculate semantic distance
- Return divergence list

##### `async def classify_christological_significance(self, divergence: LXXDivergence) -> Tuple[ChristologicalCategory, float]`
- Check against known significant passages
- Analyze semantic implications
- Cross-reference with NT usage
- Return category and confidence

##### `async def find_nt_quotations(self, verse_id: str) -> List[NTQuotation]`
- Query NT quotation database
- Match quotation to source verse
- Determine if NT follows LXX or MT
- Return quotation list

##### `async def gather_patristic_witness(self, verse_id: str) -> List[PatristicWitness]`
- Query patristic database for verse citations
- Extract Father's interpretation
- Note preference for LXX vs MT reading
- Return witness list

##### `async def calculate_nt_support(self, quotations: List[NTQuotation]) -> float`
- Count quotations following LXX
- Weight by theological importance
- Return support score (0-1)

##### `async def calculate_patristic_unanimity(self, witnesses: List[PatristicWitness]) -> float`
- Measure agreement among Fathers
- Account for different traditions (Antiochene vs Alexandrian)
- Return unanimity score (0-1)

##### `async def handle_verse_numbering(self, verse_id: str, from_system: str, to_system: str) -> str`
- Convert between MT and LXX numbering systems
- Handle Psalm numbering differences (LXX = MT - 1 for Psalms 10-147)
- Return equivalent verse ID

##### `async def scan_book_for_divergences(self, book_id: str) -> List[LXXAnalysisResult]`
- Scan entire book for significant divergences
- Return all Christologically significant findings
- Rank by significance

---

## Part 3: Divergence Detection Algorithms

### Algorithm 1: Lexical Comparison

```python
async def compare_lexemes(self, hebrew_word: str, greek_word: str) -> Optional[DivergenceType]:
    """
    Compare Hebrew lemma to Greek translation.
    """
    # Get standard translation equivalents
    standard_translations = await self.get_translation_equivalents(hebrew_word)

    if greek_word not in standard_translations:
        # Check if it's a legitimate alternative
        semantic_distance = await self.calculate_semantic_distance(
            hebrew_word, greek_word
        )

        if semantic_distance > 0.5:
            return DivergenceType.LEXICAL

    return None
```

### Algorithm 2: Semantic Expansion Detection

```python
async def detect_semantic_expansion(
    self, hebrew_lemma: str, greek_lemma: str
) -> Optional[LXXDivergence]:
    """
    Detect when LXX adds semantic content not in Hebrew.
    """
    hebrew_semantic_range = await self.get_semantic_range(hebrew_lemma, "hebrew")
    greek_semantic_range = await self.get_semantic_range(greek_lemma, "greek")

    # Check if Greek range exceeds Hebrew
    expansion = set(greek_semantic_range) - set(hebrew_semantic_range)

    if expansion:
        return LXXDivergence(
            divergence_type=DivergenceType.SEMANTIC_EXPANSION,
            christological_significance=f"LXX adds meanings: {expansion}"
        )

    return None
```

### Algorithm 3: Christological Classification

```python
async def classify_christological_import(
    self, divergence: LXXDivergence
) -> Tuple[ChristologicalCategory, float]:
    """
    Determine Christological category and confidence.
    """
    # Check against known catalog
    if divergence.verse_id in self.KNOWN_CHRISTOLOGICAL_VERSES:
        return (
            self.KNOWN_CHRISTOLOGICAL_VERSES[divergence.verse_id],
            0.95  # High confidence for known verses
        )

    # Analyze divergence content
    greek_text = divergence.lxx_text_greek.lower()

    # Check for Christological markers
    if any(term in greek_text for term in ["παρθένος", "χριστός", "κύριος"]):
        if "παρθένος" in greek_text:
            return (ChristologicalCategory.VIRGIN_BIRTH, 0.85)
        if "χριστός" in greek_text:
            return (ChristologicalCategory.MESSIANIC_TITLE, 0.80)

    # Check for passion language
    passion_terms = ["ὤρυξαν", "ἐξεκέντησαν", "πάσχω", "σταυρός"]
    if any(term in greek_text for term in passion_terms):
        return (ChristologicalCategory.PASSION, 0.75)

    # Default: needs manual review
    return (None, 0.0)
```

### Algorithm 4: NT Quotation Matching

```python
async def match_nt_quotation(
    self, ot_verse: str, nt_verse: str
) -> NTQuotation:
    """
    Determine how NT quotes OT and whether it follows LXX.
    """
    # Get texts
    mt_text = await self.get_mt_text(ot_verse)
    lxx_text = await self.get_lxx_text(ot_verse)
    nt_text = await self.get_nt_text(nt_verse)

    # Calculate similarity scores
    lxx_similarity = self.calculate_text_similarity(lxx_text, nt_text)
    mt_backtranslation = await self.backtranslate_to_greek(mt_text)
    mt_similarity = self.calculate_text_similarity(mt_backtranslation, nt_text)

    follows_lxx = lxx_similarity > mt_similarity + 0.1
    follows_mt = mt_similarity > lxx_similarity + 0.1

    return NTQuotation(
        nt_verse=nt_verse,
        quote_type=self.classify_quote_type(lxx_similarity),
        follows_lxx=follows_lxx,
        follows_mt=follows_mt and not follows_lxx,
        nt_text_greek=nt_text,
        significance=self.explain_significance(follows_lxx, ot_verse)
    )
```

---

## Part 4: Integration Points

### Integration 1: Corpus Access

**LXX Corpus**:
- Primary: Rahlfs 1935 edition (via local files or API)
- Secondary: Swete's LXX for manuscript variants
- Integration: `integrations/lxx_corpus.py` (new file)

**MT Corpus**:
- Primary: BHSA via Text-Fabric
- Secondary: Macula-Hebrew for morphology
- Integration: `integrations/text_fabric.py` (existing)

### Integration 2: Cross-Reference Pipeline

**Location**: `ml/inference/pipeline.py`

**Modification**:
- Add LXX divergence check for OT verses in cross-reference discovery
- Flag connections involving known Christological divergences
- Enhance confidence for prophetic connections with LXX support

```python
# In discover_cross_references()
if source_verse.is_ot and candidate.is_nt:
    lxx_analysis = await self.lxx_extractor.extract_christological_content(
        source_verse.id
    )
    if lxx_analysis.christological_divergence_count > 0:
        candidate.features["lxx_christological"] = True
        candidate.features["lxx_significance"] = lxx_analysis.overall_significance
```

### Integration 3: Neo4j Graph

**Schema Extension**:
```cypher
// Add LXX divergence properties
CREATE (v:Verse)-[:HAS_LXX_DIVERGENCE {
    type: "lexical",
    christological_category: "virgin_birth",
    mt_text: "הָעַלְמָה",
    lxx_text: "ἡ παρθένος",
    significance: 0.95
}]->(d:LXXDivergence)

// Query Christological divergences
MATCH (v:Verse)-[:HAS_LXX_DIVERGENCE]->(d)
WHERE d.christological_category IS NOT NULL
RETURN v.id, d.type, d.significance
ORDER BY d.significance DESC
```

### Integration 4: Data Schemas

**Add to `data/schemas.py`**:
```python
@dataclass
class VerseSchema:
    # ... existing fields ...

    # New LXX fields
    has_lxx_divergence: bool = False
    lxx_christological_category: Optional[str] = None
    lxx_significance_score: float = 0.0
    lxx_nt_support: bool = False
```

---

## Part 5: Database of Known Divergences

### Catalog Structure

Create supporting data file: `data/lxx_christological_catalog.json`

```json
{
  "ISA.7.14": {
    "mt_hebrew": "הִנֵּה הָעַלְמָה הָרָה וְיֹלֶדֶת בֵּן",
    "mt_gloss": "Behold, the young woman is pregnant and bears a son",
    "lxx_greek": "ἰδοὺ ἡ παρθένος ἐν γαστρὶ ἕξει καὶ τέξεται υἱόν",
    "lxx_gloss": "Behold, the virgin shall conceive and bear a son",
    "key_divergence": "עַלְמָה (almah) → παρθένος (parthenos)",
    "christological_category": "virgin_birth",
    "nt_quotations": ["MAT.1.23"],
    "patristic_witnesses": [
      {"father": "Justin Martyr", "work": "Dialogue with Trypho", "chapter": 43},
      {"father": "Irenaeus", "work": "Against Heresies", "book": 3, "chapter": 21}
    ],
    "significance": "LXX specifies biological virginity, supporting virgin birth doctrine"
  },
  "PSA.40.6": {
    "mt_hebrew": "אָזְנַיִם כָּרִיתָ לִּי",
    "mt_gloss": "Ears you have dug/opened for me",
    "lxx_greek": "σῶμα δὲ κατηρτίσω μοι",
    "lxx_gloss": "But a body you have prepared for me",
    "key_divergence": "אָזְנַיִם (ears) → σῶμα (body)",
    "christological_category": "incarnation",
    "nt_quotations": ["HEB.10.5"],
    "patristic_witnesses": [
      {"father": "Chrysostom", "work": "Homilies on Hebrews", "homily": 18}
    ],
    "significance": "LXX reading points to Incarnation; Hebrews quotes LXX form"
  }
  // ... extensive catalog
}
```

---

## Part 6: Testing Specification

### Unit Tests: `tests/ml/engines/test_lxx_extractor.py`

**Test 1: `test_isaiah_7_14_parthenos`**
- Input: ISA.7.14
- Expected: Divergence detected (almah → parthenos)
- Category: VIRGIN_BIRTH
- NT Support: MAT.1.23 follows LXX
- Confidence: > 0.9

**Test 2: `test_psalm_40_6_body_prepared`**
- Input: PSA.40.6 (handle numbering: LXX PSA.39.7)
- Expected: Divergence detected (ears → body)
- Category: INCARNATION
- NT Support: HEB.10.5 follows LXX
- Confidence: > 0.85

**Test 3: `test_genesis_3_15_autos`**
- Input: GEN.3.15
- Expected: Grammatical divergence (hu → autos)
- Category: SOTERIOLOGICAL
- Significance: Individual vs collective interpretation

**Test 4: `test_psalm_22_16_pierced`**
- Input: PSA.22.16 (LXX PSA.21.17)
- Expected: Lexical divergence (lion → pierced)
- Category: PASSION
- Note: DSS support for "pierced" reading

**Test 5: `test_verse_numbering_conversion`**
- Input: MT PSA.23.1
- Expected LXX: PSA.22.1
- Verify correct mapping for Psalms 10-147

**Test 6: `test_nt_quotation_detection`**
- Input: ISA.7.14
- Expected: Detect MAT.1.23 quotation
- Verify follows_lxx = True

**Test 7: `test_no_christological_divergence`**
- Input: GEN.1.1 (no significant LXX divergence)
- Expected: Empty or minimal divergence list
- Christological significance: None

**Test 8: `test_book_scan_isaiah`**
- Input: ISA (full book)
- Expected: Multiple Christological divergences found
- Key verses: 7:14, 9:6, 53:8, etc.

---

## Part 7: Configuration

### Add to `config.py`

**New Configuration Dataclass**: `LXXExtractorConfig`

Fields:
- `lxx_corpus_path: str = "data/corpora/lxx"` - Path to LXX data
- `catalog_path: str = "data/lxx_christological_catalog.json"` - Known divergences
- `min_divergence_significance: float = 0.5` - Threshold for reporting
- `include_patristic_witness: bool = True` - Gather Father's interpretations
- `include_nt_quotations: bool = True` - Check NT usage
- `cache_alignments: bool = True` - Cache MT-LXX alignments
- `alignment_cache_ttl: int = 604800` - One week
- `semantic_distance_threshold: float = 0.3` - For divergence detection
- `scan_batch_size: int = 50` - Verses per batch for book scanning

---

## Part 8: Plugins/Tools to Use

### During Implementation
- **sequential-thinking MCP**: Use for alignment algorithm design
- **memory MCP**: Store known divergences and patristic citations
- **context7 MCP**: Reference difflib documentation for text comparison

### Corpus Access
- **Text-Fabric integration**: For BHSA Hebrew corpus
- **LXX integration**: New integration for Septuagint text
- **Macula integration**: For morphological comparison

### External Tools
- **CATSS parallel alignment data**: If available
- **Accordance/Logos export**: For cross-checking divergences

### Testing Commands
```bash
# Run unit tests
pytest tests/ml/engines/test_lxx_extractor.py -v

# Run Isaiah 7:14 specific test
pytest tests/ml/engines/test_lxx_extractor.py -k "isaiah_7_14" -v

# Run with patristic witness tests
pytest tests/ml/engines/test_lxx_extractor.py -k "patristic" -v

# Performance testing for book scanning
pytest tests/ml/engines/test_lxx_extractor.py -k "book_scan" --benchmark
```

---

## Part 9: Success Criteria

### Functional Requirements
- [ ] Correctly aligns MT and LXX texts
- [ ] Detects lexical divergences accurately
- [ ] Classifies Christological categories correctly
- [ ] Finds NT quotations for source verses
- [ ] Handles verse numbering differences (especially Psalms)
- [ ] Scans entire books efficiently

### Theological Accuracy
- [ ] ISA.7.14: παρθένος divergence detected with VIRGIN_BIRTH category
- [ ] PSA.40.6: σῶμα divergence detected with INCARNATION category
- [ ] GEN.3.15: αὐτός divergence detected with SOTERIOLOGICAL category
- [ ] Known catalog entries all pass validation

### Performance Requirements
- [ ] Single verse analysis: < 1 second (with cache)
- [ ] First-time verse analysis: < 5 seconds
- [ ] Full book scan: < 2 minutes
- [ ] NT quotation lookup: < 500ms

---

## Part 10: Detailed Implementation Order

1. **Create enums**: `DivergenceType`, `ChristologicalCategory`
2. **Create dataclasses**: `NTQuotation`, `PatristicWitness`, `LXXDivergence`, `LXXAnalysisResult`
3. **Create LXX corpus integration** in `integrations/lxx_corpus.py`
4. **Implement `handle_verse_numbering()`** - crucial for Psalms
5. **Implement `align_mt_lxx()`** - core alignment algorithm
6. **Implement `detect_divergences()`** - divergence detection
7. **Implement `classify_christological_significance()`** - categorization
8. **Implement `find_nt_quotations()`** - NT cross-reference
9. **Implement `gather_patristic_witness()`** - Father's interpretations
10. **Implement main `extract_christological_content()`** - orchestration
11. **Implement `scan_book_for_divergences()`** - batch processing
12. **Create `data/lxx_christological_catalog.json`** - known divergences
13. **Add caching layer** for alignments
14. **Add configuration to `config.py`**
15. **Write and run unit tests**
16. **Validate against known Christological passages**

---

## Part 11: Dependencies on Other Sessions

### Depends On
- SESSION 03: Omni-Contextual Resolver (for Greek word meaning resolution)

### Depended On By
- SESSION 06: Fractal Typology Engine (uses LXX readings for type analysis)
- SESSION 07: Prophetic Necessity Prover (uses LXX for prophecy evidence)
- SESSION 11: Pipeline Integration (incorporates LXX analysis)

### External Dependencies
- LXX corpus data (Rahlfs or Swete)
- NT quotation apparatus
- Patristic citation database
- Greek-Hebrew lexical alignment data

---

## Part 12: New File Requirements

### New Integration File: `integrations/lxx_corpus.py`

```python
"""
Septuagint (LXX) Corpus Integration

Provides access to Septuagint Greek text with:
- Verse retrieval by reference
- Morphological analysis
- Manuscript variant tracking
- Verse numbering conversion
"""

class LXXCorpusClient:
    async def get_verse(self, verse_id: str) -> Dict
    async def get_morphology(self, verse_id: str) -> List[Dict]
    async def convert_reference(self, ref: str, from_system: str, to_system: str) -> str
    async def get_variants(self, verse_id: str) -> List[Dict]
```

---

## Session Completion Checklist

```markdown
- [ ] `ml/engines/lxx_extractor.py` implemented
- [ ] `integrations/lxx_corpus.py` created
- [ ] All enums and dataclasses defined
- [ ] Verse numbering conversion working (especially Psalms)
- [ ] MT-LXX alignment functional
- [ ] Divergence detection accurate
- [ ] Christological classification working
- [ ] NT quotation lookup functional
- [ ] Patristic witness gathering working
- [ ] `data/lxx_christological_catalog.json` created
- [ ] Caching layer integrated
- [ ] Configuration added to config.py
- [ ] ISA.7.14 test passing (παρθένος)
- [ ] PSA.40.6 test passing (σῶμα)
- [ ] GEN.3.15 test passing (αὐτός)
- [ ] Performance tests passing
- [ ] Documentation complete
```

**Next Session**: SESSION 06: Hyper-Fractal Typology Engine
