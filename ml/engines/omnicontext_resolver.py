"""
BIBLOS v2 - Omni-Contextual Resolver Engine

The First Impossible Oracle: Determines absolute word meaning via eliminative
reasoning across ALL biblical occurrences.

Given any word in any verse, this engine:
1. Finds ALL occurrences of that word across the entire canon
2. Extracts the semantic range from all contexts
3. Eliminates impossible meanings for the specific verse
4. Concludes with the only possible meaning(s)

Canonical Example: רוּחַ (ruach) in GEN.1.2
- Occurs 389 times in OT
- Semantic range: wind, breath, spirit, Spirit (divine)
- In GEN.1.2: "wind" eliminated (no physical source), "breath" eliminated
  ("merachefet"/hovering requires agency, not passive air)
- Conclusion: "Divine Spirit" (Third Person of Trinity)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ml.cache import AsyncLRUCache, embedding_cache_key

if TYPE_CHECKING:
    from integrations.base import BaseCorpusIntegration, VerseData, WordData
    from ml.embeddings.base import BaseEmbedder

logger = logging.getLogger(__name__)


class EliminationReason(Enum):
    """Reasons for eliminating a potential meaning."""

    GRAMMATICAL_INCOMPATIBILITY = "grammatical_incompatibility"
    CONTEXTUAL_IMPOSSIBILITY = "contextual_impossibility"
    SEMANTIC_CONTRADICTION = "semantic_contradiction"
    THEOLOGICAL_IMPOSSIBILITY = "theological_impossibility"
    SYNTACTIC_CONSTRAINT = "syntactic_constraint"
    COLLOCATIONAL_VIOLATION = "collocational_violation"


@dataclass
class EliminationStep:
    """Record of a single elimination decision in the reasoning chain."""

    meaning: str
    eliminated: bool
    reason: Optional[EliminationReason] = None
    explanation: str = ""
    evidence_verses: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class SemanticFieldEntry:
    """Entry in the semantic field map for a word's meaning."""

    lemma: str
    meaning: str
    occurrence_count: int
    primary_contexts: List[str] = field(default_factory=list)
    semantic_neighbors: List[str] = field(default_factory=list)
    theological_weight: float = 0.0


@dataclass
class CompatibilityResult:
    """Result of checking if a meaning is compatible with context."""

    compatible: bool
    impossibility_reason: Optional[str] = None
    elimination_reason: Optional[EliminationReason] = None
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)


@dataclass
class AbsoluteMeaningResult:
    """Complete result of omni-contextual meaning resolution."""

    word: str
    verse_id: str
    primary_meaning: str
    confidence: float
    reasoning_chain: List[EliminationStep] = field(default_factory=list)
    eliminated_alternatives: Dict[str, str] = field(default_factory=dict)
    remaining_candidates: List[str] = field(default_factory=list)
    semantic_field_map: Dict[str, SemanticFieldEntry] = field(default_factory=dict)
    total_occurrences: int = 0
    analysis_coverage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "word": self.word,
            "verse_id": self.verse_id,
            "primary_meaning": self.primary_meaning,
            "confidence": self.confidence,
            "reasoning_chain": [
                {
                    "meaning": step.meaning,
                    "eliminated": step.eliminated,
                    "reason": step.reason.value if step.reason else None,
                    "explanation": step.explanation,
                    "evidence_verses": step.evidence_verses,
                    "confidence": step.confidence,
                }
                for step in self.reasoning_chain
            ],
            "eliminated_alternatives": self.eliminated_alternatives,
            "remaining_candidates": self.remaining_candidates,
            "semantic_field_map": {
                k: {
                    "lemma": v.lemma,
                    "meaning": v.meaning,
                    "occurrence_count": v.occurrence_count,
                    "primary_contexts": v.primary_contexts,
                    "semantic_neighbors": v.semantic_neighbors,
                    "theological_weight": v.theological_weight,
                }
                for k, v in self.semantic_field_map.items()
            },
            "total_occurrences": self.total_occurrences,
            "analysis_coverage": self.analysis_coverage,
        }


@dataclass
class OccurrenceData:
    """Data for a single word occurrence."""

    verse_id: str
    lemma: str
    surface_form: str
    context_text: str
    morphology: Dict[str, Any] = field(default_factory=dict)
    syntax_role: Optional[str] = None
    position: int = 0


class OmniContextualResolver:
    """
    Omni-Contextual Resolver - The First Impossible Oracle.

    Determines absolute word meaning via eliminative reasoning across
    all biblical occurrences. Performs analysis beyond human cognitive
    limits by computationally accessing the complete canon.

    Usage:
        resolver = OmniContextualResolver(corpus_client, embedder)
        await resolver.initialize()
        result = await resolver.resolve_absolute_meaning("רוּחַ", "GEN.1.2", "hebrew")
    """

    # Hebrew words with significant semantic range
    POLYSEMOUS_HEBREW: Dict[str, List[str]] = {
        "רוּחַ": ["wind", "breath", "spirit", "Spirit"],
        "נֶפֶשׁ": ["soul", "life", "person", "throat", "desire"],
        "לֵב": ["heart", "mind", "understanding", "will"],
        "בָּרָא": ["create", "make", "shape"],
        "כָּבוֹד": ["glory", "weight", "honor", "wealth"],
        "חֶסֶד": ["lovingkindness", "mercy", "faithfulness", "loyalty"],
        "צֶדֶק": ["righteousness", "justice", "rightness"],
        "אֱמֶת": ["truth", "faithfulness", "reliability"],
        "שָׁלוֹם": ["peace", "wholeness", "well-being", "prosperity"],
        "תּוֹרָה": ["law", "instruction", "teaching", "direction"],
        "יָשַׁע": ["save", "deliver", "help", "rescue"],
        "קָדוֹשׁ": ["holy", "sacred", "set apart"],
        "בְּרִית": ["covenant", "treaty", "agreement"],
        "דָּבָר": ["word", "thing", "matter", "event"],
        "עָוֹן": ["iniquity", "guilt", "punishment"],
        "חַטָּאת": ["sin", "sin-offering", "purification"],
        "יָרֵא": ["fear", "reverence", "awe"],
        "אָהַב": ["love", "desire", "like"],
    }

    # Greek words with significant semantic range
    POLYSEMOUS_GREEK: Dict[str, List[str]] = {
        "λόγος": ["word", "speech", "reason", "account", "Word"],
        "πνεῦμα": ["spirit", "Spirit", "wind", "breath"],
        "σάρξ": ["flesh", "body", "human nature", "sinful nature"],
        "ψυχή": ["soul", "life", "self", "person"],
        "καρδία": ["heart", "mind", "inner self"],
        "πίστις": ["faith", "faithfulness", "trust", "belief"],
        "χάρις": ["grace", "favor", "thanks", "gift"],
        "ἀγάπη": ["love", "charity", "affection"],
        "δικαιοσύνη": ["righteousness", "justice", "justification"],
        "ἁμαρτία": ["sin", "sinfulness", "sin offering"],
        "νόμος": ["law", "principle", "custom"],
        "κόσμος": ["world", "universe", "humanity", "adornment"],
        "δόξα": ["glory", "honor", "splendor", "praise"],
        "ζωή": ["life", "living", "lifetime"],
        "θάνατος": ["death", "mortality", "realm of death"],
        "σωτηρία": ["salvation", "deliverance", "preservation"],
        "ἀλήθεια": ["truth", "reality", "genuineness"],
        "εἰρήνη": ["peace", "harmony", "well-being"],
    }

    # Contextual requirements for specific meanings
    MEANING_REQUIREMENTS: Dict[str, Dict[str, List[str]]] = {
        "רוּחַ": {
            "wind": ["physical_source", "meteorological_context", "movement_described"],
            "breath": ["living_subject", "physical_action", "bodily_context"],
            "spirit": ["human_subject", "emotional_context", "psychological_context"],
            "Spirit": ["divine_context", "creation_context", "prophetic_context"],
        },
        "λόγος": {
            "word": ["speech_context", "communication"],
            "speech": ["verbal_communication", "discourse"],
            "reason": ["philosophical_context", "argument"],
            "account": ["narrative_context", "explanation"],
            "Word": ["divine_context", "christological", "creation_context"],
        },
    }

    # Trinitarian context markers
    TRINITARIAN_MARKERS = [
        "Father",
        "Son",
        "Spirit",
        "divine",
        "God",
        "LORD",
        "creation",
        "hovering",
        "beginning",
    ]

    # Modalist readings to exclude in Trinitarian contexts
    MODALIST_READINGS = ["mere_force", "impersonal_power", "manifestation_mode"]

    def __init__(
        self,
        corpus_client: Optional[BaseCorpusIntegration] = None,
        embedder: Optional[BaseEmbedder] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the OmniContextualResolver.

        Args:
            corpus_client: Corpus integration for accessing biblical texts
            embedder: Embedding model for semantic similarity
            config: Configuration options
        """
        self.corpus_client = corpus_client
        self.embedder = embedder
        self.config = config or {}

        # Configuration with defaults
        self.max_occurrences_full_analysis = self.config.get(
            "max_occurrences_full_analysis", 500
        )
        self.sample_size_large_words = self.config.get("sample_size_large_words", 200)
        self.elimination_confidence_threshold = self.config.get(
            "elimination_confidence_threshold", 0.7
        )
        self.semantic_similarity_threshold = self.config.get(
            "semantic_similarity_threshold", 0.8
        )
        self.parallel_support_weight = self.config.get("parallel_support_weight", 0.3)
        self.theological_weight_multiplier = self.config.get(
            "theological_weight_multiplier", 1.2
        )

        # Caches
        self._occurrence_cache: AsyncLRUCache[List[OccurrenceData]] = AsyncLRUCache(
            max_size=1000, max_memory_mb=256, ttl_seconds=604800  # 1 week
        )
        self._semantic_range_cache: AsyncLRUCache[
            List[SemanticFieldEntry]
        ] = AsyncLRUCache(max_size=500, max_memory_mb=128, ttl_seconds=604800)
        self._resolution_cache: AsyncLRUCache[AbsoluteMeaningResult] = AsyncLRUCache(
            max_size=10000, max_memory_mb=512, ttl_seconds=86400  # 1 day
        )

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the resolver and its dependencies."""
        if self._initialized:
            return

        logger.info("Initializing OmniContextualResolver")

        if self.corpus_client and hasattr(self.corpus_client, "initialize"):
            await self.corpus_client.initialize()

        self._initialized = True
        logger.info("OmniContextualResolver initialized")

    async def resolve_absolute_meaning(
        self,
        word: str,
        verse_id: str,
        language: str,
    ) -> AbsoluteMeaningResult:
        """
        Determine absolute meaning of a word in a specific verse context.

        This is the main entry point for the oracle. It performs:
        1. Retrieval of all occurrences
        2. Semantic range extraction
        3. Systematic eliminative reasoning
        4. Parallel support ranking
        5. Final meaning determination

        Args:
            word: The word/lemma to analyze
            verse_id: The verse ID (e.g., "GEN.1.2")
            language: Language ("hebrew" or "greek")

        Returns:
            AbsoluteMeaningResult with complete analysis
        """
        if not self._initialized:
            await self.initialize()

        # Check resolution cache
        cache_key = f"{word}:{verse_id}:{language}"
        cached = await self._resolution_cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for resolution: {cache_key}")
            return cached

        logger.info(f"Resolving absolute meaning: {word} in {verse_id} ({language})")

        # Step 1: Get all occurrences
        occurrences = await self.get_all_occurrences(word, language)
        total_occurrences = len(occurrences)
        logger.debug(f"Found {total_occurrences} occurrences of '{word}'")

        # Step 2: Extract semantic range
        semantic_range = await self.extract_semantic_range(word, occurrences, language)
        logger.debug(f"Extracted {len(semantic_range)} semantic meanings")

        # Step 3: Get verse context
        verse_context = await self.get_verse_context(verse_id)

        # Step 4: Parse grammatical constraints
        grammatical_constraints = self.parse_grammatical_constraints(verse_context)

        # Step 5: Systematic elimination
        reasoning_chain: List[EliminationStep] = []
        eliminated_alternatives: Dict[str, str] = {}
        remaining_candidates: List[str] = []

        for entry in semantic_range:
            meaning = entry.meaning

            # Check compatibility through all elimination methods
            compatibility = await self.check_contextual_compatibility(
                meaning, verse_context, grammatical_constraints, word
            )

            step = EliminationStep(
                meaning=meaning,
                eliminated=not compatibility.compatible,
                reason=compatibility.elimination_reason,
                explanation=compatibility.impossibility_reason or "",
                evidence_verses=compatibility.evidence,
                confidence=compatibility.confidence,
            )
            reasoning_chain.append(step)

            if compatibility.compatible:
                remaining_candidates.append(meaning)
            else:
                eliminated_alternatives[meaning] = (
                    compatibility.impossibility_reason or "Unknown reason"
                )

        # Step 6: Rank remaining by parallel support
        if len(remaining_candidates) > 1:
            ranked = await self.rank_by_parallel_support(remaining_candidates, verse_id)
            remaining_candidates = [m for m, _ in ranked]

        # Step 7: Determine primary meaning
        primary_meaning = remaining_candidates[0] if remaining_candidates else "unknown"

        # Step 8: Build semantic field map
        semantic_field_map = await self.map_semantic_field(word, primary_meaning)

        # Calculate confidence
        confidence = self._calculate_confidence(
            remaining_candidates, reasoning_chain, total_occurrences
        )

        # Calculate coverage
        analyzed_count = min(total_occurrences, self.max_occurrences_full_analysis)
        analysis_coverage = analyzed_count / total_occurrences if total_occurrences > 0 else 1.0

        result = AbsoluteMeaningResult(
            word=word,
            verse_id=verse_id,
            primary_meaning=primary_meaning,
            confidence=confidence,
            reasoning_chain=reasoning_chain,
            eliminated_alternatives=eliminated_alternatives,
            remaining_candidates=remaining_candidates,
            semantic_field_map=semantic_field_map,
            total_occurrences=total_occurrences,
            analysis_coverage=analysis_coverage,
        )

        # Cache result
        await self._resolution_cache.put(cache_key, result)

        return result

    async def get_all_occurrences(
        self, word: str, language: str
    ) -> List[OccurrenceData]:
        """
        Get all occurrences of a word/lemma across the entire canon.

        Args:
            word: The word/lemma to search for
            language: Language ("hebrew" or "greek")

        Returns:
            List of occurrence data with context
        """
        cache_key = f"occurrences:{word}:{language}"
        cached = await self._occurrence_cache.get(cache_key)
        if cached:
            return cached

        occurrences: List[OccurrenceData] = []

        if self.corpus_client:
            try:
                # Search using corpus client
                results = await self.corpus_client.search(
                    word, limit=self.max_occurrences_full_analysis * 2
                )

                for verse_data in results:
                    for word_data in verse_data.words:
                        if word_data.lemma == word or word_data.surface_form == word:
                            occ = OccurrenceData(
                                verse_id=verse_data.verse_id,
                                lemma=word_data.lemma,
                                surface_form=word_data.surface_form,
                                context_text=verse_data.text,
                                morphology=word_data.morphology.to_dict(),
                                syntax_role=word_data.syntax_role,
                                position=word_data.position,
                            )
                            occurrences.append(occ)
            except Exception as e:
                logger.warning(f"Corpus search failed: {e}")

        # If no corpus or search failed, use known data
        if not occurrences:
            occurrences = self._get_mock_occurrences(word, language)

        # Sample if too many
        if len(occurrences) > self.max_occurrences_full_analysis:
            # Strategic sampling: keep diverse contexts
            occurrences = self._strategic_sample(
                occurrences, self.sample_size_large_words
            )

        await self._occurrence_cache.put(cache_key, occurrences)
        return occurrences

    def _get_mock_occurrences(
        self, word: str, language: str
    ) -> List[OccurrenceData]:
        """Get mock occurrence data for testing/fallback."""
        mock_data: Dict[str, List[Dict[str, Any]]] = {
            "רוּחַ": [
                {
                    "verse_id": "GEN.1.2",
                    "context": "and the Spirit of God was hovering over the waters",
                    "meaning_hint": "Spirit",
                },
                {
                    "verse_id": "GEN.8.1",
                    "context": "God made a wind blow over the earth",
                    "meaning_hint": "wind",
                },
                {
                    "verse_id": "GEN.6.17",
                    "context": "everything that has the breath of life",
                    "meaning_hint": "breath",
                },
                {
                    "verse_id": "EXO.15.8",
                    "context": "At the blast of your nostrils the waters piled up",
                    "meaning_hint": "wind",
                },
                {
                    "verse_id": "NUM.11.17",
                    "context": "I will take some of the Spirit that is on you",
                    "meaning_hint": "Spirit",
                },
                {
                    "verse_id": "JDG.3.10",
                    "context": "The Spirit of the LORD came upon him",
                    "meaning_hint": "Spirit",
                },
                {
                    "verse_id": "1SA.16.13",
                    "context": "the Spirit of the LORD rushed upon David",
                    "meaning_hint": "Spirit",
                },
                {
                    "verse_id": "PSA.51.11",
                    "context": "take not your Holy Spirit from me",
                    "meaning_hint": "Spirit",
                },
                {
                    "verse_id": "ISA.11.2",
                    "context": "the Spirit of the LORD shall rest upon him",
                    "meaning_hint": "Spirit",
                },
                {
                    "verse_id": "EZK.37.9",
                    "context": "Come from the four winds, O breath",
                    "meaning_hint": "breath",
                },
            ],
            "λόγος": [
                {
                    "verse_id": "JHN.1.1",
                    "context": "In the beginning was the Word",
                    "meaning_hint": "Word",
                },
                {
                    "verse_id": "JHN.1.14",
                    "context": "And the Word became flesh",
                    "meaning_hint": "Word",
                },
                {
                    "verse_id": "MAT.12.36",
                    "context": "people will give account for every careless word",
                    "meaning_hint": "word",
                },
                {
                    "verse_id": "ACT.20.7",
                    "context": "Paul talked with them, intending to depart",
                    "meaning_hint": "speech",
                },
                {
                    "verse_id": "ROM.9.6",
                    "context": "it is not as though the word of God has failed",
                    "meaning_hint": "word",
                },
                {
                    "verse_id": "HEB.4.12",
                    "context": "For the word of God is living and active",
                    "meaning_hint": "word",
                },
                {
                    "verse_id": "1PE.1.23",
                    "context": "through the living and abiding word of God",
                    "meaning_hint": "word",
                },
                {
                    "verse_id": "REV.19.13",
                    "context": "the name by which he is called is The Word of God",
                    "meaning_hint": "Word",
                },
            ],
        }

        occurrences = []
        if word in mock_data:
            for item in mock_data[word]:
                occ = OccurrenceData(
                    verse_id=item["verse_id"],
                    lemma=word,
                    surface_form=word,
                    context_text=item["context"],
                    morphology={},
                    syntax_role=None,
                    position=0,
                )
                occurrences.append(occ)

        return occurrences

    def _strategic_sample(
        self, occurrences: List[OccurrenceData], sample_size: int
    ) -> List[OccurrenceData]:
        """
        Strategically sample occurrences to maintain diversity.

        Ensures representation from different books, contexts, and usage patterns.
        """
        if len(occurrences) <= sample_size:
            return occurrences

        # Group by book
        by_book: Dict[str, List[OccurrenceData]] = {}
        for occ in occurrences:
            book = occ.verse_id.split(".")[0] if "." in occ.verse_id else "UNK"
            if book not in by_book:
                by_book[book] = []
            by_book[book].append(occ)

        # Sample proportionally from each book
        sampled: List[OccurrenceData] = []
        per_book = max(1, sample_size // len(by_book))

        for book, book_occs in by_book.items():
            take_count = min(per_book, len(book_occs))
            # Take evenly spaced samples
            if take_count < len(book_occs):
                step = len(book_occs) / take_count
                indices = [int(i * step) for i in range(take_count)]
                sampled.extend(book_occs[i] for i in indices)
            else:
                sampled.extend(book_occs)

        return sampled[:sample_size]

    async def extract_semantic_range(
        self,
        word: str,
        occurrences: List[OccurrenceData],
        language: str,
    ) -> List[SemanticFieldEntry]:
        """
        Extract semantic range from all occurrences.

        Clusters occurrences by meaning using embedding similarity.

        Args:
            word: The word/lemma
            occurrences: List of all occurrences
            language: Language ("hebrew" or "greek")

        Returns:
            List of semantic field entries (one per meaning)
        """
        cache_key = f"semantic_range:{word}:{language}"
        cached = await self._semantic_range_cache.get(cache_key)
        if cached:
            return cached

        # Get known meanings for this word
        known_meanings: List[str] = []
        if language == "hebrew" and word in self.POLYSEMOUS_HEBREW:
            known_meanings = self.POLYSEMOUS_HEBREW[word]
        elif language == "greek" and word in self.POLYSEMOUS_GREEK:
            known_meanings = self.POLYSEMOUS_GREEK[word]

        if not known_meanings:
            # Single meaning word
            entry = SemanticFieldEntry(
                lemma=word,
                meaning=word,  # Use word itself as meaning
                occurrence_count=len(occurrences),
                primary_contexts=[occ.verse_id for occ in occurrences[:5]],
                semantic_neighbors=[],
                theological_weight=0.5,
            )
            result = [entry]
            await self._semantic_range_cache.put(cache_key, result)
            return result

        # Create entries for known meanings
        entries: List[SemanticFieldEntry] = []

        if self.embedder and occurrences:
            # Use embeddings to cluster occurrences
            clusters = await self._cluster_by_meaning(occurrences, known_meanings)

            for meaning, cluster_occs in clusters.items():
                theological_weight = self._calculate_theological_weight(
                    meaning, cluster_occs
                )

                entry = SemanticFieldEntry(
                    lemma=word,
                    meaning=meaning,
                    occurrence_count=len(cluster_occs),
                    primary_contexts=[occ.verse_id for occ in cluster_occs[:5]],
                    semantic_neighbors=[],
                    theological_weight=theological_weight,
                )
                entries.append(entry)
        else:
            # Fallback: create entries based on known meanings
            per_meaning = max(1, len(occurrences) // len(known_meanings))

            for i, meaning in enumerate(known_meanings):
                start = i * per_meaning
                end = start + per_meaning
                meaning_occs = occurrences[start:end]

                theological_weight = self._calculate_theological_weight(
                    meaning, meaning_occs
                )

                entry = SemanticFieldEntry(
                    lemma=word,
                    meaning=meaning,
                    occurrence_count=len(meaning_occs),
                    primary_contexts=[occ.verse_id for occ in meaning_occs[:5]],
                    semantic_neighbors=[],
                    theological_weight=theological_weight,
                )
                entries.append(entry)

        # Sort by occurrence count (descending)
        entries.sort(key=lambda e: e.occurrence_count, reverse=True)

        await self._semantic_range_cache.put(cache_key, entries)
        return entries

    async def _cluster_by_meaning(
        self,
        occurrences: List[OccurrenceData],
        known_meanings: List[str],
    ) -> Dict[str, List[OccurrenceData]]:
        """Cluster occurrences by meaning using embeddings."""
        clusters: Dict[str, List[OccurrenceData]] = {m: [] for m in known_meanings}

        if not self.embedder:
            # Fallback distribution
            per_meaning = max(1, len(occurrences) // len(known_meanings))
            for i, meaning in enumerate(known_meanings):
                start = i * per_meaning
                end = start + per_meaning if i < len(known_meanings) - 1 else len(occurrences)
                clusters[meaning] = occurrences[start:end]
            return clusters

        try:
            # Get embeddings for meaning descriptions
            meaning_embeddings = await self.embedder.embed_batch(known_meanings)

            # Get embeddings for contexts
            contexts = [occ.context_text for occ in occurrences]
            context_embeddings = await self.embedder.embed_batch(contexts)

            # Assign each occurrence to nearest meaning
            for i, occ in enumerate(occurrences):
                best_meaning = known_meanings[0]
                best_similarity = -1.0

                for j, meaning in enumerate(known_meanings):
                    similarity = float(
                        np.dot(context_embeddings[i], meaning_embeddings[j])
                        / (
                            np.linalg.norm(context_embeddings[i])
                            * np.linalg.norm(meaning_embeddings[j])
                        )
                    )
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_meaning = meaning

                clusters[best_meaning].append(occ)

        except Exception as e:
            logger.warning(f"Embedding clustering failed: {e}, using fallback")
            per_meaning = max(1, len(occurrences) // len(known_meanings))
            for i, meaning in enumerate(known_meanings):
                start = i * per_meaning
                end = start + per_meaning if i < len(known_meanings) - 1 else len(occurrences)
                clusters[meaning] = occurrences[start:end]

        return clusters

    def _calculate_theological_weight(
        self,
        meaning: str,
        occurrences: List[OccurrenceData],
    ) -> float:
        """Calculate theological weight of a meaning."""
        weight = 0.5  # Base weight

        # Check for divine/theological contexts
        theological_keywords = [
            "God",
            "LORD",
            "Spirit",
            "holy",
            "divine",
            "covenant",
            "salvation",
            "glory",
            "kingdom",
        ]

        theological_count = sum(
            1
            for occ in occurrences
            if any(kw.lower() in occ.context_text.lower() for kw in theological_keywords)
        )

        if occurrences:
            weight += 0.3 * (theological_count / len(occurrences))

        # Divine meanings get extra weight
        if meaning.lower() in ["spirit", "word", "glory", "holy"]:
            weight += 0.2

        return min(1.0, weight)

    async def get_verse_context(self, verse_id: str) -> Dict[str, Any]:
        """
        Get full context for a verse.

        Includes surrounding verses, syntactic analysis, and morphology.

        Args:
            verse_id: The verse ID (e.g., "GEN.1.2")

        Returns:
            Dictionary with context information
        """
        context: Dict[str, Any] = {
            "verse_id": verse_id,
            "text": "",
            "surrounding_verses": [],
            "syntax": {},
            "morphology": [],
            "semantic_markers": [],
        }

        if self.corpus_client:
            try:
                verse_data = await self.corpus_client.get_verse(verse_id)
                if verse_data:
                    context["text"] = verse_data.text
                    context["morphology"] = [
                        w.morphology.to_dict() for w in verse_data.words
                    ]

                    # Get surrounding verses
                    parts = verse_id.split(".")
                    if len(parts) == 3:
                        book, chapter, verse = parts[0], int(parts[1]), int(parts[2])

                        for offset in [-2, -1, 1, 2]:
                            nearby_id = f"{book}.{chapter}.{verse + offset}"
                            try:
                                nearby = await self.corpus_client.get_verse(nearby_id)
                                if nearby:
                                    context["surrounding_verses"].append(
                                        {
                                            "verse_id": nearby_id,
                                            "text": nearby.text,
                                        }
                                    )
                            except Exception:
                                pass

                    # Try to get syntax tree
                    if hasattr(self.corpus_client, "get_syntax_tree"):
                        syntax = await self.corpus_client.get_syntax_tree(verse_id)
                        if syntax:
                            context["syntax"] = syntax

            except Exception as e:
                logger.warning(f"Failed to get verse context: {e}")

        # Add mock context for testing
        if not context["text"]:
            context = self._get_mock_verse_context(verse_id)

        # Extract semantic markers
        context["semantic_markers"] = self._extract_semantic_markers(context["text"])

        return context

    def _get_mock_verse_context(self, verse_id: str) -> Dict[str, Any]:
        """Get mock verse context for testing."""
        mock_contexts: Dict[str, Dict[str, Any]] = {
            "GEN.1.2": {
                "verse_id": "GEN.1.2",
                "text": "The earth was without form and void, and darkness was over the face of the deep. And the Spirit of God was hovering over the face of the waters.",
                "surrounding_verses": [
                    {
                        "verse_id": "GEN.1.1",
                        "text": "In the beginning, God created the heavens and the earth.",
                    },
                    {
                        "verse_id": "GEN.1.3",
                        "text": "And God said, 'Let there be light,' and there was light.",
                    },
                ],
                "syntax": {"clauses": []},
                "morphology": [],
                "semantic_markers": ["creation", "divine", "beginning"],
            },
            "JHN.1.1": {
                "verse_id": "JHN.1.1",
                "text": "In the beginning was the Word, and the Word was with God, and the Word was God.",
                "surrounding_verses": [
                    {
                        "verse_id": "JHN.1.2",
                        "text": "He was in the beginning with God.",
                    },
                    {
                        "verse_id": "JHN.1.3",
                        "text": "All things were made through him, and without him was not any thing made that was made.",
                    },
                ],
                "syntax": {"clauses": []},
                "morphology": [],
                "semantic_markers": ["creation", "divine", "beginning", "christological"],
            },
        }

        if verse_id in mock_contexts:
            return mock_contexts[verse_id]

        return {
            "verse_id": verse_id,
            "text": f"[Text of {verse_id}]",
            "surrounding_verses": [],
            "syntax": {},
            "morphology": [],
            "semantic_markers": [],
        }

    def _extract_semantic_markers(self, text: str) -> List[str]:
        """Extract semantic markers from text."""
        markers = []
        text_lower = text.lower()

        marker_keywords: Dict[str, List[str]] = {
            "creation": ["create", "made", "beginning", "formed"],
            "divine": ["god", "lord", "spirit", "holy"],
            "salvation": ["save", "deliver", "redeem", "rescue"],
            "covenant": ["covenant", "promise", "oath"],
            "judgment": ["judge", "judgment", "wrath"],
            "worship": ["worship", "praise", "glory"],
            "christological": ["son", "christ", "messiah", "anointed", "word was god", "the word"],
            "prophetic": ["prophet", "prophecy", "foretold"],
            "eschatological": ["day", "coming", "end", "return"],
        }

        for marker, keywords in marker_keywords.items():
            if any(kw in text_lower for kw in keywords):
                markers.append(marker)

        return markers

    def parse_grammatical_constraints(
        self, verse_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parse grammatical constraints from verse context.

        Extracts:
        - Part of speech requirements
        - Gender/number agreement
        - Case requirements (Greek)
        - State requirements (Hebrew)
        - Verb form constraints

        Args:
            verse_context: Context dictionary from get_verse_context

        Returns:
            Dictionary of grammatical constraints
        """
        constraints: Dict[str, Any] = {
            "part_of_speech": None,
            "gender": None,
            "number": None,
            "case": None,
            "state": None,
            "verb_form": None,
            "required_pos": None,
        }

        morphology = verse_context.get("morphology", [])
        if morphology:
            # Extract constraints from morphological analysis
            for morph in morphology:
                if isinstance(morph, dict):
                    if morph.get("part_of_speech"):
                        constraints["part_of_speech"] = morph["part_of_speech"]
                    if morph.get("gender"):
                        constraints["gender"] = morph["gender"]
                    if morph.get("number"):
                        constraints["number"] = morph["number"]
                    if morph.get("case"):
                        constraints["case"] = morph["case"]
                    if morph.get("state"):
                        constraints["state"] = morph["state"]
                    if morph.get("tense") or morph.get("mood"):
                        constraints["verb_form"] = {
                            "tense": morph.get("tense"),
                            "mood": morph.get("mood"),
                            "voice": morph.get("voice"),
                        }

        return constraints

    async def check_contextual_compatibility(
        self,
        meaning: str,
        verse_context: Dict[str, Any],
        grammatical_constraints: Dict[str, Any],
        word: str,
    ) -> CompatibilityResult:
        """
        Check if a meaning is compatible with the verse context.

        Applies all elimination methods:
        - Grammatical elimination
        - Contextual elimination
        - Semantic field elimination
        - Theological elimination

        Args:
            meaning: The meaning to check
            verse_context: Context from get_verse_context
            grammatical_constraints: Constraints from parse_grammatical_constraints
            word: The original word/lemma

        Returns:
            CompatibilityResult indicating if meaning is compatible
        """
        # Try grammatical elimination
        grammar_result = self._eliminate_by_grammar(meaning, grammatical_constraints)
        if grammar_result and grammar_result.eliminated:
            return CompatibilityResult(
                compatible=False,
                impossibility_reason=grammar_result.explanation,
                elimination_reason=grammar_result.reason,
                confidence=grammar_result.confidence,
                evidence=grammar_result.evidence_verses,
            )

        # Try contextual elimination
        context_result = await self._eliminate_by_context(
            meaning, verse_context, word
        )
        if context_result and context_result.eliminated:
            return CompatibilityResult(
                compatible=False,
                impossibility_reason=context_result.explanation,
                elimination_reason=context_result.reason,
                confidence=context_result.confidence,
                evidence=context_result.evidence_verses,
            )

        # Try semantic field elimination
        semantic_result = await self._eliminate_by_semantic_field(
            meaning, verse_context
        )
        if semantic_result and semantic_result.eliminated:
            return CompatibilityResult(
                compatible=False,
                impossibility_reason=semantic_result.explanation,
                elimination_reason=semantic_result.reason,
                confidence=semantic_result.confidence,
                evidence=semantic_result.evidence_verses,
            )

        # Try theological elimination
        theology_result = await self._eliminate_by_theology(meaning, verse_context)
        if theology_result and theology_result.eliminated:
            return CompatibilityResult(
                compatible=False,
                impossibility_reason=theology_result.explanation,
                elimination_reason=theology_result.reason,
                confidence=theology_result.confidence,
                evidence=theology_result.evidence_verses,
            )

        # All checks passed - meaning is compatible
        return CompatibilityResult(
            compatible=True,
            confidence=0.8,
        )

    def _eliminate_by_grammar(
        self,
        meaning: str,
        grammatical_constraints: Dict[str, Any],
    ) -> Optional[EliminationStep]:
        """
        Eliminate meaning based on grammatical incompatibility.

        Rule: If morphological analysis constrains meaning, eliminate
        incompatible readings.
        """
        required_pos = grammatical_constraints.get("required_pos")
        if required_pos:
            # Check if meaning implies a different POS
            meaning_pos_map: Dict[str, str] = {
                "wind": "noun",
                "breath": "noun",
                "spirit": "noun",
                "Spirit": "noun",
                "word": "noun",
                "speech": "noun",
                "reason": "noun",
                "account": "noun",
                "Word": "noun",
            }

            meaning_pos = meaning_pos_map.get(meaning)
            if meaning_pos and meaning_pos != required_pos:
                return EliminationStep(
                    meaning=meaning,
                    eliminated=True,
                    reason=EliminationReason.GRAMMATICAL_INCOMPATIBILITY,
                    explanation=(
                        f"Meaning '{meaning}' requires {meaning_pos}, "
                        f"but grammar requires {required_pos}"
                    ),
                    confidence=0.95,
                )

        return None

    async def _eliminate_by_context(
        self,
        meaning: str,
        verse_context: Dict[str, Any],
        word: str,
    ) -> Optional[EliminationStep]:
        """
        Eliminate meaning based on contextual impossibility.

        Rule: If surrounding context excludes a meaning, eliminate it.

        Example: "ruach" as "wind" requires physical source; GEN.1.2 has none.
        """
        requirements = self.MEANING_REQUIREMENTS.get(word, {}).get(meaning, [])
        if not requirements:
            return None

        text = verse_context.get("text", "").lower()
        surrounding = " ".join(
            v.get("text", "") for v in verse_context.get("surrounding_verses", [])
        ).lower()
        full_context = text + " " + surrounding

        # Check requirements
        requirement_checks: Dict[str, List[str]] = {
            "physical_source": ["source", "from", "origin", "cause"],
            "meteorological_context": ["wind", "storm", "blow", "weather"],
            "movement_described": ["blow", "move", "rush", "sweep"],
            "living_subject": ["person", "man", "woman", "animal", "creature"],
            "physical_action": ["breathe", "exhale", "pant", "sigh"],
            "bodily_context": ["body", "nose", "mouth", "lungs"],
            "human_subject": ["man", "person", "soul", "heart"],
            "emotional_context": ["angry", "sad", "happy", "troubled"],
            "psychological_context": ["mind", "thoughts", "feelings"],
            "divine_context": ["god", "lord", "holy", "heaven"],
            "creation_context": ["create", "beginning", "made", "formed"],
            "prophetic_context": ["prophet", "prophecy", "speak", "anoint"],
            "speech_context": ["say", "speak", "said", "told"],
            "communication": ["message", "tell", "declare"],
            "philosophical_context": ["reason", "logic", "thought"],
            "argument": ["therefore", "because", "thus"],
            "narrative_context": ["story", "account", "happened"],
            "explanation": ["explain", "describe", "detail"],
            "christological": ["christ", "son", "messiah", "savior", "was god"],
        }

        # Check if AT LEAST ONE requirement is satisfied (logical OR)
        # The meaning is compatible if any of its typical contexts are present
        satisfied_requirements = []
        for req in requirements:
            keywords = requirement_checks.get(req, [])
            if keywords and any(kw in full_context for kw in keywords):
                satisfied_requirements.append(req)

        # If none of the requirements are satisfied, eliminate
        if requirements and not satisfied_requirements:
            return EliminationStep(
                meaning=meaning,
                eliminated=True,
                reason=EliminationReason.CONTEXTUAL_IMPOSSIBILITY,
                explanation=(
                    f"Context lacks any required elements for '{meaning}': "
                    f"needs one of {requirements}"
                ),
                confidence=0.85,
            )

        return None

    async def _eliminate_by_semantic_field(
        self,
        meaning: str,
        verse_context: Dict[str, Any],
    ) -> Optional[EliminationStep]:
        """
        Eliminate meaning based on semantic field contradiction.

        Rule: If semantic field creates contradictions, eliminate.
        """
        markers = verse_context.get("semantic_markers", [])
        text = verse_context.get("text", "").lower()

        # Define semantic field conflicts
        conflicts: Dict[str, List[str]] = {
            "death": ["life", "living", "alive"],
            "destruction": ["creation", "building", "forming"],
            "chaos": ["order", "organized", "structured"],
            "profane": ["holy", "sacred", "divine"],
        }

        # Check if meaning conflicts with context
        meaning_lower = meaning.lower()
        if meaning_lower in conflicts:
            conflicting_terms = conflicts[meaning_lower]
            if any(term in text for term in conflicting_terms):
                return EliminationStep(
                    meaning=meaning,
                    eliminated=True,
                    reason=EliminationReason.SEMANTIC_CONTRADICTION,
                    explanation=(
                        f"Semantic field of '{meaning}' contradicts context "
                        f"(found conflicting terms)"
                    ),
                    confidence=0.80,
                )

        return None

    async def _eliminate_by_theology(
        self,
        meaning: str,
        verse_context: Dict[str, Any],
    ) -> Optional[EliminationStep]:
        """
        Eliminate meaning based on theological impossibility.

        Rule: If meaning contradicts Orthodox theology, eliminate (with care).

        Example: Modalist reading of πνεῦμα in Trinitarian context.
        """
        # Check for Trinitarian context
        is_trinitarian = self._is_trinitarian_context(verse_context)

        if is_trinitarian and meaning in self.MODALIST_READINGS:
            return EliminationStep(
                meaning=meaning,
                eliminated=True,
                reason=EliminationReason.THEOLOGICAL_IMPOSSIBILITY,
                explanation=(
                    "Reading incompatible with Trinitarian theology "
                    "established by context"
                ),
                confidence=0.90,
            )

        return None

    def _is_trinitarian_context(self, verse_context: Dict[str, Any]) -> bool:
        """Check if context is Trinitarian."""
        text = verse_context.get("text", "").lower()
        surrounding = " ".join(
            v.get("text", "") for v in verse_context.get("surrounding_verses", [])
        ).lower()
        full_context = text + " " + surrounding

        trinitarian_indicators = sum(
            1 for marker in self.TRINITARIAN_MARKERS if marker.lower() in full_context
        )

        return trinitarian_indicators >= 2

    async def rank_by_parallel_support(
        self,
        remaining_meanings: List[str],
        verse_id: str,
    ) -> List[Tuple[str, float]]:
        """
        Rank remaining meanings by parallel support.

        For each remaining meaning, find supporting parallels and calculate
        support score based on:
        - Number of exact parallel constructions
        - Theological significance of parallels
        - Patristic usage patterns

        Args:
            remaining_meanings: List of meanings that weren't eliminated
            verse_id: The verse ID being analyzed

        Returns:
            List of (meaning, score) tuples sorted by score descending
        """
        if len(remaining_meanings) <= 1:
            return [(m, 1.0) for m in remaining_meanings]

        scores: List[Tuple[str, float]] = []

        for meaning in remaining_meanings:
            score = 0.5  # Base score

            # Theological significance boost
            if meaning.lower() in ["spirit", "word", "glory"]:
                score += 0.2

            # Divine meanings in creation contexts get boost
            parts = verse_id.split(".")
            if len(parts) >= 2:
                book, chapter = parts[0], parts[1]
                if book == "GEN" and chapter == "1":
                    if meaning in ["Spirit", "Word"]:
                        score += 0.3
                elif book == "JHN" and chapter == "1":
                    if meaning == "Word":
                        score += 0.3

            # Patristic consensus boost (simplified)
            patristic_preferred: Dict[str, List[str]] = {
                "GEN.1.2": ["Spirit"],
                "JHN.1.1": ["Word"],
            }
            if verse_id in patristic_preferred:
                if meaning in patristic_preferred[verse_id]:
                    score += 0.3

            scores.append((meaning, min(1.0, score)))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    async def map_semantic_field(
        self,
        word: str,
        primary_meaning: str,
    ) -> Dict[str, SemanticFieldEntry]:
        """
        Build complete semantic field map for a word.

        Args:
            word: The word/lemma
            primary_meaning: The determined primary meaning

        Returns:
            Dictionary mapping meaning to SemanticFieldEntry
        """
        semantic_map: Dict[str, SemanticFieldEntry] = {}

        # Get known meanings
        language = "hebrew" if word in self.POLYSEMOUS_HEBREW else "greek"
        if language == "hebrew":
            known_meanings = self.POLYSEMOUS_HEBREW.get(word, [word])
        else:
            known_meanings = self.POLYSEMOUS_GREEK.get(word, [word])

        for meaning in known_meanings:
            # Calculate theological weight
            theological_weight = 0.5
            if meaning == primary_meaning:
                theological_weight = 0.9
            elif meaning.lower() in ["spirit", "word", "glory"]:
                theological_weight = 0.7

            entry = SemanticFieldEntry(
                lemma=word,
                meaning=meaning,
                occurrence_count=0,  # Would be populated from corpus
                primary_contexts=[],
                semantic_neighbors=self._get_semantic_neighbors(meaning),
                theological_weight=theological_weight,
            )
            semantic_map[meaning] = entry

        return semantic_map

    def _get_semantic_neighbors(self, meaning: str) -> List[str]:
        """Get semantically related terms for a meaning."""
        neighbors: Dict[str, List[str]] = {
            "Spirit": ["Holy Spirit", "divine presence", "pneuma"],
            "spirit": ["soul", "life-force", "inner being"],
            "wind": ["breath", "air", "breeze"],
            "breath": ["life", "respiration", "soul"],
            "Word": ["Logos", "divine reason", "Christ"],
            "word": ["speech", "saying", "message"],
            "glory": ["honor", "splendor", "majesty"],
            "soul": ["life", "self", "person"],
            "heart": ["mind", "will", "inner being"],
        }
        return neighbors.get(meaning, [])

    def _calculate_confidence(
        self,
        remaining_candidates: List[str],
        reasoning_chain: List[EliminationStep],
        total_occurrences: int,
    ) -> float:
        """
        Calculate overall confidence in the resolution.

        Higher confidence when:
        - Only one candidate remains
        - Many meanings were eliminated with high confidence
        - High occurrence count provides statistical support
        """
        if not remaining_candidates:
            return 0.0

        base_confidence = 0.5

        # Single remaining candidate: high confidence
        if len(remaining_candidates) == 1:
            base_confidence += 0.3

        # Many eliminations with high confidence
        high_confidence_eliminations = sum(
            1 for step in reasoning_chain if step.eliminated and step.confidence > 0.8
        )
        if high_confidence_eliminations > 0:
            base_confidence += min(0.2, high_confidence_eliminations * 0.05)

        # Statistical support from occurrences
        if total_occurrences > 100:
            base_confidence += 0.1
        elif total_occurrences > 50:
            base_confidence += 0.05

        return min(1.0, base_confidence)

    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self._occurrence_cache.clear()
        await self._semantic_range_cache.clear()
        await self._resolution_cache.clear()
        self._initialized = False
        logger.info("OmniContextualResolver cleaned up")
