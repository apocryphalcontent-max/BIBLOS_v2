"""
Golden Record - Complete BIBLOS v2 Output

The authoritative record for a verse containing all analysis results.
"""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

from pipeline.context import ProcessingContext


logger = logging.getLogger(__name__)


@dataclass
class OracleInsights:
    """
    Aggregated insights from the Five Impossible Oracles.
    """
    # From OmniContextual Resolver
    absolute_meanings: Dict[str, str] = field(default_factory=dict)
    meaning_confidence: Dict[str, float] = field(default_factory=dict)

    # From Necessity Calculator
    essential_connections: List[str] = field(default_factory=list)
    necessity_scores: Dict[str, float] = field(default_factory=dict)

    # From LXX Extractor
    christological_insights: List[str] = field(default_factory=list)
    oldest_manuscript_support: Dict[str, str] = field(default_factory=dict)

    # From Fractal Typology
    typological_layers: Dict[str, List[str]] = field(default_factory=dict)
    pattern_signatures: List[str] = field(default_factory=list)

    # From Prophetic Prover
    prophetic_significance: float = 0.0
    fulfillment_evidence: List[str] = field(default_factory=list)
    bayesian_conclusion: Optional[str] = None


@dataclass
class GoldenRecord:
    """
    Complete output of BIBLOS v2 processing.
    The authoritative record for a verse.
    """
    # Identification
    verse_id: str
    book: str
    chapter: int
    verse: int
    testament: str

    # Text
    text_hebrew: Optional[str] = None
    text_greek: Optional[str] = None
    text_english: str = ""

    # Linguistic Analysis
    words: List[Any] = field(default_factory=list)
    resolved_meanings: Dict[int, Any] = field(default_factory=dict)

    # Theological Analysis
    lxx_divergences: List[Any] = field(default_factory=list)
    patristic_interpretations: List[Any] = field(default_factory=list)
    patristic_consensus: float = 0.0
    theological_themes: List[str] = field(default_factory=list)

    # Intertextual Analysis
    typological_connections: List[Any] = field(default_factory=list)
    covenant_position: Optional[Any] = None
    prophetic_data: Optional[Any] = None

    # Cross-References
    cross_references: List[Any] = field(default_factory=list)
    centrality_score: float = 0.0

    # Oracle Insights
    oracle_insights: OracleInsights = field(default_factory=OracleInsights)

    # Metadata
    processing_timestamp: datetime = field(default_factory=lambda: datetime.utcnow())
    pipeline_version: str = "2.0.0"
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    processing_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "verse_id": self.verse_id,
            "book": self.book,
            "chapter": self.chapter,
            "verse": self.verse,
            "testament": self.testament,
            "text_hebrew": self.text_hebrew,
            "text_greek": self.text_greek,
            "text_english": self.text_english,
            "words": [str(w) for w in self.words],
            "resolved_meanings": {k: str(v) for k, v in self.resolved_meanings.items()},
            "lxx_divergences": [str(d) for d in self.lxx_divergences],
            "patristic_interpretations": [str(p) for p in self.patristic_interpretations],
            "patristic_consensus": self.patristic_consensus,
            "theological_themes": self.theological_themes,
            "typological_connections": [str(t) for t in self.typological_connections],
            "cross_references": [str(c) for c in self.cross_references],
            "centrality_score": self.centrality_score,
            "processing_timestamp": self.processing_timestamp.isoformat(),
            "pipeline_version": self.pipeline_version,
            "confidence_scores": self.confidence_scores,
            "processing_duration_ms": self.processing_duration_ms,
        }


class GoldenRecordBuilder:
    """
    Builds GoldenRecord from ProcessingContext.
    """

    def __init__(self, db_client=None, neo4j_client=None):
        """
        Initialize builder with database clients.

        Args:
            db_client: PostgreSQL client for verse data
            neo4j_client: Neo4j client for centrality scores
        """
        self.db_client = db_client
        self.neo4j_client = neo4j_client

    async def build(self, context: ProcessingContext) -> GoldenRecord:
        """
        Build Golden Record from processing context.

        Args:
            context: Completed processing context

        Returns:
            Complete GoldenRecord
        """
        # Parse verse ID
        parts = context.verse_id.split(".")
        book = parts[0]
        chapter = int(parts[1]) if len(parts) > 1 else 1
        verse = int(parts[2]) if len(parts) > 2 else 1

        # Get verse text data (would query database in production)
        verse_data = await self._get_verse_data(context.verse_id)

        return GoldenRecord(
            verse_id=context.verse_id,
            book=book,
            chapter=chapter,
            verse=verse,
            testament=context.testament or self._determine_testament(book),

            text_hebrew=verse_data.get("text_hebrew"),
            text_greek=verse_data.get("text_greek"),
            text_english=verse_data.get("text_english", ""),

            words=self._build_word_analysis(context.linguistic_analysis),
            resolved_meanings=context.linguistic_analysis.get("resolved_meanings", {}),

            lxx_divergences=self._extract_lxx_divergences(context.lxx_analysis),
            patristic_interpretations=context.patristic_witness,
            patristic_consensus=self._calculate_consensus(context.patristic_witness),
            theological_themes=self._extract_themes(context),

            typological_connections=self._build_typological(context.typological_connections),
            covenant_position=self._build_covenant_position(context),
            prophetic_data=self._build_prophetic_data(context.prophetic_analysis),

            cross_references=self._build_cross_refs(context.validated_cross_references),
            centrality_score=await self._get_centrality(context.verse_id),

            oracle_insights=self._build_oracle_insights(context),

            processing_timestamp=datetime.utcnow(),
            pipeline_version="2.0.0",
            confidence_scores=self._aggregate_confidence(context),
            processing_duration_ms=sum(context.phase_durations.values())
        )

    async def _get_verse_data(self, verse_id: str) -> Dict[str, Any]:
        """Get verse text data from database."""
        if self.db_client:
            try:
                return await self.db_client.get_verse(verse_id)
            except Exception as e:
                logger.warning(f"Failed to get verse data for {verse_id}: {e}")

        # Fallback to placeholder
        return {
            "text_hebrew": None,
            "text_greek": None,
            "text_english": f"Verse {verse_id}"
        }

    def _determine_testament(self, book: str) -> str:
        """Determine testament from book code."""
        ot_books = {
            "GEN", "EXO", "LEV", "NUM", "DEU", "JOS", "JDG", "RUT",
            "1SA", "2SA", "1KI", "2KI", "1CH", "2CH", "EZR", "NEH",
            "EST", "JOB", "PSA", "PRO", "ECC", "SNG", "ISA", "JER",
            "LAM", "EZK", "DAN", "HOS", "JOL", "AMO", "OBA", "JON",
            "MIC", "NAH", "HAB", "ZEP", "HAG", "ZEC", "MAL"
        }
        return "OT" if book in ot_books else "NT"

    def _build_word_analysis(self, linguistic_analysis: Dict) -> List[Any]:
        """Build word analysis list."""
        return linguistic_analysis.get("words", [])

    def _extract_lxx_divergences(self, lxx_analysis: Any) -> List[Any]:
        """Extract LXX divergences."""
        if not lxx_analysis:
            return []
        if hasattr(lxx_analysis, "divergences"):
            return lxx_analysis.divergences
        return []

    def _calculate_consensus(self, patristic_witnesses: List) -> float:
        """Calculate patristic consensus score."""
        if not patristic_witnesses:
            return 0.0

        # Simple consensus: ratio of agreement
        # In production, would analyze interpretations for agreement
        return min(1.0, len(patristic_witnesses) / 5.0)

    def _extract_themes(self, context: ProcessingContext) -> List[str]:
        """Extract theological themes."""
        themes = []

        # From typological connections
        for typo in context.typological_connections:
            if hasattr(typo, "themes"):
                themes.extend(typo.themes)

        # From patristic witnesses
        for witness in context.patristic_witness:
            if hasattr(witness, "themes"):
                themes.extend(witness.themes)

        return list(set(themes))

    def _build_typological(self, typological_connections: List) -> List[Any]:
        """Build typological connections list."""
        return typological_connections

    def _build_covenant_position(self, context: ProcessingContext) -> Optional[Any]:
        """Build covenant position."""
        # Would analyze covenant relationships in production
        return None

    def _build_prophetic_data(self, prophetic_analysis: Any) -> Optional[Any]:
        """Build prophetic data."""
        return prophetic_analysis

    def _build_cross_refs(self, validated_cross_references: List) -> List[Any]:
        """Build cross-references list."""
        return validated_cross_references

    async def _get_centrality(self, verse_id: str) -> float:
        """Get centrality score from graph."""
        if self.neo4j_client:
            try:
                result = await self.neo4j_client.execute("""
                    MATCH (v:Verse {id: $verse_id})
                    RETURN v.centrality_score AS centrality
                """, verse_id=verse_id)

                if result and len(result) > 0:
                    return result[0].get("centrality", 0.0)
            except Exception as e:
                logger.warning(f"Failed to get centrality for {verse_id}: {e}")

        return 0.0

    def _build_oracle_insights(self, context: ProcessingContext) -> OracleInsights:
        """Build aggregated oracle insights."""
        insights = OracleInsights()

        # OmniContextual Resolver insights
        resolved_meanings = context.linguistic_analysis.get("resolved_meanings", {})
        for position, result in resolved_meanings.items():
            if hasattr(result, "primary_meaning"):
                insights.absolute_meanings[str(position)] = result.primary_meaning
                insights.meaning_confidence[str(position)] = getattr(result, "confidence", 0.0)

        # Necessity Calculator insights
        for typo in context.typological_connections:
            if hasattr(typo, "necessity_score") and typo.necessity_score > 0.5:
                target = getattr(typo, "antitype_reference", None) or getattr(typo, "target", None)
                if target:
                    insights.essential_connections.append(target)
                    insights.necessity_scores[target] = typo.necessity_score

        # LXX Extractor insights
        if context.lxx_analysis:
            if hasattr(context.lxx_analysis, "primary_christological_insight"):
                insights.christological_insights.append(context.lxx_analysis.primary_christological_insight)

        # Typology insights
        for typo in context.typological_connections:
            if hasattr(typo, "pattern_signature"):
                insights.pattern_signatures.append(typo.pattern_signature)

        # Prophetic insights
        if context.prophetic_analysis:
            if hasattr(context.prophetic_analysis, "posterior_probability"):
                insights.prophetic_significance = context.prophetic_analysis.posterior_probability
            if hasattr(context.prophetic_analysis, "fulfillment_references"):
                insights.fulfillment_evidence = context.prophetic_analysis.fulfillment_references

        return insights

    def _aggregate_confidence(self, context: ProcessingContext) -> Dict[str, float]:
        """Aggregate confidence scores across all phases."""
        scores = {}

        # Linguistic confidence
        if context.linguistic_analysis:
            resolved = context.linguistic_analysis.get("resolved_meanings", {})
            if resolved:
                avg_conf = sum(getattr(r, "confidence", 0.0) for r in resolved.values()) / len(resolved)
                scores["linguistic"] = avg_conf

        # Cross-reference confidence
        if context.validated_cross_references:
            avg_conf = sum(getattr(r, "final_confidence", 0.0) for r in context.validated_cross_references)
            avg_conf /= len(context.validated_cross_references)
            scores["cross_references"] = avg_conf

        # Typological confidence
        if context.typological_connections:
            avg_conf = sum(getattr(t, "composite_strength", 0.0) for t in context.typological_connections)
            avg_conf /= len(context.typological_connections)
            scores["typological"] = avg_conf

        # Overall confidence (average of all)
        if scores:
            scores["overall"] = sum(scores.values()) / len(scores)

        return scores
