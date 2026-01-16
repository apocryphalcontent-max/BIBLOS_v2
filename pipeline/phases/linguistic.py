"""
Linguistic Phase

Phase 1: Linguistic analysis including OmniContextual resolution.
"""
import time
import logging
from typing import List

from pipeline.phases.base import Phase, PhasePriority, PhaseCategory, PhaseDependency
from pipeline.context import ProcessingContext


logger = logging.getLogger(__name__)


class LinguisticPhase(Phase):
    """
    Phase 1: Linguistic analysis including OmniContextual resolution.
    """
    name = "linguistic"
    category = PhaseCategory.LINGUISTIC
    priority = PhasePriority.CRITICAL
    is_critical = True
    base_timeout_seconds = 45.0

    @property
    def dependencies(self) -> List[PhaseDependency]:
        return []  # First phase - no dependencies

    @property
    def outputs(self) -> List[str]:
        return ["linguistic_analysis", "resolved_meanings"]

    async def execute(self, context: ProcessingContext) -> ProcessingContext:
        """
        Execute linguistic analysis phase.

        Steps:
        1. Get verse words from corpus
        2. Run OmniContextual Resolver for polysemous words
        3. Get morphology and syntax analysis
        4. Store results in context
        """
        start_time = time.time()

        # Get verse text  (would query database in production)
        verse_id = context.verse_id
        parts = verse_id.split(".")
        testament = self._determine_testament(parts[0]) if len(parts) > 0 else "NT"
        context.testament = testament

        # Determine language
        language = "hebrew" if testament == "OT" else "greek"

        # Get verse words (placeholder - would use corpus integration)
        words = await self._get_verse_words(verse_id, language)

        # Run OmniContextual Resolver for polysemous words
        resolved_meanings = {}
        for i, word in enumerate(words):
            if await self._is_polysemous(word.get("lemma", ""), language):
                try:
                    result = await self.orchestrator.omni_resolver.resolve_absolute_meaning(
                        word=word.get("surface", ""),
                        verse_id=context.verse_id,
                        language=language
                    )
                    resolved_meanings[i] = result

                    # Emit resolution event if event publisher available
                    if hasattr(self.orchestrator, 'event_publisher'):
                        from db.events import OmniResolutionComputed
                        await self.orchestrator.event_publisher.publish(
                            OmniResolutionComputed(
                                aggregate_id=context.verse_id,
                                correlation_id=context.correlation_id,
                                verse_id=context.verse_id,
                                word=word.get("surface", ""),
                                language=language,
                                primary_meaning=result.primary_meaning if hasattr(result, "primary_meaning") else "",
                                total_occurrences=getattr(result, "total_occurrences", 0),
                                confidence=getattr(result, "confidence", 0.0),
                                semantic_field_map=getattr(result, "semantic_field_map", {})
                            )
                        )
                except Exception as e:
                    logger.warning(f"Failed to resolve word {word.get('surface')}: {e}")
                    context.add_warning(self.name, f"Failed to resolve word: {e}")

        # Get morphology and syntax (placeholders)
        morphology = await self._get_morphology(verse_id)
        syntax = await self._get_syntax(verse_id)

        # Store in context
        context.linguistic_analysis = {
            "words": words,
            "resolved_meanings": resolved_meanings,
            "morphology": morphology,
            "syntax": syntax,
            "language": language
        }

        duration_ms = (time.time() - start_time) * 1000
        context.phase_durations[self.name] = duration_ms

        logger.info(f"Linguistic analysis completed for {verse_id} in {duration_ms:.0f}ms")
        return context

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

    async def _get_verse_words(self, verse_id: str, language: str) -> List[dict]:
        """Get verse words from corpus."""
        # Placeholder - would integrate with Text-Fabric or Macula in production
        # For now, return mock data
        return [
            {"position": 0, "surface": "word1", "lemma": "lemma1", "part_of_speech": "noun"},
            {"position": 1, "surface": "word2", "lemma": "lemma2", "part_of_speech": "verb"},
        ]

    async def _is_polysemous(self, lemma: str, language: str) -> bool:
        """Check if a word is polysemous (has multiple meanings)."""
        # Placeholder - would check lexicon in production
        # For now, simple heuristic
        return len(lemma) > 3  # Simplistic placeholder

    async def _get_morphology(self, verse_id: str) -> dict:
        """Get morphological analysis."""
        # Placeholder
        return {"analyzed": True}

    async def _get_syntax(self, verse_id: str) -> dict:
        """Get syntactic analysis."""
        # Placeholder
        return {"parsed": True}
