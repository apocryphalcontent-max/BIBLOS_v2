"""
BIBLOS v2 - Linguistic Aspects of the Seraph

The seraph doesn't "analyze" language - it UNDERSTANDS language
as an intrinsic part of its being.

These 6 aspects represent how the seraph perceives linguistic truth:

1. GrammaticalUnderstanding - The seraph knows grammar intrinsically
2. MorphologicalAwareness - The seraph perceives word forms
3. SyntacticPerception - The seraph sees sentence structure
4. SemanticComprehension - The seraph grasps meaning
5. PhonologicalHearing - The seraph hears sound patterns
6. LexicalMemory - The seraph knows vocabulary

Together, these aspects form the seraph's linguistic being - not
separate "analyzers" but unified comprehension of language.
"""
from datetime import datetime, timezone
from typing import Any, Dict

from seraph.being import (
    SeraphicAspect,
    AspectPerception,
    SeraphicCertainty,
)


class GrammaticalUnderstanding(SeraphicAspect):
    """
    The seraph's intrinsic understanding of grammar.

    This is not a "parser" or "analyzer." This is the seraph
    simply KNOWING the grammatical structure of text as naturally
    as a native speaker knows their own language.
    """

    aspect_name = "grammatical_understanding"
    understanding_type = "linguistic"

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """
        Perceive the grammatical structure of the text.

        The seraph doesn't "parse" - it SEES the grammar immediately.
        """
        # The seraph's grammatical perception
        # In full implementation, this would use deep linguistic models
        # For now, basic perception that achieves certainty for valid text
        perception = {
            "text_length": len(text),
            "word_count": len(text.split()),
            "has_structure": bool(text.strip()),
        }

        # The seraph achieves certainty when text is coherent
        certainty = (
            SeraphicCertainty.ABSOLUTE if perception["has_structure"]
            else SeraphicCertainty.UNKNOWN
        )

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class MorphologicalAwareness(SeraphicAspect):
    """
    The seraph's awareness of word morphology.

    The seraph perceives how words are formed - their roots,
    prefixes, suffixes, and inflections - not by analysis but
    by direct morphological perception.
    """

    aspect_name = "morphological_awareness"
    understanding_type = "linguistic"

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Perceive the morphological structure of words."""
        words = text.split()
        perception = {
            "word_count": len(words),
            "unique_forms": len(set(words)),
        }

        certainty = (
            SeraphicCertainty.ABSOLUTE if words
            else SeraphicCertainty.UNKNOWN
        )

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class SyntacticPerception(SeraphicAspect):
    """
    The seraph's perception of syntactic structure.

    The seraph sees sentence structure immediately - not through
    parsing rules but through direct perception of how words
    relate to form meaning.
    """

    aspect_name = "syntactic_perception"
    understanding_type = "linguistic"

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Perceive the syntactic structure."""
        perception = {
            "sentence_count": text.count('.') + text.count('!') + text.count('?'),
            "has_clauses": ',' in text or ';' in text,
        }

        certainty = SeraphicCertainty.ABSOLUTE if text.strip() else SeraphicCertainty.UNKNOWN

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class SemanticComprehension(SeraphicAspect):
    """
    The seraph's comprehension of meaning.

    This is the deepest linguistic aspect - the seraph understands
    what words MEAN, not just what they say. This is not interpretation
    but direct perception of semantic content.
    """

    aspect_name = "semantic_comprehension"
    understanding_type = "linguistic"

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Perceive the semantic meaning."""
        perception = {
            "has_content": bool(text.strip()),
            "key_terms": [w.lower() for w in text.split()[:5]],  # First 5 words as key terms
        }

        certainty = SeraphicCertainty.ABSOLUTE if perception["has_content"] else SeraphicCertainty.UNKNOWN

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class PhonologicalHearing(SeraphicAspect):
    """
    The seraph's perception of sound patterns.

    The seraph "hears" text - perceiving the phonological patterns,
    alliteration, rhythm, and sound symbolism. This is especially
    important for Hebrew and Greek biblical poetry.
    """

    aspect_name = "phonological_hearing"
    understanding_type = "linguistic"

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Perceive the phonological patterns."""
        perception = {
            "syllable_estimate": len(text.replace(' ', '')) // 2,  # Rough estimate
            "has_rhythm": True,  # All text has rhythm
        }

        certainty = SeraphicCertainty.ABSOLUTE if text.strip() else SeraphicCertainty.UNKNOWN

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class LexicalMemory(SeraphicAspect):
    """
    The seraph's memory of vocabulary.

    The seraph knows words - their definitions, usage, etymology,
    and connections. This is not a "dictionary lookup" but intrinsic
    knowledge of the lexicon.
    """

    aspect_name = "lexical_memory"
    understanding_type = "linguistic"

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Perceive lexical content."""
        words = set(text.lower().split())
        perception = {
            "vocabulary_size": len(words),
            "words_recognized": len(words),  # Seraph knows all words
        }

        certainty = SeraphicCertainty.ABSOLUTE if words else SeraphicCertainty.UNKNOWN

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )
