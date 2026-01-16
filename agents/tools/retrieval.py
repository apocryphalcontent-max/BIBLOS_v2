"""
BIBLOS v2 - Retrieval Tools

LangChain-compatible tools for retrieving patristic citations,
cross-references, and verse text from various sources.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict

from data.schemas import (
    ConnectionType,
    normalize_verse_id,
    validate_verse_id,
)


logger = logging.getLogger("biblos.tools.retrieval")


# =============================================================================
# PATRISTIC CITATION TOOL
# =============================================================================


class PatristicCitationInput(BaseModel):
    """Input schema for patristic citation retrieval."""

    model_config = ConfigDict(extra="forbid")

    verse_ref: str = Field(
        ...,
        description="Verse reference to find patristic citations for (e.g., GEN.1.1)",
        pattern=r"^[A-Z0-9]{3}\.\d+\.\d+$",
    )
    fathers: Optional[List[str]] = Field(
        default=None,
        description="Filter by specific Church Fathers (e.g., ['Chrysostom', 'Augustine'])",
    )
    schools: Optional[List[str]] = Field(
        default=None,
        description="Filter by patristic schools (alexandrian, antiochene, cappadocian, latin_west)",
    )
    interpretation_types: Optional[List[str]] = Field(
        default=None,
        description="Filter by interpretation sense (literal, allegorical, tropological, anagogical)",
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of citations to return",
    )


class PatristicCitationOutput(BaseModel):
    """Output schema for patristic citation retrieval."""

    model_config = ConfigDict(extra="forbid")

    verse_ref: str
    citations: List[Dict[str, Any]]
    total_found: int
    fathers_represented: List[str]
    schools_represented: List[str]


class PatristicCitationTool(BaseTool):
    """
    Tool for retrieving patristic (Church Father) citations.

    Searches the patristic corpus for citations, commentaries, and
    interpretations of biblical verses by Church Fathers.
    """

    name: str = "patristic_citation_lookup"
    description: str = (
        "Retrieve patristic (Church Father) citations and interpretations for a biblical verse. "
        "Returns citations from major Church Fathers like Chrysostom, Augustine, Basil, Gregory, "
        "Cyril, Origen, and others. Includes the interpretive sense (literal, allegorical, etc.) "
        "and the patristic school of interpretation."
    )
    args_schema: Type[BaseModel] = PatristicCitationInput
    return_direct: bool = False

    # Simulated patristic database (in production, this would connect to actual DB)
    _patristic_index: Dict[str, List[Dict[str, Any]]] = {}

    def __init__(self, patristic_index: Optional[Dict] = None, **kwargs):
        super().__init__(**kwargs)
        if patristic_index:
            self._patristic_index = patristic_index

    def _run(
        self,
        verse_ref: str,
        fathers: Optional[List[str]] = None,
        schools: Optional[List[str]] = None,
        interpretation_types: Optional[List[str]] = None,
        max_results: int = 10,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Synchronous execution."""
        return asyncio.run(
            self._arun(verse_ref, fathers, schools, interpretation_types, max_results)
        )

    async def _arun(
        self,
        verse_ref: str,
        fathers: Optional[List[str]] = None,
        schools: Optional[List[str]] = None,
        interpretation_types: Optional[List[str]] = None,
        max_results: int = 10,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Asynchronous execution."""
        try:
            verse_ref = normalize_verse_id(verse_ref)

            # Get citations from index
            citations = self._patristic_index.get(verse_ref, [])

            # Apply filters
            if fathers:
                fathers_lower = [f.lower() for f in fathers]
                citations = [
                    c for c in citations
                    if c.get("father", "").lower() in fathers_lower
                ]

            if schools:
                schools_lower = [s.lower() for s in schools]
                citations = [
                    c for c in citations
                    if c.get("school", "").lower() in schools_lower
                ]

            if interpretation_types:
                types_lower = [t.lower() for t in interpretation_types]
                citations = [
                    c for c in citations
                    if c.get("interpretation_type", "").lower() in types_lower
                ]

            # Limit results
            citations = citations[:max_results]

            # Extract unique fathers and schools
            fathers_found = list(set(c.get("father", "") for c in citations if c.get("father")))
            schools_found = list(set(c.get("school", "") for c in citations if c.get("school")))

            output = PatristicCitationOutput(
                verse_ref=verse_ref,
                citations=citations,
                total_found=len(citations),
                fathers_represented=fathers_found,
                schools_represented=schools_found,
            )

            return output.model_dump_json(indent=2)

        except Exception as e:
            logger.error(f"Patristic citation lookup failed: {e}")
            return f'{{"error": "{str(e)}"}}'


# =============================================================================
# CROSS-REFERENCE LOOKUP TOOL
# =============================================================================


class CrossReferenceLookupInput(BaseModel):
    """Input schema for cross-reference lookup."""

    model_config = ConfigDict(extra="forbid")

    verse_ref: str = Field(
        ...,
        description="Source verse reference (e.g., GEN.1.1)",
        pattern=r"^[A-Z0-9]{3}\.\d+\.\d+$",
    )
    connection_types: Optional[List[str]] = Field(
        default=None,
        description="Filter by connection types (typological, prophetic, verbal, thematic, etc.)",
    )
    strength: Optional[str] = Field(
        default=None,
        description="Filter by connection strength (strong, moderate, weak)",
        pattern="^(strong|moderate|weak)$",
    )
    testament_filter: Optional[str] = Field(
        default=None,
        description="Filter target verses by testament (OT, NT)",
        pattern="^(OT|NT)$",
    )
    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold",
    )
    max_results: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of results",
    )


class CrossReferenceLookupOutput(BaseModel):
    """Output schema for cross-reference lookup."""

    model_config = ConfigDict(extra="forbid")

    source_ref: str
    cross_references: List[Dict[str, Any]]
    total_found: int
    connection_types_found: List[str]
    avg_confidence: float


class CrossReferenceLookupTool(BaseTool):
    """
    Tool for looking up cross-references from the curated database.

    Searches the existing cross-reference database for connections
    to a given verse, with filtering by type and strength.
    """

    name: str = "cross_reference_lookup"
    description: str = (
        "Look up existing cross-references for a biblical verse from the curated database. "
        "Returns connections with their types (typological, prophetic, verbal, thematic), "
        "strength ratings, and supporting evidence. Use this to find known connections "
        "before discovering new ones."
    )
    args_schema: Type[BaseModel] = CrossReferenceLookupInput
    return_direct: bool = False

    _crossref_index: Dict[str, List[Dict[str, Any]]] = {}

    def __init__(self, crossref_index: Optional[Dict] = None, **kwargs):
        super().__init__(**kwargs)
        if crossref_index:
            self._crossref_index = crossref_index

    def _run(
        self,
        verse_ref: str,
        connection_types: Optional[List[str]] = None,
        strength: Optional[str] = None,
        testament_filter: Optional[str] = None,
        min_confidence: float = 0.5,
        max_results: int = 20,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Synchronous execution."""
        return asyncio.run(
            self._arun(
                verse_ref, connection_types, strength,
                testament_filter, min_confidence, max_results
            )
        )

    async def _arun(
        self,
        verse_ref: str,
        connection_types: Optional[List[str]] = None,
        strength: Optional[str] = None,
        testament_filter: Optional[str] = None,
        min_confidence: float = 0.5,
        max_results: int = 20,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Asynchronous execution."""
        try:
            verse_ref = normalize_verse_id(verse_ref)

            # OT books for testament filtering
            ot_books = {
                "GEN", "EXO", "LEV", "NUM", "DEU", "JOS", "JDG", "RUT",
                "1SA", "2SA", "1KI", "2KI", "1CH", "2CH", "EZR", "NEH",
                "EST", "JOB", "PSA", "PRO", "ECC", "SNG", "ISA", "JER",
                "LAM", "EZK", "DAN", "HOS", "JOL", "AMO", "OBA", "JON",
                "MIC", "NAH", "HAB", "ZEP", "HAG", "ZEC", "MAL"
            }

            # Get cross-references from index
            refs = self._crossref_index.get(verse_ref, [])

            # Apply filters
            if connection_types:
                types_lower = [t.lower() for t in connection_types]
                refs = [
                    r for r in refs
                    if r.get("connection_type", "").lower() in types_lower
                ]

            if strength:
                refs = [
                    r for r in refs
                    if r.get("strength", "").lower() == strength.lower()
                ]

            if testament_filter:
                if testament_filter == "OT":
                    refs = [
                        r for r in refs
                        if r.get("target_ref", "").split(".")[0] in ot_books
                    ]
                else:  # NT
                    refs = [
                        r for r in refs
                        if r.get("target_ref", "").split(".")[0] not in ot_books
                    ]

            # Filter by confidence
            refs = [r for r in refs if r.get("confidence", 0) >= min_confidence]

            # Sort by confidence descending
            refs.sort(key=lambda x: x.get("confidence", 0), reverse=True)

            # Limit results
            refs = refs[:max_results]

            # Calculate stats
            connection_types_found = list(set(
                r.get("connection_type", "") for r in refs
                if r.get("connection_type")
            ))
            avg_confidence = (
                sum(r.get("confidence", 0) for r in refs) / len(refs)
                if refs else 0.0
            )

            output = CrossReferenceLookupOutput(
                source_ref=verse_ref,
                cross_references=refs,
                total_found=len(refs),
                connection_types_found=connection_types_found,
                avg_confidence=avg_confidence,
            )

            return output.model_dump_json(indent=2)

        except Exception as e:
            logger.error(f"Cross-reference lookup failed: {e}")
            return f'{{"error": "{str(e)}"}}'


# =============================================================================
# VERSE LOOKUP TOOL
# =============================================================================


class VerseLookupInput(BaseModel):
    """Input schema for verse text lookup."""

    model_config = ConfigDict(extra="forbid")

    verse_refs: List[str] = Field(
        ...,
        description="List of verse references to look up (e.g., ['GEN.1.1', 'JHN.1.1'])",
        min_length=1,
        max_length=50,
    )
    version: str = Field(
        default="default",
        description="Text version to retrieve (default, lxx, mt, byzantine)",
    )
    include_context: bool = Field(
        default=False,
        description="Include surrounding verses for context",
    )
    context_size: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Number of verses before and after to include",
    )


class VerseLookupOutput(BaseModel):
    """Output schema for verse lookup."""

    model_config = ConfigDict(extra="forbid")

    verses: List[Dict[str, Any]]
    found_count: int
    not_found: List[str]
    version: str


class VerseLookupTool(BaseTool):
    """
    Tool for looking up verse text.

    Retrieves verse text in various versions (original languages,
    translations) with optional surrounding context.
    """

    name: str = "verse_text_lookup"
    description: str = (
        "Look up the text of biblical verses. Can retrieve multiple verses at once "
        "and optionally include surrounding verses for context. Supports different "
        "text versions including original Greek (LXX, Byzantine) and Hebrew (MT)."
    )
    args_schema: Type[BaseModel] = VerseLookupInput
    return_direct: bool = False

    _verse_database: Dict[str, Dict[str, Any]] = {}

    def __init__(self, verse_database: Optional[Dict] = None, **kwargs):
        super().__init__(**kwargs)
        if verse_database:
            self._verse_database = verse_database

    def _run(
        self,
        verse_refs: List[str],
        version: str = "default",
        include_context: bool = False,
        context_size: int = 2,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Synchronous execution."""
        return asyncio.run(
            self._arun(verse_refs, version, include_context, context_size)
        )

    async def _arun(
        self,
        verse_refs: List[str],
        version: str = "default",
        include_context: bool = False,
        context_size: int = 2,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Asynchronous execution."""
        try:
            verses = []
            not_found = []

            for ref in verse_refs:
                ref = normalize_verse_id(ref)

                verse_data = self._verse_database.get(ref)
                if verse_data:
                    result = {
                        "reference": ref,
                        "text": verse_data.get(version, verse_data.get("default", "")),
                        "book": ref.split(".")[0],
                        "chapter": int(ref.split(".")[1]),
                        "verse": int(ref.split(".")[2]),
                    }

                    if include_context:
                        result["context"] = self._get_context(ref, context_size, version)

                    verses.append(result)
                else:
                    not_found.append(ref)

            output = VerseLookupOutput(
                verses=verses,
                found_count=len(verses),
                not_found=not_found,
                version=version,
            )

            return output.model_dump_json(indent=2)

        except Exception as e:
            logger.error(f"Verse lookup failed: {e}")
            return f'{{"error": "{str(e)}"}}'

    def _get_context(
        self,
        verse_ref: str,
        context_size: int,
        version: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get surrounding context verses."""
        parts = verse_ref.split(".")
        book, chapter, verse = parts[0], int(parts[1]), int(parts[2])

        preceding = []
        following = []

        # Get preceding verses
        for i in range(context_size, 0, -1):
            prev_verse = verse - i
            if prev_verse > 0:
                prev_ref = f"{book}.{chapter}.{prev_verse}"
                if prev_ref in self._verse_database:
                    data = self._verse_database[prev_ref]
                    preceding.append({
                        "reference": prev_ref,
                        "text": data.get(version, data.get("default", "")),
                    })

        # Get following verses
        for i in range(1, context_size + 1):
            next_verse = verse + i
            next_ref = f"{book}.{chapter}.{next_verse}"
            if next_ref in self._verse_database:
                data = self._verse_database[next_ref]
                following.append({
                    "reference": next_ref,
                    "text": data.get(version, data.get("default", "")),
                })

        return {
            "preceding": preceding,
            "following": following,
        }


# =============================================================================
# TOOL FACTORY
# =============================================================================


def create_retrieval_tools(
    patristic_index: Optional[Dict] = None,
    crossref_index: Optional[Dict] = None,
    verse_database: Optional[Dict] = None,
) -> List[BaseTool]:
    """
    Factory function to create all retrieval tools.

    Args:
        patristic_index: Optional index of patristic citations
        crossref_index: Optional index of cross-references
        verse_database: Optional verse text database

    Returns:
        List of configured retrieval tools
    """
    return [
        PatristicCitationTool(patristic_index=patristic_index),
        CrossReferenceLookupTool(crossref_index=crossref_index),
        VerseLookupTool(verse_database=verse_database),
    ]


# =============================================================================
# TOOL DESCRIPTIONS FOR AGENT PROMPTS
# =============================================================================


RETRIEVAL_TOOL_DESCRIPTIONS = """
Available Retrieval Tools:

1. patristic_citation_lookup: Find Church Father citations for a verse
   - Input: verse_ref, fathers (optional), schools (optional), interpretation_types
   - Returns: Patristic citations with father, work, quote, and interpretation type
   - Fathers: Chrysostom, Augustine, Basil, Gregory (of Nyssa, Nazianzen),
              Cyril, Athanasius, Origen, Jerome, Ephrem
   - Schools: alexandrian, antiochene, cappadocian, latin_west, syrian
   - Interpretation Types: literal, allegorical, tropological, anagogical

2. cross_reference_lookup: Find existing cross-references in the database
   - Input: verse_ref, connection_types, strength, testament_filter, min_confidence
   - Returns: Cross-references with type, strength, confidence, and notes
   - Connection Types: typological, prophetic, verbal, thematic, conceptual,
                       historical, liturgical, narrative, genealogical, geographical
   - Strength: strong, moderate, weak

3. verse_text_lookup: Get verse text and surrounding context
   - Input: verse_refs (list), version, include_context, context_size
   - Returns: Verse text with optional preceding and following verses
   - Versions: default, lxx (Septuagint), mt (Masoretic), byzantine

Orthodox Hermeneutical Note:
- When analyzing Scripture, consider all four senses (literal, allegorical,
  tropological, anagogical) as taught by the Church Fathers
- Prioritize patristic consensus (consensus patrum) when interpreting
- The Septuagint (LXX) is the Old Testament of the Orthodox Church
"""
