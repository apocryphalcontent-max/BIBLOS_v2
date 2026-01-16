"""
BIBLOS v2 - Result Postprocessor

Post-processing and filtering of inference results.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from enum import Enum


class FilterMode(Enum):
    """Filtering strictness modes."""
    STRICT = "strict"       # High confidence only
    MODERATE = "moderate"   # Balanced filtering
    PERMISSIVE = "permissive"  # Include more candidates


@dataclass
class PostprocessConfig:
    """Configuration for result postprocessing."""
    filter_mode: FilterMode = FilterMode.MODERATE
    min_confidence: float = 0.5
    max_results: int = 50
    deduplicate: bool = True
    validate_references: bool = True
    apply_theological_constraints: bool = True
    enrich_metadata: bool = True


class ResultPostprocessor:
    """
    Postprocessor for inference results.

    Applies filtering, validation, and enrichment to raw inference results.
    """

    # Valid connection types
    VALID_CONNECTION_TYPES = {
        "thematic", "verbal", "conceptual", "historical",
        "typological", "prophetic", "liturgical", "narrative",
        "genealogical", "geographical"
    }

    # OT/NT book classifications
    OT_BOOKS = {
        "GEN", "EXO", "LEV", "NUM", "DEU", "JOS", "JDG", "RUT",
        "1SA", "2SA", "1KI", "2KI", "1CH", "2CH", "EZR", "NEH",
        "EST", "JOB", "PSA", "PRO", "ECC", "SNG", "ISA", "JER",
        "LAM", "EZK", "DAN", "HOS", "JOL", "AMO", "OBA", "JON",
        "MIC", "NAH", "HAB", "ZEP", "HAG", "ZEC", "MAL"
    }

    NT_BOOKS = {
        "MAT", "MRK", "LUK", "JHN", "ACT", "ROM", "1CO", "2CO",
        "GAL", "EPH", "PHP", "COL", "1TH", "2TH", "1TI", "2TI",
        "TIT", "PHM", "HEB", "JAS", "1PE", "2PE", "1JN", "2JN",
        "3JN", "JUD", "REV"
    }

    def __init__(self, config: Optional[PostprocessConfig] = None):
        self.config = config or PostprocessConfig()
        self.logger = logging.getLogger("biblos.ml.inference.postprocessor")
        self._known_crossrefs: Set[str] = set()

    def process(
        self,
        results: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process and filter inference results.

        Args:
            results: Raw inference results
            context: Optional pipeline context

        Returns:
            Filtered and enriched results
        """
        if not results:
            return []

        # Apply pipeline
        processed = results

        # 1. Deduplicate
        if self.config.deduplicate:
            processed = self._deduplicate(processed)

        # 2. Filter by confidence
        processed = self._filter_confidence(processed)

        # 3. Validate references
        if self.config.validate_references:
            processed = self._validate_references(processed)

        # 4. Apply theological constraints
        if self.config.apply_theological_constraints:
            processed = self._apply_theological_constraints(processed, context)

        # 5. Enrich metadata
        if self.config.enrich_metadata:
            processed = self._enrich_metadata(processed, context)

        # 6. Sort by confidence
        processed.sort(key=lambda x: x.get("confidence", 0), reverse=True)

        # 7. Limit results
        processed = processed[:self.config.max_results]

        self.logger.info(f"Postprocessed {len(results)} -> {len(processed)} results")
        return processed

    def _deduplicate(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate cross-references."""
        seen = set()
        unique = []

        for result in results:
            source = result.get("source_verse", "")
            target = result.get("target_verse", "")

            # Create canonical key (sorted to handle bidirectional)
            key = tuple(sorted([source, target]))

            if key not in seen:
                seen.add(key)
                unique.append(result)

        return unique

    def _filter_confidence(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter by confidence threshold based on mode."""
        thresholds = {
            FilterMode.STRICT: 0.7,
            FilterMode.MODERATE: 0.5,
            FilterMode.PERMISSIVE: 0.3
        }

        threshold = max(
            thresholds.get(self.config.filter_mode, 0.5),
            self.config.min_confidence
        )

        return [
            r for r in results
            if r.get("confidence", 0) >= threshold
        ]

    def _validate_references(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate reference format and existence."""
        valid = []

        for result in results:
            source = result.get("source_verse", "")
            target = result.get("target_verse", "")

            if self._is_valid_reference(source) and self._is_valid_reference(target):
                # Validate connection type
                conn_type = result.get("connection_type", "thematic")
                if conn_type not in self.VALID_CONNECTION_TYPES:
                    result["connection_type"] = "thematic"

                valid.append(result)
            else:
                self.logger.warning(f"Invalid reference: {source} -> {target}")

        return valid

    def _is_valid_reference(self, ref: str) -> bool:
        """Check if reference format is valid."""
        if not ref or not isinstance(ref, str):
            return False

        # Expected format: BOOK.CHAPTER.VERSE
        parts = ref.upper().replace(" ", ".").replace(":", ".").split(".")

        if len(parts) < 3:
            return False

        book = parts[0]
        if book not in self.OT_BOOKS and book not in self.NT_BOOKS:
            return False

        try:
            chapter = int(parts[1])
            verse = int(parts[2])
            return chapter > 0 and verse > 0
        except ValueError:
            return False

    def _apply_theological_constraints(
        self,
        results: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply theological validation to results."""
        constrained = []

        for result in results:
            source = result.get("source_verse", "")
            target = result.get("target_verse", "")
            conn_type = result.get("connection_type", "thematic")

            # Get book codes
            source_book = source.split(".")[0] if "." in source else ""
            target_book = target.split(".")[0] if "." in target else ""

            source_ot = source_book in self.OT_BOOKS
            target_ot = target_book in self.OT_BOOKS

            # Validate typological connections (OT type -> NT antitype)
            if conn_type == "typological":
                if not source_ot or target_ot:
                    # Typological should be OT->NT, adjust confidence
                    result["confidence"] *= 0.8
                    result["metadata"] = result.get("metadata", {})
                    result["metadata"]["typological_warning"] = "Non-standard direction"

            # Validate prophetic connections
            if conn_type == "prophetic":
                prophetic_books = {"ISA", "JER", "EZK", "DAN", "HOS", "JOL", "AMO",
                                  "OBA", "JON", "MIC", "NAH", "HAB", "ZEP", "HAG",
                                  "ZEC", "MAL", "PSA"}
                if source_book not in prophetic_books:
                    result["confidence"] *= 0.9

            # Boost confidence for known patristic connections
            if context:
                patrologos = context.get("agent_results", {}).get("patrologos", {})
                citations = patrologos.get("data", {}).get("citations", [])

                for citation in citations:
                    refs = citation.get("references", [])
                    if target in refs:
                        result["confidence"] = min(1.0, result["confidence"] * 1.15)
                        result["patristic_support"] = True
                        break

            constrained.append(result)

        return constrained

    def _enrich_metadata(
        self,
        results: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enrich results with additional metadata."""
        for result in results:
            source = result.get("source_verse", "")
            target = result.get("target_verse", "")

            # Add testament info
            source_book = source.split(".")[0] if "." in source else ""
            target_book = target.split(".")[0] if "." in target else ""

            result["source_testament"] = "OT" if source_book in self.OT_BOOKS else "NT"
            result["target_testament"] = "OT" if target_book in self.OT_BOOKS else "NT"
            result["cross_testament"] = result["source_testament"] != result["target_testament"]

            # Add strength classification
            confidence = result.get("confidence", 0.5)
            if confidence >= 0.8:
                result["strength"] = "strong"
            elif confidence >= 0.6:
                result["strength"] = "moderate"
            else:
                result["strength"] = "weak"

            # Add context evidence if available
            if context:
                evidence = []

                # Check syndesmos results
                syndesmos = context.get("agent_results", {}).get("syndesmos", {})
                refs = syndesmos.get("data", {}).get("cross_references", [])
                for ref in refs:
                    if ref.get("target_ref") == target:
                        evidence.append(f"SYNDESMOS: {ref.get('connection_type', 'linked')}")

                # Check typologos results
                typologos = context.get("agent_results", {}).get("typologos", {})
                connections = typologos.get("data", {}).get("connections", [])
                for conn in connections:
                    if conn.get("antitype") == target or conn.get("type") == target:
                        evidence.append(f"TYPOLOGOS: {conn.get('type_category', 'typological')}")

                if evidence:
                    result["agent_evidence"] = evidence

        return results

    def add_known_crossref(self, source: str, target: str) -> None:
        """Add a known cross-reference for validation."""
        key = tuple(sorted([source, target]))
        self._known_crossrefs.add(key)

    def is_known_crossref(self, source: str, target: str) -> bool:
        """Check if a cross-reference is already known."""
        key = tuple(sorted([source, target]))
        return key in self._known_crossrefs

    def filter_novel(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter to only novel (previously unknown) cross-references."""
        novel = []

        for result in results:
            source = result.get("source_verse", "")
            target = result.get("target_verse", "")

            if not self.is_known_crossref(source, target):
                result["is_novel"] = True
                novel.append(result)
            else:
                result["is_novel"] = False

        return novel

    def format_for_export(
        self,
        results: List[Dict[str, Any]],
        format_type: str = "json"
    ) -> Any:
        """Format results for export."""
        if format_type == "json":
            return self._format_json(results)
        elif format_type == "csv":
            return self._format_csv(results)
        elif format_type == "schema":
            return self._format_schema(results)
        else:
            return results

    def _format_json(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format as JSON-compatible structure."""
        formatted = []

        for result in results:
            formatted.append({
                "source_ref": result.get("source_verse", ""),
                "target_ref": result.get("target_verse", ""),
                "connection_type": result.get("connection_type", "thematic"),
                "strength": result.get("strength", "moderate"),
                "confidence": round(result.get("confidence", 0.5), 3),
                "cross_testament": result.get("cross_testament", False),
                "metadata": {
                    "source_testament": result.get("source_testament", ""),
                    "target_testament": result.get("target_testament", ""),
                    "patristic_support": result.get("patristic_support", False),
                    "is_novel": result.get("is_novel", True)
                }
            })

        return formatted

    def _format_csv(
        self,
        results: List[Dict[str, Any]]
    ) -> str:
        """Format as CSV string."""
        lines = ["source_ref,target_ref,connection_type,strength,confidence"]

        for result in results:
            line = ",".join([
                result.get("source_verse", ""),
                result.get("target_verse", ""),
                result.get("connection_type", "thematic"),
                result.get("strength", "moderate"),
                str(round(result.get("confidence", 0.5), 3))
            ])
            lines.append(line)

        return "\n".join(lines)

    def _format_schema(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format to match cross-reference schema."""
        formatted = []

        for result in results:
            formatted.append({
                "source_ref": result.get("source_verse", ""),
                "target_ref": result.get("target_verse", ""),
                "connection_type": result.get("connection_type", "thematic"),
                "strength": result.get("strength", "moderate"),
                "notes": result.get("agent_evidence", []),
                "sources": ["BIBLOS-v2-ML"],
                "verified": False
            })

        return formatted

    def generate_report(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate summary report of processed results."""
        if not results:
            return {"total": 0, "by_type": {}, "by_strength": {}}

        # Count by connection type
        by_type = {}
        for result in results:
            conn_type = result.get("connection_type", "unknown")
            by_type[conn_type] = by_type.get(conn_type, 0) + 1

        # Count by strength
        by_strength = {"strong": 0, "moderate": 0, "weak": 0}
        for result in results:
            strength = result.get("strength", "moderate")
            by_strength[strength] = by_strength.get(strength, 0) + 1

        # Count cross-testament
        cross_testament = sum(1 for r in results if r.get("cross_testament", False))

        # Average confidence
        confidences = [r.get("confidence", 0.5) for r in results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        # Novel discoveries
        novel_count = sum(1 for r in results if r.get("is_novel", True))

        return {
            "total": len(results),
            "by_type": by_type,
            "by_strength": by_strength,
            "cross_testament_count": cross_testament,
            "average_confidence": round(avg_confidence, 3),
            "novel_discoveries": novel_count,
            "novel_percentage": round(novel_count / len(results) * 100, 1) if results else 0
        }
