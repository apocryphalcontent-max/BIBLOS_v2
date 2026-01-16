"""
BIBLOS v2 - HARMONIKOS Agent

Parallel passage harmonization and analysis.
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from agents.base import (
    BaseExtractionAgent,
    AgentConfig,
    ExtractionResult,
    ExtractionType,
    ProcessingStatus
)


class ParallelType(Enum):
    """Types of parallel passages."""
    SYNOPTIC = "synoptic"  # Gospel synoptic parallels
    DUPLICATE = "duplicate"  # Same event in different books
    VARIANT = "variant"  # Variant accounts
    COMPLEMENTARY = "complementary"  # Complementary accounts
    SEQUENTIAL = "sequential"  # Sequential narrative
    ANTHOLOGY = "anthology"  # Collected sayings/material


class HarmonyLevel(Enum):
    """Level of harmony between passages."""
    VERBATIM = "verbatim"  # Word-for-word
    SUBSTANTIAL = "substantial"  # Mostly same
    PARTIAL = "partial"  # Some overlap
    THEMATIC = "thematic"  # Same theme only


@dataclass
class ParallelPassage:
    """A parallel passage record."""
    primary_ref: str
    parallel_ref: str
    parallel_type: ParallelType
    harmony_level: HarmonyLevel
    shared_content: List[str]
    differences: List[str]
    unique_to_primary: List[str]
    unique_to_parallel: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_ref": self.primary_ref,
            "parallel_ref": self.parallel_ref,
            "parallel_type": self.parallel_type.value,
            "harmony_level": self.harmony_level.value,
            "shared_content": self.shared_content,
            "differences": self.differences,
            "unique_to_primary": self.unique_to_primary,
            "unique_to_parallel": self.unique_to_parallel
        }


class HarmonikosAgent(BaseExtractionAgent):
    """
    HARMONIKOS - Parallel passage analysis agent.

    Performs:
    - Synoptic parallel identification
    - Variant analysis
    - Harmony construction
    - Difference cataloging
    - Complementary reading synthesis
    """

    # Gospel synoptic sections
    SYNOPTIC_SECTIONS = {
        "birth_narratives": {
            "matthew": ["MAT.1", "MAT.2"],
            "luke": ["LUK.1", "LUK.2"]
        },
        "baptism": {
            "matthew": ["MAT.3.13-17"],
            "mark": ["MRK.1.9-11"],
            "luke": ["LUK.3.21-22"],
            "john": ["JHN.1.29-34"]
        },
        "temptation": {
            "matthew": ["MAT.4.1-11"],
            "mark": ["MRK.1.12-13"],
            "luke": ["LUK.4.1-13"]
        },
        "sermon_mount_plain": {
            "matthew": ["MAT.5", "MAT.6", "MAT.7"],
            "luke": ["LUK.6.17-49"]
        },
        "lords_prayer": {
            "matthew": ["MAT.6.9-13"],
            "luke": ["LUK.11.2-4"]
        },
        "passion": {
            "matthew": ["MAT.26", "MAT.27"],
            "mark": ["MRK.14", "MRK.15"],
            "luke": ["LUK.22", "LUK.23"],
            "john": ["JHN.18", "JHN.19"]
        },
        "resurrection": {
            "matthew": ["MAT.28"],
            "mark": ["MRK.16"],
            "luke": ["LUK.24"],
            "john": ["JHN.20", "JHN.21"]
        }
    }

    # OT parallel sections
    OT_PARALLELS = {
        "samuel_chronicles": {
            "2SA": ["1CH.11-29"],
            "1KI": ["2CH.1-9"]
        },
        "kings_chronicles": {
            "1KI": ["2CH"],
            "2KI": ["2CH"]
        },
        "psalms": {
            "PSA.14": ["PSA.53"],
            "PSA.18": ["2SA.22"],
            "PSA.40.13-17": ["PSA.70"],
            "PSA.57.7-11": ["PSA.108.1-5"]
        }
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="harmonikos",
                extraction_type=ExtractionType.INTERTEXTUAL,
                batch_size=200,
                min_confidence=0.65
            )
        super().__init__(config)
        self.logger = logging.getLogger("biblos.agents.harmonikos")

    async def extract(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any]
    ) -> ExtractionResult:
        """Extract parallel passage analysis."""
        # Identify parallel passages
        parallels = self._identify_parallels(verse_id, context)

        # Analyze harmony levels
        analyzed = self._analyze_harmony(parallels, text, context)

        # Build harmony table
        harmony_table = self._build_harmony_table(analyzed)

        # Identify unique elements
        unique = self._identify_unique_elements(analyzed, text)

        # Assess theological significance
        significance = self._assess_significance(analyzed)

        data = {
            "parallels": [p.to_dict() for p in analyzed],
            "harmony_table": harmony_table,
            "unique_elements": unique,
            "synoptic_section": self._identify_synoptic_section(verse_id),
            "theological_significance": significance,
            "harmony_score": self._calculate_harmony_score(analyzed)
        }

        confidence = self._calculate_confidence(analyzed)

        return ExtractionResult(
            agent_name=self.config.name,
            extraction_type=self.config.extraction_type,
            verse_id=verse_id,
            status=ProcessingStatus.COMPLETED,
            data=data,
            confidence=confidence
        )

    def _identify_parallels(
        self,
        verse_id: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify parallel passages for verse."""
        parallels = []
        book = verse_id.split(".")[0]

        # Check synoptic parallels
        for section, gospels in self.SYNOPTIC_SECTIONS.items():
            for gospel, refs in gospels.items():
                for ref in refs:
                    if verse_id.startswith(ref.split("-")[0][:7]):
                        # Found a synoptic section, add other gospels
                        for other_gospel, other_refs in gospels.items():
                            if other_gospel != gospel:
                                parallels.append({
                                    "section": section,
                                    "ref": other_refs[0],
                                    "type": "synoptic"
                                })

        # Check context for known parallels
        context_parallels = context.get("parallel_passages", [])
        for p in context_parallels:
            parallels.append({
                "section": "context",
                "ref": p.get("reference", p.get("ref", "")),
                "type": p.get("type", "duplicate")
            })

        return parallels

    def _analyze_harmony(
        self,
        parallels: List[Dict[str, Any]],
        text: str,
        context: Dict[str, Any]
    ) -> List[ParallelPassage]:
        """Analyze harmony between passages."""
        analyzed = []
        verse_id = context.get("verse_id", "")

        for parallel in parallels:
            parallel_type = self._determine_parallel_type(parallel["type"])
            harmony_level = self._assess_harmony_level(text, parallel, context)

            analyzed.append(ParallelPassage(
                primary_ref=verse_id,
                parallel_ref=parallel["ref"],
                parallel_type=parallel_type,
                harmony_level=harmony_level,
                shared_content=self._extract_shared(text, parallel, context),
                differences=self._extract_differences(text, parallel, context),
                unique_to_primary=[],
                unique_to_parallel=[]
            ))

        return analyzed

    def _determine_parallel_type(self, type_str: str) -> ParallelType:
        """Determine parallel type from string."""
        type_map = {
            "synoptic": ParallelType.SYNOPTIC,
            "duplicate": ParallelType.DUPLICATE,
            "variant": ParallelType.VARIANT,
            "complementary": ParallelType.COMPLEMENTARY,
            "sequential": ParallelType.SEQUENTIAL,
            "anthology": ParallelType.ANTHOLOGY
        }
        return type_map.get(type_str.lower(), ParallelType.DUPLICATE)

    def _assess_harmony_level(
        self,
        text: str,
        parallel: Dict[str, Any],
        context: Dict[str, Any]
    ) -> HarmonyLevel:
        """Assess harmony level between passages."""
        # Check if parallel text is in context
        parallel_text = context.get("parallel_texts", {}).get(
            parallel["ref"], ""
        )

        if not parallel_text:
            return HarmonyLevel.THEMATIC

        # Simple word overlap analysis
        text_words = set(text.lower().split())
        parallel_words = set(parallel_text.lower().split())

        overlap = len(text_words & parallel_words)
        total = len(text_words | parallel_words)

        if total == 0:
            return HarmonyLevel.THEMATIC

        ratio = overlap / total

        if ratio > 0.9:
            return HarmonyLevel.VERBATIM
        elif ratio > 0.6:
            return HarmonyLevel.SUBSTANTIAL
        elif ratio > 0.3:
            return HarmonyLevel.PARTIAL
        else:
            return HarmonyLevel.THEMATIC

    def _extract_shared(
        self,
        text: str,
        parallel: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[str]:
        """Extract shared content between passages."""
        shared = []
        parallel_text = context.get("parallel_texts", {}).get(
            parallel["ref"], ""
        )

        if parallel_text:
            text_words = set(text.lower().split())
            parallel_words = set(parallel_text.lower().split())
            shared = list(text_words & parallel_words)

        return shared[:10]  # Limit to top 10

    def _extract_differences(
        self,
        text: str,
        parallel: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[str]:
        """Extract differences between passages."""
        differences = []
        parallel_text = context.get("parallel_texts", {}).get(
            parallel["ref"], ""
        )

        if parallel_text:
            text_words = set(text.lower().split())
            parallel_words = set(parallel_text.lower().split())
            differences = list(text_words ^ parallel_words)  # Symmetric diff

        return differences[:10]

    def _build_harmony_table(
        self,
        analyzed: List[ParallelPassage]
    ) -> Dict[str, Any]:
        """Build harmony comparison table."""
        if not analyzed:
            return {"columns": [], "rows": []}

        columns = ["Primary"]
        for p in analyzed:
            columns.append(p.parallel_ref)

        return {
            "columns": columns,
            "parallel_count": len(analyzed),
            "harmony_levels": [p.harmony_level.value for p in analyzed]
        }

    def _identify_unique_elements(
        self,
        analyzed: List[ParallelPassage],
        text: str
    ) -> Dict[str, Any]:
        """Identify unique elements in primary text."""
        all_shared = set()
        for p in analyzed:
            all_shared.update(p.shared_content)

        text_words = set(text.lower().split())
        unique = text_words - all_shared

        return {
            "unique_words": list(unique)[:20],
            "unique_count": len(unique),
            "shared_count": len(all_shared)
        }

    def _identify_synoptic_section(
        self,
        verse_id: str
    ) -> Optional[str]:
        """Identify which synoptic section verse belongs to."""
        for section, gospels in self.SYNOPTIC_SECTIONS.items():
            for gospel, refs in gospels.items():
                for ref in refs:
                    if verse_id.startswith(ref.split("-")[0][:7]):
                        return section
        return None

    def _assess_significance(
        self,
        analyzed: List[ParallelPassage]
    ) -> Dict[str, Any]:
        """Assess theological significance of parallels."""
        if not analyzed:
            return {
                "has_parallels": False,
                "significance": "low"
            }

        # Higher significance for synoptic parallels with differences
        has_synoptic = any(
            p.parallel_type == ParallelType.SYNOPTIC for p in analyzed
        )
        has_differences = any(len(p.differences) > 0 for p in analyzed)

        if has_synoptic and has_differences:
            significance = "high"
        elif has_synoptic or has_differences:
            significance = "medium"
        else:
            significance = "low"

        return {
            "has_parallels": True,
            "significance": significance,
            "synoptic": has_synoptic,
            "has_variants": has_differences
        }

    def _calculate_harmony_score(
        self,
        analyzed: List[ParallelPassage]
    ) -> float:
        """Calculate overall harmony score."""
        if not analyzed:
            return 0.0

        level_scores = {
            HarmonyLevel.VERBATIM: 1.0,
            HarmonyLevel.SUBSTANTIAL: 0.75,
            HarmonyLevel.PARTIAL: 0.5,
            HarmonyLevel.THEMATIC: 0.25
        }

        scores = [level_scores[p.harmony_level] for p in analyzed]
        return sum(scores) / len(scores)

    def _calculate_confidence(
        self,
        analyzed: List[ParallelPassage]
    ) -> float:
        """Calculate confidence score."""
        confidence = 0.5

        if analyzed:
            confidence += 0.2
            # Boost for higher harmony levels
            high_harmony = sum(
                1 for p in analyzed
                if p.harmony_level in [HarmonyLevel.VERBATIM, HarmonyLevel.SUBSTANTIAL]
            )
            confidence += high_harmony * 0.1

        return min(1.0, confidence)

    async def validate(self, result: ExtractionResult) -> bool:
        """Validate extraction result."""
        data = result.data
        return "parallels" in data and "harmony_table" in data

    def get_dependencies(self) -> List[str]:
        """Get agent dependencies."""
        return ["grammateus", "syndesmos"]
