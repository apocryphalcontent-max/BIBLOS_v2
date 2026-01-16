"""
BIBLOS v2 - PARADEIGMA Agent

Example and precedent identification in biblical texts.
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


class ExampleType(Enum):
    """Types of biblical examples."""
    POSITIVE = "positive"  # Example to follow
    NEGATIVE = "negative"  # Example to avoid
    TYPOLOGICAL = "typological"  # Type pointing forward
    HISTORICAL = "historical"  # Historical precedent
    RHETORICAL = "rhetorical"  # Rhetorical illustration
    PARABOLIC = "parabolic"  # Parable/story example


class PrecedentCategory(Enum):
    """Categories of precedent."""
    FAITH = "faith"  # Faith examples
    OBEDIENCE = "obedience"  # Obedience examples
    SIN = "sin"  # Sin and failure
    JUDGMENT = "judgment"  # Divine judgment
    BLESSING = "blessing"  # Divine blessing
    LEADERSHIP = "leadership"  # Leadership examples
    SUFFERING = "suffering"  # Suffering examples
    DELIVERANCE = "deliverance"  # Deliverance examples


@dataclass
class BiblicalExample:
    """A biblical example or precedent."""
    figure: str
    reference: str
    example_type: ExampleType
    category: PrecedentCategory
    lesson: str
    cited_in: List[str]
    relevance: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "figure": self.figure,
            "reference": self.reference,
            "example_type": self.example_type.value,
            "category": self.category.value,
            "lesson": self.lesson,
            "cited_in": self.cited_in,
            "relevance": self.relevance
        }


class ParadeigmaAgent(BaseExtractionAgent):
    """
    PARADEIGMA - Example/precedent agent.

    Performs:
    - Biblical figure identification
    - Example type classification
    - Precedent cataloging
    - Moral lesson extraction
    - Cross-testament example mapping
    """

    # Key biblical figures and their examples
    BIBLICAL_FIGURES = {
        "abraham": {
            "category": PrecedentCategory.FAITH,
            "example_type": ExampleType.POSITIVE,
            "lesson": "Faith counted as righteousness",
            "ot_refs": ["GEN.12", "GEN.15", "GEN.22"],
            "nt_citations": ["ROM.4", "GAL.3", "HEB.11.8-19", "JAM.2.21-23"]
        },
        "moses": {
            "category": PrecedentCategory.LEADERSHIP,
            "example_type": ExampleType.POSITIVE,
            "lesson": "Faithful servant of God",
            "ot_refs": ["EXO.3", "DEU.34"],
            "nt_citations": ["HEB.3.2-5", "HEB.11.23-29"]
        },
        "david": {
            "category": PrecedentCategory.FAITH,
            "example_type": ExampleType.POSITIVE,
            "lesson": "Man after God's heart",
            "ot_refs": ["1SA.17", "2SA.7", "PSA"],
            "nt_citations": ["ACT.13.22", "HEB.11.32"]
        },
        "rahab": {
            "category": PrecedentCategory.FAITH,
            "example_type": ExampleType.POSITIVE,
            "lesson": "Faith from outside Israel",
            "ot_refs": ["JOS.2", "JOS.6.25"],
            "nt_citations": ["HEB.11.31", "JAM.2.25"]
        },
        "samson": {
            "category": PrecedentCategory.FAITH,
            "example_type": ExampleType.POSITIVE,
            "lesson": "Strength through weakness",
            "ot_refs": ["JDG.13-16"],
            "nt_citations": ["HEB.11.32"]
        },
        "elijah": {
            "category": PrecedentCategory.FAITH,
            "example_type": ExampleType.POSITIVE,
            "lesson": "Prophetic boldness",
            "ot_refs": ["1KI.17-19", "2KI.1-2"],
            "nt_citations": ["JAM.5.17-18", "ROM.11.2-4"]
        },
        "job": {
            "category": PrecedentCategory.SUFFERING,
            "example_type": ExampleType.POSITIVE,
            "lesson": "Patience in suffering",
            "ot_refs": ["JOB.1-2", "JOB.42"],
            "nt_citations": ["JAM.5.11"]
        },
        "adam": {
            "category": PrecedentCategory.SIN,
            "example_type": ExampleType.NEGATIVE,
            "lesson": "Consequences of disobedience",
            "ot_refs": ["GEN.3"],
            "nt_citations": ["ROM.5.12-21", "1CO.15.22"]
        },
        "cain": {
            "category": PrecedentCategory.SIN,
            "example_type": ExampleType.NEGATIVE,
            "lesson": "Sinful hatred and murder",
            "ot_refs": ["GEN.4"],
            "nt_citations": ["1JN.3.12", "JUD.1.11"]
        },
        "esau": {
            "category": PrecedentCategory.SIN,
            "example_type": ExampleType.NEGATIVE,
            "lesson": "Profane disregard for spiritual things",
            "ot_refs": ["GEN.25.29-34", "GEN.27"],
            "nt_citations": ["HEB.12.16-17"]
        },
        "balaam": {
            "category": PrecedentCategory.SIN,
            "example_type": ExampleType.NEGATIVE,
            "lesson": "Greed and false teaching",
            "ot_refs": ["NUM.22-24", "NUM.31.16"],
            "nt_citations": ["2PE.2.15", "JUD.1.11", "REV.2.14"]
        },
        "korah": {
            "category": PrecedentCategory.JUDGMENT,
            "example_type": ExampleType.NEGATIVE,
            "lesson": "Rebellion against authority",
            "ot_refs": ["NUM.16"],
            "nt_citations": ["JUD.1.11"]
        },
        "sodom": {
            "category": PrecedentCategory.JUDGMENT,
            "example_type": ExampleType.NEGATIVE,
            "lesson": "Divine judgment on wickedness",
            "ot_refs": ["GEN.19"],
            "nt_citations": ["2PE.2.6", "JUD.1.7", "MAT.10.15"]
        }
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="paradeigma",
                extraction_type=ExtractionType.INTERTEXTUAL,
                batch_size=200,
                min_confidence=0.6
            )
        super().__init__(config)
        self.logger = logging.getLogger("biblos.agents.paradeigma")

    async def extract(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any]
    ) -> ExtractionResult:
        """Extract example/precedent analysis."""
        # Identify biblical figures
        figures = self._identify_figures(text)

        # Build example records
        examples = self._build_examples(figures, verse_id, text)

        # Classify examples
        classified = self._classify_examples(examples)

        # Extract lessons
        lessons = self._extract_lessons(examples)

        # Analyze patterns
        patterns = self._analyze_patterns(examples)

        data = {
            "examples": [e.to_dict() for e in examples],
            "figures_mentioned": [f["name"] for f in figures],
            "lessons": lessons,
            "patterns": patterns,
            "positive_count": sum(
                1 for e in examples
                if e.example_type == ExampleType.POSITIVE
            ),
            "negative_count": sum(
                1 for e in examples
                if e.example_type == ExampleType.NEGATIVE
            ),
            "categories": list(set(e.category.value for e in examples))
        }

        confidence = self._calculate_confidence(examples, figures)

        return ExtractionResult(
            agent_name=self.config.name,
            extraction_type=self.config.extraction_type,
            verse_id=verse_id,
            status=ProcessingStatus.COMPLETED,
            data=data,
            confidence=confidence
        )

    def _identify_figures(self, text: str) -> List[Dict[str, Any]]:
        """Identify biblical figures in text."""
        figures = []
        text_lower = text.lower()

        for figure, info in self.BIBLICAL_FIGURES.items():
            if figure in text_lower:
                figures.append({
                    "name": figure,
                    "info": info
                })

        return figures

    def _build_examples(
        self,
        figures: List[Dict[str, Any]],
        verse_id: str,
        text: str
    ) -> List[BiblicalExample]:
        """Build example records from identified figures."""
        examples = []

        for figure in figures:
            info = figure["info"]
            examples.append(BiblicalExample(
                figure=figure["name"],
                reference=verse_id,
                example_type=info["example_type"],
                category=info["category"],
                lesson=info["lesson"],
                cited_in=info["nt_citations"],
                relevance=0.8
            ))

        return examples

    def _classify_examples(
        self,
        examples: List[BiblicalExample]
    ) -> Dict[str, List[str]]:
        """Classify examples by type and category."""
        by_type = {}
        by_category = {}

        for example in examples:
            t = example.example_type.value
            c = example.category.value

            if t not in by_type:
                by_type[t] = []
            by_type[t].append(example.figure)

            if c not in by_category:
                by_category[c] = []
            by_category[c].append(example.figure)

        return {
            "by_type": by_type,
            "by_category": by_category
        }

    def _extract_lessons(
        self,
        examples: List[BiblicalExample]
    ) -> List[Dict[str, str]]:
        """Extract moral lessons from examples."""
        lessons = []

        for example in examples:
            lessons.append({
                "figure": example.figure,
                "lesson": example.lesson,
                "type": "positive" if example.example_type == ExampleType.POSITIVE else "negative"
            })

        return lessons

    def _analyze_patterns(
        self,
        examples: List[BiblicalExample]
    ) -> Dict[str, Any]:
        """Analyze patterns in examples."""
        if not examples:
            return {
                "dominant_type": None,
                "dominant_category": None,
                "consistency": 0.0
            }

        types = [e.example_type.value for e in examples]
        categories = [e.category.value for e in examples]

        type_counts = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1

        category_counts = {}
        for c in categories:
            category_counts[c] = category_counts.get(c, 0) + 1

        dominant_type = max(type_counts, key=type_counts.get) if type_counts else None
        dominant_category = max(category_counts, key=category_counts.get) if category_counts else None

        # Consistency = how uniform are the examples
        if type_counts:
            max_count = max(type_counts.values())
            consistency = max_count / len(examples)
        else:
            consistency = 0.0

        return {
            "dominant_type": dominant_type,
            "dominant_category": dominant_category,
            "consistency": consistency,
            "type_distribution": type_counts,
            "category_distribution": category_counts
        }

    def _calculate_confidence(
        self,
        examples: List[BiblicalExample],
        figures: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score."""
        confidence = 0.5

        if figures:
            confidence += 0.2

        if examples:
            confidence += 0.2
            # Boost for well-documented figures
            documented = sum(
                1 for e in examples
                if len(e.cited_in) > 2
            )
            confidence += documented * 0.05

        return min(1.0, confidence)

    async def validate(self, result: ExtractionResult) -> bool:
        """Validate extraction result."""
        data = result.data
        return "examples" in data and "figures_mentioned" in data

    def get_dependencies(self) -> List[str]:
        """Get agent dependencies."""
        return ["grammateus", "syndesmos", "typologos"]
