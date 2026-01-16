"""
BIBLOS v2 - ALLOGRAPHOS Agent

Quotation and allusion detection for biblical texts.
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import re

from agents.base import (
    BaseExtractionAgent,
    AgentConfig,
    ExtractionResult,
    ExtractionType,
    ProcessingStatus
)


class ReferenceType(Enum):
    """Types of intertextual references."""
    DIRECT_QUOTATION = "direct_quotation"  # Verbatim or near-verbatim
    ALLUSION = "allusion"  # Clear reference without quote
    ECHO = "echo"  # Faint reference
    PARAPHRASE = "paraphrase"  # Restated content
    CONFLATION = "conflation"  # Combined sources
    FORMULA = "formula"  # Set phrase/formula


class QuotationSource(Enum):
    """Source of quotation."""
    MT = "masoretic_text"  # Hebrew MT
    LXX = "septuagint"  # Greek LXX
    MIXED = "mixed"  # Both or unclear
    TARGUM = "targum"  # Aramaic targum
    OTHER = "other"  # Other source


@dataclass
class Quotation:
    """A biblical quotation or allusion."""
    source_ref: str
    target_ref: str
    reference_type: ReferenceType
    source_text: Optional[str]
    quoted_text: str
    text_form: QuotationSource
    verbal_similarity: float
    introduction_formula: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_ref": self.source_ref,
            "target_ref": self.target_ref,
            "reference_type": self.reference_type.value,
            "source_text": self.source_text,
            "quoted_text": self.quoted_text,
            "text_form": self.text_form.value,
            "verbal_similarity": self.verbal_similarity,
            "introduction_formula": self.introduction_formula
        }


class AllographosAgent(BaseExtractionAgent):
    """
    ALLOGRAPHOS - Quotation/allusion agent.

    Performs:
    - OT quotation identification in NT
    - Allusion detection
    - Text form analysis (MT vs LXX)
    - Introduction formula cataloging
    - Verbal similarity calculation
    """

    # Introduction formulae
    INTRODUCTION_FORMULAE = {
        "explicit": [
            "it is written", "as it is written", "scripture says",
            "the scripture says", "for it is written", "as the scripture says",
            "γέγραπται", "καθὼς γέγραπται", "ἡ γραφὴ λέγει"
        ],
        "prophetic": [
            "spoken by the prophet", "through the prophet",
            "the prophet says", "as the prophet said",
            "διὰ τοῦ προφήτου"
        ],
        "mosaic": [
            "moses said", "moses wrote", "the law says",
            "moses commanded", "μωϋσῆς λέγει"
        ],
        "davidic": [
            "david said", "david says", "in the psalms",
            "δαυὶδ λέγει"
        ],
        "fulfillment": [
            "that it might be fulfilled", "this was to fulfill",
            "fulfilling what was spoken", "ἵνα πληρωθῇ"
        ]
    }

    # Common OT quotations in NT
    KNOWN_QUOTATIONS = {
        # Isaiah quotations
        "ISA.6.9-10": ["MAT.13.14-15", "MRK.4.12", "LUK.8.10", "JHN.12.40", "ACT.28.26-27"],
        "ISA.7.14": ["MAT.1.23"],
        "ISA.40.3": ["MAT.3.3", "MRK.1.3", "LUK.3.4", "JHN.1.23"],
        "ISA.53.1": ["JHN.12.38", "ROM.10.16"],
        "ISA.53.4": ["MAT.8.17"],
        # Psalms quotations
        "PSA.2.7": ["ACT.13.33", "HEB.1.5", "HEB.5.5"],
        "PSA.22.1": ["MAT.27.46", "MRK.15.34"],
        "PSA.110.1": ["MAT.22.44", "MRK.12.36", "LUK.20.42-43", "ACT.2.34-35", "HEB.1.13"],
        "PSA.118.22-23": ["MAT.21.42", "MRK.12.10-11", "LUK.20.17", "ACT.4.11", "1PE.2.7"],
        # Deuteronomy quotations
        "DEU.6.4-5": ["MAT.22.37", "MRK.12.29-30", "LUK.10.27"],
        "DEU.6.13": ["MAT.4.10", "LUK.4.8"],
        "DEU.6.16": ["MAT.4.7", "LUK.4.12"],
        "DEU.8.3": ["MAT.4.4", "LUK.4.4"],
        # Genesis quotations
        "GEN.1.27": ["MAT.19.4", "MRK.10.6"],
        "GEN.2.24": ["MAT.19.5", "MRK.10.7-8", "EPH.5.31"],
        "GEN.15.6": ["ROM.4.3", "GAL.3.6", "JAM.2.23"]
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="allographos",
                extraction_type=ExtractionType.INTERTEXTUAL,
                batch_size=200,
                min_confidence=0.6
            )
        super().__init__(config)
        self.logger = logging.getLogger("biblos.agents.allographos")

    async def extract(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any]
    ) -> ExtractionResult:
        """Extract quotation/allusion analysis."""
        is_nt = self._is_new_testament(verse_id)

        # Detect quotations
        quotations = self._detect_quotations(verse_id, text, context, is_nt)

        # Analyze text form
        text_forms = self._analyze_text_forms(quotations)

        # Catalog introduction formulae
        formulae = self._catalog_formulae(text)

        # Calculate metrics
        metrics = self._calculate_metrics(quotations)

        data = {
            "is_nt": is_nt,
            "quotations": [q.to_dict() for q in quotations],
            "text_forms": text_forms,
            "introduction_formulae": formulae,
            "metrics": metrics,
            "has_explicit_quotation": any(
                q.reference_type == ReferenceType.DIRECT_QUOTATION
                for q in quotations
            ),
            "quotation_density": self._calculate_density(quotations, text)
        }

        confidence = self._calculate_confidence(quotations, formulae)

        return ExtractionResult(
            agent_name=self.config.name,
            extraction_type=self.config.extraction_type,
            verse_id=verse_id,
            status=ProcessingStatus.COMPLETED,
            data=data,
            confidence=confidence
        )

    def _is_new_testament(self, verse_id: str) -> bool:
        """Check if verse is from New Testament."""
        nt_books = {
            "MAT", "MRK", "LUK", "JHN", "ACT", "ROM", "1CO", "2CO",
            "GAL", "EPH", "PHP", "COL", "1TH", "2TH", "1TI", "2TI",
            "TIT", "PHM", "HEB", "JAM", "1PE", "2PE", "1JN", "2JN",
            "3JN", "JUD", "REV"
        }
        book = verse_id.split(".")[0]
        return book in nt_books

    def _detect_quotations(
        self,
        verse_id: str,
        text: str,
        context: Dict[str, Any],
        is_nt: bool
    ) -> List[Quotation]:
        """Detect quotations and allusions."""
        quotations = []

        if is_nt:
            # Check for known OT quotations
            for ot_ref, nt_refs in self.KNOWN_QUOTATIONS.items():
                for nt_ref in nt_refs:
                    if verse_id.startswith(nt_ref.split("-")[0][:9]):
                        formula = self._find_formula(text)
                        ref_type = (
                            ReferenceType.DIRECT_QUOTATION
                            if formula else ReferenceType.ALLUSION
                        )
                        quotations.append(Quotation(
                            source_ref=ot_ref,
                            target_ref=verse_id,
                            reference_type=ref_type,
                            source_text=None,
                            quoted_text=text,
                            text_form=QuotationSource.LXX,
                            verbal_similarity=0.8,
                            introduction_formula=formula
                        ))

        # Check context for additional quotations
        context_quotes = context.get("ot_quotations", [])
        for quote in context_quotes:
            quotations.append(Quotation(
                source_ref=quote.get("source", ""),
                target_ref=verse_id,
                reference_type=self._determine_ref_type(quote),
                source_text=quote.get("source_text"),
                quoted_text=quote.get("quoted_text", text),
                text_form=self._determine_text_form(quote),
                verbal_similarity=quote.get("similarity", 0.7),
                introduction_formula=quote.get("formula")
            ))

        return quotations

    def _find_formula(self, text: str) -> Optional[str]:
        """Find introduction formula in text."""
        text_lower = text.lower()

        for formula_type, formulae in self.INTRODUCTION_FORMULAE.items():
            for formula in formulae:
                if formula.lower() in text_lower:
                    return formula

        return None

    def _determine_ref_type(self, quote: Dict[str, Any]) -> ReferenceType:
        """Determine reference type from quote data."""
        type_str = quote.get("type", "allusion").lower()
        type_map = {
            "direct": ReferenceType.DIRECT_QUOTATION,
            "quotation": ReferenceType.DIRECT_QUOTATION,
            "allusion": ReferenceType.ALLUSION,
            "echo": ReferenceType.ECHO,
            "paraphrase": ReferenceType.PARAPHRASE,
            "conflation": ReferenceType.CONFLATION,
            "formula": ReferenceType.FORMULA
        }
        return type_map.get(type_str, ReferenceType.ALLUSION)

    def _determine_text_form(self, quote: Dict[str, Any]) -> QuotationSource:
        """Determine text form of quotation."""
        form_str = quote.get("text_form", "lxx").lower()
        form_map = {
            "mt": QuotationSource.MT,
            "masoretic": QuotationSource.MT,
            "hebrew": QuotationSource.MT,
            "lxx": QuotationSource.LXX,
            "septuagint": QuotationSource.LXX,
            "greek": QuotationSource.LXX,
            "mixed": QuotationSource.MIXED,
            "targum": QuotationSource.TARGUM
        }
        return form_map.get(form_str, QuotationSource.LXX)

    def _analyze_text_forms(
        self,
        quotations: List[Quotation]
    ) -> Dict[str, Any]:
        """Analyze text forms of quotations."""
        if not quotations:
            return {"forms": [], "primary": None}

        forms = [q.text_form.value for q in quotations]
        form_counts = {}
        for f in forms:
            form_counts[f] = form_counts.get(f, 0) + 1

        primary = max(form_counts.keys(), key=lambda k: form_counts[k]) if form_counts else None

        return {
            "forms": list(set(forms)),
            "counts": form_counts,
            "primary": primary,
            "has_lxx": QuotationSource.LXX.value in forms,
            "has_mt": QuotationSource.MT.value in forms
        }

    def _catalog_formulae(self, text: str) -> List[Dict[str, Any]]:
        """Catalog introduction formulae in text."""
        formulae = []
        text_lower = text.lower()

        for formula_type, formula_list in self.INTRODUCTION_FORMULAE.items():
            for formula in formula_list:
                if formula.lower() in text_lower:
                    formulae.append({
                        "formula": formula,
                        "type": formula_type
                    })

        return formulae

    def _calculate_metrics(
        self,
        quotations: List[Quotation]
    ) -> Dict[str, Any]:
        """Calculate quotation metrics."""
        if not quotations:
            return {
                "total": 0,
                "direct": 0,
                "allusions": 0,
                "avg_similarity": 0.0
            }

        direct = sum(
            1 for q in quotations
            if q.reference_type == ReferenceType.DIRECT_QUOTATION
        )
        allusions = sum(
            1 for q in quotations
            if q.reference_type == ReferenceType.ALLUSION
        )
        avg_similarity = sum(q.verbal_similarity for q in quotations) / max(1, len(quotations))

        return {
            "total": len(quotations),
            "direct": direct,
            "allusions": allusions,
            "echoes": len(quotations) - direct - allusions,
            "avg_similarity": avg_similarity
        }

    def _calculate_density(
        self,
        quotations: List[Quotation],
        text: str
    ) -> float:
        """Calculate quotation density."""
        if not quotations or not text:
            return 0.0

        # Density based on quotation count and text length
        word_count = len(text.split())
        base_density = len(quotations) / max(1, word_count / 10)

        # Boost for direct quotations
        direct_boost = sum(
            0.1 for q in quotations
            if q.reference_type == ReferenceType.DIRECT_QUOTATION
        )

        return min(1.0, base_density + direct_boost)

    def _calculate_confidence(
        self,
        quotations: List[Quotation],
        formulae: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score."""
        confidence = 0.5

        if quotations:
            confidence += 0.2
            # Boost for high similarity
            high_sim = sum(1 for q in quotations if q.verbal_similarity > 0.7)
            confidence += high_sim * 0.05

        if formulae:
            confidence += 0.15

        return min(1.0, confidence)

    async def validate(self, result: ExtractionResult) -> bool:
        """Validate extraction result."""
        data = result.data
        return "quotations" in data and "is_nt" in data

    def get_dependencies(self) -> List[str]:
        """Get agent dependencies."""
        return ["grammateus", "syndesmos"]
