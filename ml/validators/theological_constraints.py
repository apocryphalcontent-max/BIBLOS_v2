"""
BIBLOS v2 - Theological Constraint Validator

Encodes patristic theological principles as algorithmic constraints for
"covert theological governance" - truth enforcement without explicit
attribution to sources.

Principles Encoded:
1. Antitype Escalation - Antitype must exceed type in scope/magnitude
2. Prophetic Coherence - Fulfillment extends promise, never contradicts
3. Chronological Priority - Type MUST historically precede antitype
4. Christological Warrant - Requires apostolic use OR patristic consensus
5. Liturgical Amplification - Liturgical connections boost confidence
6. Fourfold Foundation - Allegorical reading requires literal foundation
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
import logging
import re

# Import canonical book order for chronological validation
from config import BOOK_ORDER


logger = logging.getLogger("biblos.ml.validators.theological_constraints")


# =============================================================================
# ENUMS
# =============================================================================

class ConstraintViolationSeverity(Enum):
    """
    Severity levels for constraint violations.

    IMPOSSIBLE: Logical impossibility, reject outright (confidence = 0)
    CRITICAL: Severe theological error (confidence × 0.2-0.3)
    SOFT: Marginal violation (confidence × 0.7-0.8)
    WARNING: Not ideal but acceptable (confidence × 0.9)
    BOOST: Positive validation (confidence × 1.1-1.2)
    """
    IMPOSSIBLE = "IMPOSSIBLE"
    CRITICAL = "CRITICAL"
    SOFT = "SOFT"
    WARNING = "WARNING"
    BOOST = "BOOST"


class ConstraintType(Enum):
    """Types of theological constraints."""
    TYPOLOGICAL_ESCALATION = "TYPOLOGICAL_ESCALATION"
    PROPHETIC_COHERENCE = "PROPHETIC_COHERENCE"
    CHRONOLOGICAL_PRIORITY = "CHRONOLOGICAL_PRIORITY"
    CHRISTOLOGICAL_WARRANT = "CHRISTOLOGICAL_WARRANT"
    LITURGICAL_AMPLIFICATION = "LITURGICAL_AMPLIFICATION"
    FOURFOLD_FOUNDATION = "FOURFOLD_FOUNDATION"


class Scope(Enum):
    """Scope levels for typological analysis."""
    LOCAL = "LOCAL"           # Individual, momentary, place-specific
    NATIONAL = "NATIONAL"     # Family/nation, period, regional
    UNIVERSAL = "UNIVERSAL"   # All humanity, era-spanning, worldwide
    COSMIC = "COSMIC"         # All creation, eternal, transcendent


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class ConstraintResult:
    """
    Result of a constraint evaluation.

    Attributes:
        passed: Did the constraint pass?
        constraint_type: Which constraint was evaluated
        violation_severity: If failed, how severe
        confidence_modifier: Multiplier to apply (0.0 to 1.5)
        reason: Human-readable explanation
        evidence: Supporting evidence for the decision
        recoverable: Can this be fixed with additional evidence?
    """
    passed: bool
    constraint_type: ConstraintType
    violation_severity: Optional[ConstraintViolationSeverity] = None
    confidence_modifier: float = 1.0
    reason: str = ""
    evidence: List[str] = field(default_factory=list)
    recoverable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "passed": self.passed,
            "constraint_type": self.constraint_type.value,
            "violation_severity": self.violation_severity.value if self.violation_severity else None,
            "confidence_modifier": self.confidence_modifier,
            "reason": self.reason,
            "evidence": self.evidence,
            "recoverable": self.recoverable,
        }


# =============================================================================
# SCOPE/MAGNITUDE ANALYZER
# =============================================================================

class ScopeMagnitudeAnalyzer:
    """
    Analyzes scope and magnitude for typological escalation validation.

    Determines whether the antitype properly exceeds the type in theological
    significance according to patristic typological principles.
    """

    # Keywords indicating scope levels
    SCOPE_INDICATORS = {
        Scope.COSMIC: [
            "creation", "universe", "all things", "heaven and earth",
            "eternal", "forever", "ages", "everlasting", "cosmos"
        ],
        Scope.UNIVERSAL: [
            "all nations", "humanity", "mankind", "world", "every",
            "all people", "gentiles", "all flesh", "whosoever"
        ],
        Scope.NATIONAL: [
            "israel", "people", "nation", "tribe", "house of",
            "children of", "assembly", "congregation"
        ],
        Scope.LOCAL: [
            "man", "person", "individual", "one", "place", "city",
            "village", "tent", "house"
        ],
    }

    # Agent significance rankings
    AGENT_SIGNIFICANCE = {
        "god": 100,
        "lord": 100,
        "christ": 95,
        "messiah": 95,
        "son of god": 95,
        "spirit": 90,
        "holy spirit": 90,
        "angel": 70,
        "prophet": 60,
        "king": 50,
        "priest": 50,
        "judge": 40,
        "patriarch": 40,
        "servant": 30,
        "man": 20,
        "person": 20,
    }

    def analyze_scope(
        self,
        element: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Scope:
        """
        Determine the scope of a typological element.

        Args:
            element: The typological element data
            context: Additional context for analysis

        Returns:
            Scope enum value (LOCAL, NATIONAL, UNIVERSAL, COSMIC)
        """
        text = self._extract_text(element, context)
        text_lower = text.lower()

        # Check for scope indicators from most expansive to least
        for scope in [Scope.COSMIC, Scope.UNIVERSAL, Scope.NATIONAL, Scope.LOCAL]:
            indicators = self.SCOPE_INDICATORS[scope]
            if any(indicator in text_lower for indicator in indicators):
                return scope

        # Default to LOCAL if no indicators found
        return Scope.LOCAL

    def calculate_magnitude(
        self,
        element: Dict[str, Any]
    ) -> float:
        """
        Calculate the magnitude of a typological element (0-100 scale).

        Factors:
        - Agent significance (who is acting)
        - Action reversibility (temporary vs eternal)
        - Effect breadth (narrow vs comprehensive)

        Args:
            element: The typological element data

        Returns:
            Magnitude score (0-100)
        """
        text = element.get("text", element.get("description", "")).lower()

        # Agent significance (0-100)
        agent_score = 20  # Default
        for agent, score in self.AGENT_SIGNIFICANCE.items():
            if agent in text:
                agent_score = max(agent_score, score)

        # Action reversibility (0-1)
        eternal_keywords = ["eternal", "forever", "everlasting", "never", "always"]
        lasting_keywords = ["covenant", "promise", "inheritance", "salvation"]
        temporary_keywords = ["day", "moment", "time", "season"]

        if any(kw in text for kw in eternal_keywords):
            reversibility = 1.0
        elif any(kw in text for kw in lasting_keywords):
            reversibility = 0.7
        elif any(kw in text for kw in temporary_keywords):
            reversibility = 0.3
        else:
            reversibility = 0.5

        # Effect breadth (0-1)
        comprehensive_keywords = ["all", "every", "complete", "full", "whole"]
        broad_keywords = ["many", "much", "great", "abundant"]
        narrow_keywords = ["one", "single", "only", "alone"]

        if any(kw in text for kw in comprehensive_keywords):
            breadth = 1.0
        elif any(kw in text for kw in broad_keywords):
            breadth = 0.7
        elif any(kw in text for kw in narrow_keywords):
            breadth = 0.3
        else:
            breadth = 0.5

        # Combine factors: agent * (0.5 * reversibility + 0.5 * breadth)
        magnitude = agent_score * (0.5 * reversibility + 0.5 * breadth)

        return magnitude

    def analyze_fulfillment_completeness(
        self,
        type_elem: Dict[str, Any],
        antitype_elem: Dict[str, Any]
    ) -> float:
        """
        Analyze how completely the antitype fulfills the type.

        Returns:
            Completeness score (0-1)
            1.0 = antitype fulfills ALL aspects of type
            0.5 = partial fulfillment
            < 0.5 = incomplete, raises concerns
        """
        type_text = self._extract_text(type_elem, {}).lower()
        antitype_text = self._extract_text(antitype_elem, {}).lower()

        # Extract key concepts from type
        type_concepts = self._extract_concepts(type_text)
        antitype_concepts = self._extract_concepts(antitype_text)

        if not type_concepts:
            return 0.5  # Can't assess if no concepts extracted

        # Calculate overlap
        overlap = len(type_concepts & antitype_concepts)
        completeness = overlap / len(type_concepts) if type_concepts else 0.5

        # Bonus for antitype having additional concepts (escalation)
        additional = len(antitype_concepts - type_concepts)
        if additional > 0:
            completeness = min(1.0, completeness + 0.1 * min(additional, 3))

        return completeness

    def _extract_text(
        self,
        element: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Extract text from element or context."""
        if isinstance(element, str):
            return element
        return element.get("text", element.get("description", ""))

    def _extract_concepts(self, text: str) -> Set[str]:
        """Extract key theological concepts from text."""
        # Theological concept keywords
        concepts = {
            "sacrifice", "blood", "lamb", "altar", "offering",
            "covenant", "promise", "blessing", "inheritance",
            "salvation", "redemption", "deliverance", "freedom",
            "king", "kingdom", "throne", "crown", "reign",
            "priest", "temple", "worship", "holy",
            "prophet", "word", "message", "revelation",
            "death", "resurrection", "life", "eternal",
            "sin", "forgiveness", "atonement", "cleansing",
            "water", "baptism", "birth", "new",
            "bread", "body", "wine", "blood",
            "shepherd", "sheep", "flock", "pasture",
            "servant", "son", "father", "spirit"
        }

        found = set()
        text_lower = text.lower()
        for concept in concepts:
            if concept in text_lower:
                found.add(concept)

        return found


# =============================================================================
# SEMANTIC COHERENCE CHECKER
# =============================================================================

class SemanticCoherenceChecker:
    """
    Checks semantic coherence between prophetic promises and fulfillments.

    Ensures that fulfillment extends/completes the promise without contradiction.
    """

    # Semantic contradiction pairs
    CONTRADICTION_PAIRS = [
        ("life", "death"),
        ("blessing", "curse"),
        ("victory", "defeat"),
        ("salvation", "destruction"),
        ("peace", "war"),
        ("unity", "division"),
        ("freedom", "bondage"),
        ("light", "darkness"),
        ("truth", "lie"),
        ("love", "hate"),
    ]

    # Extension patterns (promise → fulfillment expansion)
    EXTENSION_PATTERNS = [
        ("land", "kingdom"),
        ("nation", "all nations"),
        ("temporal", "eternal"),
        ("physical", "spiritual"),
        ("local", "universal"),
        ("shadow", "reality"),
        ("partial", "complete"),
        ("earthly", "heavenly"),
    ]

    def extract_promise_components(
        self,
        semantics: Dict[str, Any]
    ) -> Set[str]:
        """
        Extract promise components from semantic analysis.

        Args:
            semantics: Semantic analysis data

        Returns:
            Set of promised outcomes/conditions
        """
        components = set()

        text = semantics.get("text", "").lower()
        themes = semantics.get("themes", [])
        keywords = semantics.get("keywords", [])

        # Add themes and keywords
        components.update(str(t).lower() for t in themes)
        components.update(str(k).lower() for k in keywords)

        # Extract action-object patterns
        promise_markers = ["will", "shall", "promise", "give", "make", "establish"]
        for marker in promise_markers:
            if marker in text:
                components.add(f"promise_{marker}")

        return components

    def extract_fulfillment_claims(
        self,
        semantics: Dict[str, Any]
    ) -> Set[str]:
        """
        Extract fulfillment claims from semantic analysis.

        Args:
            semantics: Semantic analysis data

        Returns:
            Set of claimed fulfillments
        """
        claims = set()

        text = semantics.get("text", "").lower()
        themes = semantics.get("themes", [])
        keywords = semantics.get("keywords", [])

        # Add themes and keywords
        claims.update(str(t).lower() for t in themes)
        claims.update(str(k).lower() for k in keywords)

        # Extract fulfillment markers
        fulfillment_markers = ["fulfilled", "completed", "accomplished", "finished"]
        for marker in fulfillment_markers:
            if marker in text:
                claims.add(f"fulfillment_{marker}")

        return claims

    def detect_contradictions(
        self,
        promise_semantics: Dict[str, Any],
        fulfillment_semantics: Dict[str, Any]
    ) -> List[str]:
        """
        Detect semantic contradictions between promise and fulfillment.

        Args:
            promise_semantics: Promise semantic analysis
            fulfillment_semantics: Fulfillment semantic analysis

        Returns:
            List of contradiction descriptions
        """
        contradictions = []

        promise_text = promise_semantics.get("text", "").lower()
        fulfillment_text = fulfillment_semantics.get("text", "").lower()

        # Check contradiction pairs
        for pos, neg in self.CONTRADICTION_PAIRS:
            # Promise has positive, fulfillment has negative without resolution
            if pos in promise_text and neg in fulfillment_text:
                # Check if there's a resolution context
                resolution_markers = ["through", "by", "resulting in", "leading to"]
                has_resolution = any(m in fulfillment_text for m in resolution_markers)

                if not has_resolution:
                    contradictions.append(
                        f"Promise implies '{pos}' but fulfillment contains '{neg}' without resolution"
                    )

        return contradictions

    def detect_extensions(
        self,
        promise_semantics: Dict[str, Any],
        fulfillment_semantics: Dict[str, Any]
    ) -> List[str]:
        """
        Detect where fulfillment exceeds/extends promise.

        Args:
            promise_semantics: Promise semantic analysis
            fulfillment_semantics: Fulfillment semantic analysis

        Returns:
            List of extension descriptions
        """
        extensions = []

        promise_text = promise_semantics.get("text", "").lower()
        fulfillment_text = fulfillment_semantics.get("text", "").lower()

        # Check extension patterns
        for limited, expanded in self.EXTENSION_PATTERNS:
            if limited in promise_text and expanded in fulfillment_text:
                extensions.append(
                    f"Promise of '{limited}' extended to '{expanded}' in fulfillment"
                )

        return extensions

    def check_entailment(
        self,
        promise_semantics: Dict[str, Any],
        fulfillment_semantics: Dict[str, Any]
    ) -> Tuple[bool, float]:
        """
        Check if fulfillment semantically entails promise.

        Returns:
            Tuple of (entails, confidence)
        """
        promise_components = self.extract_promise_components(promise_semantics)
        fulfillment_claims = self.extract_fulfillment_claims(fulfillment_semantics)

        if not promise_components:
            return True, 0.5  # Can't disprove if no components

        # Calculate overlap
        overlap = len(promise_components & fulfillment_claims)
        coverage = overlap / len(promise_components)

        # Check for contradictions
        contradictions = self.detect_contradictions(promise_semantics, fulfillment_semantics)

        if contradictions:
            return False, 0.2

        # Check for extensions (positive)
        extensions = self.detect_extensions(promise_semantics, fulfillment_semantics)

        # High coverage + extensions = good entailment
        if coverage >= 0.5 and extensions:
            return True, min(1.0, coverage + 0.2)

        return coverage >= 0.3, coverage


# =============================================================================
# MAIN VALIDATOR CLASS
# =============================================================================

class TheologicalConstraintValidator:
    """
    Validates cross-references against patristic theological constraints.

    Implements "covert theological governance" by encoding Church Father
    principles as algorithmic rules without explicit attribution.
    """

    # Major Church Fathers whose consensus constitutes warrant
    MAJOR_FATHERS = [
        "Athanasius", "Basil", "Gregory_Nazianzen", "Gregory_Nyssa",
        "Chrysostom", "Cyril_Alexandria", "Augustine", "Ambrose",
        "Jerome", "Maximus_Confessor", "John_Damascus", "Origen",
        "Irenaeus", "Clement_Alexandria", "Ephrem", "Leo_Great"
    ]

    # OT books (for chronological validation)
    OT_BOOKS = set(BOOK_ORDER[:39]) if len(BOOK_ORDER) >= 39 else set(BOOK_ORDER)

    # NT books
    NT_BOOKS = set(BOOK_ORDER[39:]) if len(BOOK_ORDER) >= 39 else set()

    # Major liturgical contexts
    LITURGICAL_CONTEXTS = {
        "pascha": 2.0,           # Highest - Paschal significance
        "holy_week": 1.8,
        "nativity": 1.5,
        "theophany": 1.5,
        "pentecost": 1.5,
        "transfiguration": 1.4,
        "dormition": 1.3,
        "lectionary": 1.2,
        "vespers": 1.1,
        "matins": 1.1,
        "divine_liturgy": 1.3,
    }

    # Constraint applicability by connection type
    CONSTRAINT_APPLICABILITY = {
        "typological": [
            ConstraintType.TYPOLOGICAL_ESCALATION,
            ConstraintType.CHRONOLOGICAL_PRIORITY,
            ConstraintType.LITURGICAL_AMPLIFICATION,
            ConstraintType.FOURFOLD_FOUNDATION,
        ],
        "prophetic": [
            ConstraintType.PROPHETIC_COHERENCE,
            ConstraintType.CHRONOLOGICAL_PRIORITY,
            ConstraintType.CHRISTOLOGICAL_WARRANT,
        ],
        "verbal": [
            ConstraintType.LITURGICAL_AMPLIFICATION,
        ],
        "thematic": [
            ConstraintType.LITURGICAL_AMPLIFICATION,
        ],
        "conceptual": [
            ConstraintType.LITURGICAL_AMPLIFICATION,
        ],
        "historical": [],
        "narrative": [],
        "genealogical": [],
        "geographical": [],
        "liturgical": [
            ConstraintType.LITURGICAL_AMPLIFICATION,
        ],
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the validator.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.scope_analyzer = ScopeMagnitudeAnalyzer()
        self.coherence_checker = SemanticCoherenceChecker()

        # Configuration with defaults
        self.enable_escalation = self.config.get("enable_escalation_validation", True)
        self.enable_prophetic = self.config.get("enable_prophetic_coherence", True)
        self.enable_chronological = self.config.get("enable_chronological_priority", True)
        self.enable_warrant = self.config.get("enable_christological_warrant", True)
        self.enable_liturgical = self.config.get("enable_liturgical_amplification", True)
        self.enable_fourfold = self.config.get("enable_fourfold_foundation", True)

        self.min_patristic_witnesses = self.config.get("minimum_patristic_witnesses", 2)
        self.escalation_critical_threshold = self.config.get("escalation_critical_threshold", 1.0)
        self.escalation_boost_threshold = self.config.get("escalation_boost_threshold", 1.5)
        self.liturgical_boost_factor = self.config.get("liturgical_boost_factor", 1.1)
        self.apostolic_boost_factor = self.config.get("apostolic_boost_factor", 1.2)
        self.patristic_boost_factor = self.config.get("patristic_boost_factor", 1.1)

        logger.info(
            f"TheologicalConstraintValidator initialized with "
            f"min_patristic_witnesses={self.min_patristic_witnesses}"
        )

    def validate_chronological_priority(
        self,
        type_ref: str,
        antitype_ref: str,
        canon_order: Optional[List[str]] = None
    ) -> ConstraintResult:
        """
        Validate that type chronologically precedes antitype.

        This is a HARD constraint with no exceptions. Types must come
        before antitypes in canonical order.

        Args:
            type_ref: Type verse reference (e.g., "GEN.22.2")
            antitype_ref: Antitype verse reference (e.g., "JHN.3.16")
            canon_order: Optional custom canonical ordering

        Returns:
            ConstraintResult with pass/fail and modifier
        """
        if not self.enable_chronological:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.CHRONOLOGICAL_PRIORITY,
                reason="Chronological validation disabled",
            )

        order = canon_order or BOOK_ORDER

        # Extract book codes
        type_book = self._extract_book_code(type_ref)
        antitype_book = self._extract_book_code(antitype_ref)

        if not type_book or not antitype_book:
            return ConstraintResult(
                passed=False,
                constraint_type=ConstraintType.CHRONOLOGICAL_PRIORITY,
                violation_severity=ConstraintViolationSeverity.WARNING,
                confidence_modifier=0.9,
                reason=f"Could not parse book codes from references: {type_ref}, {antitype_ref}",
                recoverable=True,
            )

        # Get positions in canonical order
        try:
            type_pos = order.index(type_book.upper())
        except ValueError:
            type_pos = -1

        try:
            antitype_pos = order.index(antitype_book.upper())
        except ValueError:
            antitype_pos = -1

        if type_pos == -1 or antitype_pos == -1:
            return ConstraintResult(
                passed=False,
                constraint_type=ConstraintType.CHRONOLOGICAL_PRIORITY,
                violation_severity=ConstraintViolationSeverity.WARNING,
                confidence_modifier=0.9,
                reason=f"Book not found in canonical order: {type_book} or {antitype_book}",
                recoverable=True,
            )

        # Type must precede antitype
        if type_pos >= antitype_pos:
            logger.warning(
                f"Chronological violation: {type_ref} ({type_pos}) >= {antitype_ref} ({antitype_pos})"
            )
            return ConstraintResult(
                passed=False,
                constraint_type=ConstraintType.CHRONOLOGICAL_PRIORITY,
                violation_severity=ConstraintViolationSeverity.IMPOSSIBLE,
                confidence_modifier=0.0,
                reason=f"Type ({type_ref}) does not precede antitype ({antitype_ref}) in canonical order",
                evidence=[f"{type_book}@{type_pos} >= {antitype_book}@{antitype_pos}"],
                recoverable=False,  # Hard constraint - cannot be recovered
            )

        return ConstraintResult(
            passed=True,
            constraint_type=ConstraintType.CHRONOLOGICAL_PRIORITY,
            confidence_modifier=1.0,
            reason=f"Type ({type_ref}) correctly precedes antitype ({antitype_ref})",
            evidence=[f"{type_book}@{type_pos} < {antitype_book}@{antitype_pos}"],
        )

    def validate_typological_escalation(
        self,
        type_element: Dict[str, Any],
        antitype_element: Dict[str, Any],
        type_context: Dict[str, Any],
        antitype_context: Dict[str, Any]
    ) -> ConstraintResult:
        """
        Validate that antitype exceeds type in scope and magnitude.

        Universal patristic principle: the antitype must be greater
        than the type it fulfills.

        Args:
            type_element: Type element data
            antitype_element: Antitype element data
            type_context: Context for type analysis
            antitype_context: Context for antitype analysis

        Returns:
            ConstraintResult with escalation analysis
        """
        if not self.enable_escalation:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.TYPOLOGICAL_ESCALATION,
                reason="Escalation validation disabled",
            )

        # Analyze scope
        type_scope = self.scope_analyzer.analyze_scope(type_element, type_context)
        antitype_scope = self.scope_analyzer.analyze_scope(antitype_element, antitype_context)

        # Analyze magnitude
        type_magnitude = self.scope_analyzer.calculate_magnitude(type_element)
        antitype_magnitude = self.scope_analyzer.calculate_magnitude(antitype_element)

        # Analyze fulfillment completeness
        completeness = self.scope_analyzer.analyze_fulfillment_completeness(
            type_element, antitype_element
        )

        # Calculate escalation ratio
        escalation_ratio = antitype_magnitude / max(type_magnitude, 1.0)

        # Scope comparison (cosmic > universal > national > local)
        scope_order = [Scope.LOCAL, Scope.NATIONAL, Scope.UNIVERSAL, Scope.COSMIC]
        type_scope_idx = scope_order.index(type_scope)
        antitype_scope_idx = scope_order.index(antitype_scope)
        scope_escalation = antitype_scope_idx >= type_scope_idx

        evidence = [
            f"Type scope: {type_scope.value}, magnitude: {type_magnitude:.1f}",
            f"Antitype scope: {antitype_scope.value}, magnitude: {antitype_magnitude:.1f}",
            f"Escalation ratio: {escalation_ratio:.2f}",
            f"Completeness: {completeness:.2f}",
        ]

        # Determine result
        if escalation_ratio < self.escalation_critical_threshold and not scope_escalation:
            return ConstraintResult(
                passed=False,
                constraint_type=ConstraintType.TYPOLOGICAL_ESCALATION,
                violation_severity=ConstraintViolationSeverity.CRITICAL,
                confidence_modifier=0.3,
                reason="Antitype does not exceed type in scope or magnitude",
                evidence=evidence,
                recoverable=True,
            )

        if escalation_ratio < self.escalation_critical_threshold:
            return ConstraintResult(
                passed=False,
                constraint_type=ConstraintType.TYPOLOGICAL_ESCALATION,
                violation_severity=ConstraintViolationSeverity.SOFT,
                confidence_modifier=0.7,
                reason="Antitype magnitude is below critical threshold",
                evidence=evidence,
                recoverable=True,
            )

        if escalation_ratio >= self.escalation_boost_threshold and scope_escalation:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.TYPOLOGICAL_ESCALATION,
                violation_severity=ConstraintViolationSeverity.BOOST,
                confidence_modifier=1.2,
                reason="Antitype significantly exceeds type - strong typological connection",
                evidence=evidence,
            )

        return ConstraintResult(
            passed=True,
            constraint_type=ConstraintType.TYPOLOGICAL_ESCALATION,
            confidence_modifier=1.0,
            reason="Antitype adequately exceeds type",
            evidence=evidence,
        )

    def validate_prophetic_coherence(
        self,
        promise_verse: str,
        fulfillment_verse: str,
        promise_semantics: Dict[str, Any],
        fulfillment_semantics: Dict[str, Any]
    ) -> ConstraintResult:
        """
        Validate that fulfillment extends/completes promise without contradiction.

        Patristic principle: prophecy fulfillment must extend the promise,
        never contradict it.

        Args:
            promise_verse: Promise verse reference
            fulfillment_verse: Fulfillment verse reference
            promise_semantics: Semantic analysis of promise
            fulfillment_semantics: Semantic analysis of fulfillment

        Returns:
            ConstraintResult with coherence analysis
        """
        if not self.enable_prophetic:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.PROPHETIC_COHERENCE,
                reason="Prophetic coherence validation disabled",
            )

        # Check for contradictions
        contradictions = self.coherence_checker.detect_contradictions(
            promise_semantics, fulfillment_semantics
        )

        # Check for extensions
        extensions = self.coherence_checker.detect_extensions(
            promise_semantics, fulfillment_semantics
        )

        # Check entailment
        entails, confidence = self.coherence_checker.check_entailment(
            promise_semantics, fulfillment_semantics
        )

        evidence = []
        if extensions:
            evidence.extend([f"Extension: {ext}" for ext in extensions])
        if contradictions:
            evidence.extend([f"Contradiction: {cont}" for cont in contradictions])

        if contradictions:
            return ConstraintResult(
                passed=False,
                constraint_type=ConstraintType.PROPHETIC_COHERENCE,
                violation_severity=ConstraintViolationSeverity.CRITICAL,
                confidence_modifier=0.2,
                reason=f"Semantic contradictions detected between {promise_verse} and {fulfillment_verse}",
                evidence=evidence,
                recoverable=True,
            )

        if not entails and not extensions:
            return ConstraintResult(
                passed=False,
                constraint_type=ConstraintType.PROPHETIC_COHERENCE,
                violation_severity=ConstraintViolationSeverity.SOFT,
                confidence_modifier=0.7,
                reason="Fulfillment does not clearly entail or extend promise",
                evidence=evidence,
                recoverable=True,
            )

        if extensions:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.PROPHETIC_COHERENCE,
                violation_severity=ConstraintViolationSeverity.BOOST,
                confidence_modifier=1.15,
                reason="Fulfillment extends promise as expected in typological reading",
                evidence=evidence,
            )

        return ConstraintResult(
            passed=True,
            constraint_type=ConstraintType.PROPHETIC_COHERENCE,
            confidence_modifier=1.0,
            reason="No contradictions detected in prophetic fulfillment",
            evidence=evidence,
        )

    def validate_christological_warrant(
        self,
        ot_verse: str,
        christological_claim: str,
        nt_quotations: List[str],
        patristic_witnesses: List[str]
    ) -> ConstraintResult:
        """
        Validate that christological OT reading has apostolic/patristic warrant.

        Patristic principle: Christological OT readings require either
        apostolic use (NT quotation) OR patristic consensus.

        Args:
            ot_verse: OT verse being read christologically
            christological_claim: The christological interpretation
            nt_quotations: List of NT passages quoting this OT verse
            patristic_witnesses: List of Church Fathers supporting this reading

        Returns:
            ConstraintResult with warrant assessment
        """
        if not self.enable_warrant:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.CHRISTOLOGICAL_WARRANT,
                reason="Christological warrant validation disabled",
            )

        evidence = []

        # Check for apostolic (NT) warrant
        has_apostolic = len(nt_quotations) > 0
        if has_apostolic:
            evidence.append(f"Apostolic warrant: {len(nt_quotations)} NT quotation(s)")
            evidence.extend([f"  - {q}" for q in nt_quotations[:3]])  # First 3

        # Check for patristic witness
        major_witnesses = [
            f for f in patristic_witnesses
            if any(major in f for major in self.MAJOR_FATHERS)
        ]
        has_patristic_consensus = len(major_witnesses) >= self.min_patristic_witnesses

        if has_patristic_consensus:
            evidence.append(f"Patristic consensus: {len(major_witnesses)} major Father(s)")
            evidence.extend([f"  - {w}" for w in major_witnesses[:3]])

        # Determine result
        if has_apostolic:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.CHRISTOLOGICAL_WARRANT,
                violation_severity=ConstraintViolationSeverity.BOOST,
                confidence_modifier=self.apostolic_boost_factor,
                reason=f"Apostolic warrant for christological reading of {ot_verse}",
                evidence=evidence,
            )

        if has_patristic_consensus:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.CHRISTOLOGICAL_WARRANT,
                violation_severity=ConstraintViolationSeverity.BOOST,
                confidence_modifier=self.patristic_boost_factor,
                reason=f"Patristic consensus for christological reading of {ot_verse}",
                evidence=evidence,
            )

        if patristic_witnesses:
            # Some patristic support but not consensus
            evidence.append(f"Partial patristic support: {len(patristic_witnesses)} witness(es)")
            return ConstraintResult(
                passed=False,
                constraint_type=ConstraintType.CHRISTOLOGICAL_WARRANT,
                violation_severity=ConstraintViolationSeverity.WARNING,
                confidence_modifier=0.9,
                reason="Christological reading has some support but lacks consensus",
                evidence=evidence,
                recoverable=True,
            )

        # No warrant at all
        return ConstraintResult(
            passed=False,
            constraint_type=ConstraintType.CHRISTOLOGICAL_WARRANT,
            violation_severity=ConstraintViolationSeverity.CRITICAL,
            confidence_modifier=0.3,
            reason=f"Novel christological reading of {ot_verse} lacks apostolic or patristic warrant",
            evidence=["No NT quotations found", "No patristic witnesses found"],
            recoverable=True,
        )

    def validate_liturgical_amplification(
        self,
        verse_ref: str,
        liturgical_contexts: List[str]
    ) -> ConstraintResult:
        """
        Validate and boost connections with liturgical significance.

        Orthodox principle: Liturgical usage amplifies theological weight.

        Args:
            verse_ref: Verse reference
            liturgical_contexts: List of liturgical contexts (e.g., "Pascha", "lectionary")

        Returns:
            ConstraintResult with liturgical assessment
        """
        if not self.enable_liturgical:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.LITURGICAL_AMPLIFICATION,
                reason="Liturgical amplification disabled",
            )

        if not liturgical_contexts:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.LITURGICAL_AMPLIFICATION,
                confidence_modifier=1.0,
                reason="No liturgical contexts provided",
            )

        # Calculate boost based on liturgical significance
        max_boost = 1.0
        evidence = []

        for context in liturgical_contexts:
            context_lower = context.lower().replace(" ", "_")
            boost = self.LITURGICAL_CONTEXTS.get(context_lower, 1.0)
            if boost > max_boost:
                max_boost = boost
            evidence.append(f"{context}: boost factor {boost}")

        if max_boost > 1.0:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.LITURGICAL_AMPLIFICATION,
                violation_severity=ConstraintViolationSeverity.BOOST,
                confidence_modifier=min(max_boost, 1.5),  # Cap at 1.5
                reason=f"Liturgical amplification for {verse_ref}",
                evidence=evidence,
            )

        return ConstraintResult(
            passed=True,
            constraint_type=ConstraintType.LITURGICAL_AMPLIFICATION,
            confidence_modifier=1.0,
            reason="Liturgical contexts recognized but no significant boost",
            evidence=evidence,
        )

    def validate_fourfold_foundation(
        self,
        verse_ref: str,
        literal_analysis: Dict[str, Any],
        allegorical_claim: Dict[str, Any]
    ) -> ConstraintResult:
        """
        Validate that allegorical reading has literal foundation.

        Patristic principle (Origen, Cassian): Allegorical interpretation
        requires established literal sense.

        Args:
            verse_ref: Verse reference
            literal_analysis: Literal sense analysis
            allegorical_claim: Allegorical interpretation being made

        Returns:
            ConstraintResult with foundation assessment
        """
        if not self.enable_fourfold:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.FOURFOLD_FOUNDATION,
                reason="Fourfold foundation validation disabled",
            )

        # Check if literal sense is established
        has_literal = bool(literal_analysis)
        literal_complete = (
            literal_analysis.get("text") or
            literal_analysis.get("historical_context") or
            literal_analysis.get("grammatical_analysis")
        ) if has_literal else False

        # Check if allegorical reading exists
        has_allegorical = bool(allegorical_claim)
        allegorical_type = allegorical_claim.get("type", "allegorical") if has_allegorical else None

        evidence = []
        if has_literal:
            evidence.append("Literal sense established")
        else:
            evidence.append("Literal sense not provided")

        if has_allegorical:
            evidence.append(f"Allegorical reading type: {allegorical_type}")

        # If no allegorical claim, no validation needed
        if not has_allegorical:
            return ConstraintResult(
                passed=True,
                constraint_type=ConstraintType.FOURFOLD_FOUNDATION,
                confidence_modifier=1.0,
                reason="No allegorical reading to validate",
            )

        # Allegorical without literal foundation
        if has_allegorical and not literal_complete:
            return ConstraintResult(
                passed=False,
                constraint_type=ConstraintType.FOURFOLD_FOUNDATION,
                violation_severity=ConstraintViolationSeverity.WARNING,
                confidence_modifier=0.9,
                reason="Allegorical reading without established literal foundation",
                evidence=evidence,
                recoverable=True,
            )

        return ConstraintResult(
            passed=True,
            constraint_type=ConstraintType.FOURFOLD_FOUNDATION,
            confidence_modifier=1.0,
            reason="Allegorical reading properly grounded in literal sense",
            evidence=evidence,
        )

    async def validate_all_constraints(
        self,
        candidate: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[ConstraintResult]:
        """
        Run all applicable constraints for a candidate.

        Args:
            candidate: Cross-reference candidate
            context: Pipeline context

        Returns:
            List of all constraint evaluation results
        """
        results = []

        connection_type = candidate.get("connection_type", "thematic")
        source_verse = candidate.get("source_verse", candidate.get("source_ref", ""))
        target_verse = candidate.get("target_verse", candidate.get("target_ref", ""))

        # Get applicable constraints for this connection type
        applicable = self.CONSTRAINT_APPLICABILITY.get(connection_type, [])

        # Always check chronological for typological/prophetic
        if connection_type in ["typological", "prophetic"]:
            result = self.validate_chronological_priority(source_verse, target_verse)
            results.append(result)
            self._log_constraint_result(result, source_verse, target_verse)

            # If chronological fails with IMPOSSIBLE, no point checking others
            if not result.passed and result.violation_severity == ConstraintViolationSeverity.IMPOSSIBLE:
                return results

        # Typological escalation
        if ConstraintType.TYPOLOGICAL_ESCALATION in applicable:
            type_elem = context.get("type_element", {"text": source_verse})
            antitype_elem = context.get("antitype_element", {"text": target_verse})
            type_ctx = context.get("type_context", {})
            antitype_ctx = context.get("antitype_context", {})

            result = self.validate_typological_escalation(
                type_elem, antitype_elem, type_ctx, antitype_ctx
            )
            results.append(result)
            self._log_constraint_result(result, source_verse, target_verse)

        # Prophetic coherence
        if ConstraintType.PROPHETIC_COHERENCE in applicable:
            promise_sem = context.get("promise_semantics", {})
            fulfill_sem = context.get("fulfillment_semantics", {})

            result = self.validate_prophetic_coherence(
                source_verse, target_verse, promise_sem, fulfill_sem
            )
            results.append(result)
            self._log_constraint_result(result, source_verse, target_verse)

        # Christological warrant
        if ConstraintType.CHRISTOLOGICAL_WARRANT in applicable:
            nt_quotes = context.get("nt_quotations", [])
            patristic = context.get("patristic_witnesses", [])
            claim = context.get("christological_claim", "")

            result = self.validate_christological_warrant(
                source_verse, claim, nt_quotes, patristic
            )
            results.append(result)
            self._log_constraint_result(result, source_verse, target_verse)

        # Liturgical amplification
        if ConstraintType.LITURGICAL_AMPLIFICATION in applicable:
            liturgical = context.get("liturgical_contexts", [])

            result = self.validate_liturgical_amplification(source_verse, liturgical)
            results.append(result)
            self._log_constraint_result(result, source_verse, target_verse)

        # Fourfold foundation
        if ConstraintType.FOURFOLD_FOUNDATION in applicable:
            literal = context.get("literal_analysis", {})
            allegorical = context.get("allegorical_claim", {})

            result = self.validate_fourfold_foundation(source_verse, literal, allegorical)
            results.append(result)
            self._log_constraint_result(result, source_verse, target_verse)

        return results

    def calculate_composite_modifier(
        self,
        results: List[ConstraintResult]
    ) -> float:
        """
        Calculate composite confidence modifier from all constraint results.

        IMPOSSIBLE constraint → return 0.0 immediately
        Otherwise, multiply all modifiers with floor (0.1) and ceiling (1.5)

        Args:
            results: List of constraint results

        Returns:
            Composite confidence modifier
        """
        if not results:
            return 1.0

        # Check for IMPOSSIBLE violations first
        for result in results:
            if (not result.passed and
                result.violation_severity == ConstraintViolationSeverity.IMPOSSIBLE):
                logger.info("IMPOSSIBLE constraint violation - returning 0.0")
                return 0.0

        # Multiply all modifiers
        composite = 1.0
        for result in results:
            composite *= result.confidence_modifier

        # Apply floor and ceiling
        composite = max(0.1, min(1.5, composite))

        logger.debug(f"Composite modifier from {len(results)} constraints: {composite:.3f}")
        return composite

    def _extract_book_code(self, verse_ref: str) -> Optional[str]:
        """Extract book code from verse reference."""
        if not verse_ref:
            return None

        # Handle formats: "GEN.1.1", "GEN 1:1", "Genesis 1:1"
        parts = re.split(r'[.\s:]+', verse_ref)
        if parts:
            return parts[0].upper()[:3]
        return None

    def _log_constraint_result(
        self,
        result: ConstraintResult,
        source_verse: str,
        target_verse: str
    ) -> None:
        """Log constraint evaluation result with structured logging."""
        log_data = {
            "constraint_type": result.constraint_type.value,
            "source_verse": source_verse,
            "target_verse": target_verse,
            "passed": result.passed,
            "severity": result.violation_severity.value if result.violation_severity else None,
            "modifier": result.confidence_modifier,
            "reason": result.reason,
        }

        if result.passed:
            logger.debug("Constraint passed", extra=log_data)
        elif result.violation_severity == ConstraintViolationSeverity.IMPOSSIBLE:
            logger.warning("IMPOSSIBLE constraint violation", extra=log_data)
        elif result.violation_severity == ConstraintViolationSeverity.CRITICAL:
            logger.warning("Critical constraint violation", extra=log_data)
        else:
            logger.info("Constraint evaluated", extra=log_data)
