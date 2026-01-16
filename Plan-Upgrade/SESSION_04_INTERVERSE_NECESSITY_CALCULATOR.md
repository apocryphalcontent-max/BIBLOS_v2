# SESSION 04: INTER-VERSE NECESSITY CALCULATOR

## Session Overview

**Objective**: Implement the `InterVerseNecessityCalculator` engine that determines whether verse B is NECESSARY to understand verse A (not just helpful). This is the second of the Five Impossible Oracles.

**Prerequisites**:
- Session 01 complete (Mutual Transformation Metric)
- Session 03 complete (Omni-Contextual Resolver)
- Understanding of propositional logic (implications, necessities)
- Familiarity with dependency graph concepts
- Text-Fabric/Macula corpus access for syntactic analysis

---

## Part 1: Understanding the Oracle Concept

### Core Capability
Determine if verse B is **logically necessary** to understand verse A:
1. A cannot be fully understood without B
2. B provides information that A presupposes
3. Removing B leaves A with semantic gaps

### The Fundamental Distinction: Necessity vs. Helpfulness

**Helpful Connection** (Standard Cross-Reference):
- GEN.1.1 → JHN.1.1 - "Beginning" echoes, enriches understanding
- Understanding GEN.1.1 is possible without JHN.1.1
- Both verses are **semantically complete** independently

**Necessary Connection** (What This Engine Detects):
- HEB.11.17-19 → GEN.22 - Hebrews REQUIRES Genesis Isaac narrative
- "Abraham reasoned that God could raise the dead" makes NO SENSE without the Akedah
- This is a logical dependency, not just a thematic echo
- Verse A is **semantically incomplete** without verse B

### Mathematical Formalization of Necessity

**Definition**: Verse B is necessary for verse A if and only if:

```
Necessity(A, B) = P(Understanding(A) | ¬Knowledge(B)) << P(Understanding(A) | Knowledge(B))
```

More formally, we define the **Comprehensibility Function** C:

```python
def comprehensibility(verse_a: str, available_context: Set[str]) -> float:
    """
    C(A, Ω) = measure of how fully verse A can be understood
    given context set Ω (available verses)

    Returns 0.0 (incomprehensible) to 1.0 (fully understood)
    """
    semantic_gaps = identify_gaps(verse_a)
    filled_gaps = count_gaps_filled_by_context(semantic_gaps, available_context)

    # Weighted by gap criticality
    return sum(g.filled * g.criticality for g in semantic_gaps) / sum(g.criticality for g in semantic_gaps)
```

**Necessity Score Definition**:
```
Necessity(A, B) = C(A, Ω ∪ {B}) - C(A, Ω \ {B})
                 ─────────────────────────────────
                        C(A, Ω ∪ {B})

Where Ω = full canonical context
```

This measures the **proportional comprehension loss** when B is removed.

### Canonical Example: HEB.11.17-19 ↔ GEN.22.1-14

```
HEB.11.17: "By faith Abraham, when he was tested, offered up Isaac"
HEB.11.18: "of whom it was said, 'In Isaac your seed shall be called'"
HEB.11.19: "concluding that God was able to raise him up, even from the dead"

Semantic Gap Analysis:
┌─────────────────────────────────────────────────────────────────────┐
│ GAP TYPE          │ ELEMENT        │ SEVERITY │ RESOLVER           │
├───────────────────┼────────────────┼──────────┼────────────────────┤
│ ENTITY            │ "Abraham"      │ 0.95     │ GEN.12.1 (intro)   │
│ ENTITY            │ "Isaac"        │ 0.95     │ GEN.21.3 (birth)   │
│ EVENT             │ "tested"       │ 0.98     │ GEN.22.1-2         │
│ EVENT             │ "offered up"   │ 0.95     │ GEN.22.9-10        │
│ PROMISE           │ "seed promise" │ 0.90     │ GEN.12.7, 17.19    │
│ INFERENCE         │ "resurrection" │ 0.85     │ GEN.22.5, 12       │
│ QUOTATION         │ "it was said"  │ 1.00     │ GEN.21.12          │
└─────────────────────────────────────────────────────────────────────┘

Comprehensibility WITHOUT GEN.22 context: 0.08 (near-zero)
Comprehensibility WITH GEN.22 context: 0.98 (near-complete)

Necessity Score = (0.98 - 0.08) / 0.98 = 0.918 → ABSOLUTE
```

### Necessity Type Taxonomy

```python
class NecessityType(Enum):
    """
    Exhaustive taxonomy of inter-verse necessity relationships.
    Each type has distinct detection algorithms and scoring weights.
    """

    # EXPLICIT LINGUISTIC DEPENDENCY
    REFERENTIAL = "referential"
    # Verse A explicitly quotes or references B
    # Detection: Citation formulas, quotation markers
    # Example: MAT.1.23 → ISA.7.14 ("as it is written")
    # Weight: 1.0 (highest - explicit linguistic marker)

    # IMPLICIT KNOWLEDGE DEPENDENCY
    PRESUPPOSITIONAL = "presuppositional"
    # Verse A assumes knowledge established in B without explicit reference
    # Detection: Definite NPs referring to entities introduced elsewhere
    # Example: HEB.11.17 → GEN.22 (assumes Akedah narrative)
    # Weight: 0.95 (very high - semantic incompleteness without source)

    # LOGICAL ARGUMENT DEPENDENCY
    ARGUMENTATIVE = "argumentative"
    # Verse A's argument logically depends on premises in B
    # Detection: Discourse markers (therefore, thus), logical structure
    # Example: ROM.4.3 → GEN.15.6 (Abraham's faith argument)
    # Weight: 0.90 (high - argument invalid without premise)

    # TERMINOLOGICAL DEPENDENCY
    DEFINITIONAL = "definitional"
    # Key terms in A require definition/context from B
    # Detection: Technical terms, hapax legomena, specialized vocabulary
    # Example: ROM.3.25 (ἱλαστήριον) → LEV.16 (mercy seat ritual)
    # Weight: 0.85 (significant - term opaque without definition)

    # NARRATIVE SEQUENCE DEPENDENCY
    NARRATIVE = "narrative"
    # Story in A requires prior narrative events in B
    # Detection: Temporal markers, anaphoric reference to events
    # Example: 2SAM.12.7 → 2SAM.11 (Nathan's "You are the man")
    # Weight: 0.80 (important - narrative context required)

    # COVENANT PROMISE-FULFILLMENT CHAIN
    COVENANTAL = "covenantal"
    # Covenant invocation requires understanding of original covenant
    # Detection: Covenant formula references, promise language
    # Example: LUK.1.55 → GEN.12.1-3, 17.1-8 (Abrahamic covenant)
    # Weight: 0.90 (high - covenant identity at stake)

    # PROPHECY-FULFILLMENT CHAIN
    PROPHETIC = "prophetic"
    # Fulfillment text requires prophecy for full meaning
    # Detection: Fulfillment formulas, typological markers
    # Example: MAT.2.15 → HOS.11.1 ("Out of Egypt I called my son")
    # Weight: 0.88 (high - prophetic framework essential)

    # LITURGICAL/RITUAL DEPENDENCY
    LITURGICAL = "liturgical"
    # Verse references liturgical practice defined elsewhere
    # Detection: Ritual terms, feast names, worship vocabulary
    # Example: JHN.7.37-38 → LEV.23.36, NEH.8.18 (Feast of Tabernacles)
    # Weight: 0.75 (moderate - liturgical context illuminating)

    # GENEALOGICAL/IDENTITY DEPENDENCY
    GENEALOGICAL = "genealogical"
    # Identity claims require genealogical establishment
    # Detection: Patronymic references, tribal identifications
    # Example: REV.5.5 ("Lion of Judah") → GEN.49.9-10
    # Weight: 0.70 (moderate - identity context needed)
```

### Necessity Strength Classification with Confidence Intervals

```python
class NecessityStrength(Enum):
    """
    Classification of necessity strength with statistical rigor.
    Each level corresponds to comprehension loss percentage.
    """

    ABSOLUTE = "absolute"
    # Score: >0.90 | CI: [0.85, 1.0]
    # Comprehension Loss: >90%
    # Verse A is essentially incomprehensible without B
    # Example: HEB.11.17 without GEN.22

    STRONG = "strong"
    # Score: 0.70-0.90 | CI: [0.65, 0.95]
    # Comprehension Loss: 70-90%
    # Major meaning loss, but surface reading possible
    # Example: MAT.1.23 without ISA.7.14 (quotation)

    MODERATE = "moderate"
    # Score: 0.50-0.70 | CI: [0.45, 0.75]
    # Comprehension Loss: 50-70%
    # Significant context loss, partial understanding possible
    # Example: ROM.3.25 without LEV.16 (propitiation term)

    WEAK = "weak"
    # Score: 0.30-0.50 | CI: [0.25, 0.55]
    # Comprehension Loss: 30-50%
    # Helpful context, but verse mostly self-contained
    # Example: PSA.23.1 without EXO.3.1 (shepherd imagery)

    NONE = "none"
    # Score: <0.30 | CI: [0.0, 0.35]
    # Comprehension Loss: <30%
    # No necessity relationship, merely thematic
    # Example: GEN.1.1 without JHN.1.1 (parallel themes)
```

---

## Part 2: File Creation Specification

### File: `ml/engines/necessity_calculator.py`

**Location**: `ml/engines/` (directory created in Session 03)

**Dependencies to Import**:
```python
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Protocol
from collections import defaultdict
import hashlib

import numpy as np
import networkx as nx
from scipy.stats import beta as beta_dist

# Internal imports
from ml.engines.omni_contextual_resolver import OmniContextualResolver
from ml.embeddings.generator import EmbeddingGenerator
from db.neo4j_client import Neo4jClient
from db.redis_client import RedisClient
from data.schemas import VerseSchema, CrossReferenceSchema
from config import NecessityCalculatorConfig
```

### Core Data Structures

#### 1. `SemanticGap` (Dataclass) - Enhanced

```python
@dataclass
class SemanticGap:
    """
    Represents a semantic gap in verse A that creates dependency on external context.

    A semantic gap is a conceptual "hole" in the verse that requires information
    from another verse to fill. The gap model follows formal semantics:

    Gap = (Type, Trigger, Scope, Criticality, Candidates)
    """

    gap_id: str  # Unique identifier: hash(verse_id + trigger_text)
    gap_type: GapType  # Enum: ENTITY, EVENT, CONCEPT, TERM, QUOTATION, etc.

    # The text in verse A that creates the gap
    trigger_text: str  # E.g., "Abraham" (undefined entity)
    trigger_span: Tuple[int, int]  # Character offsets in verse

    # Syntactic analysis of the gap
    syntactic_role: SyntacticRole  # SUBJECT, OBJECT, MODIFIER, etc.
    clause_type: ClauseType  # MAIN, SUBORDINATE, CONDITIONAL, RELATIVE
    is_focus: bool  # Is this gap in the information focus position?

    # Semantic analysis
    description: str  # Human-readable: "Reference to patriarch Abraham"
    semantic_frame: str  # FrameNet-style: "Historical_figure"
    presupposition_type: PresuppositionType  # EXISTENTIAL, FACTIVE, etc.

    # Resolution data
    resolution_candidates: List[ResolutionCandidate] = field(default_factory=list)
    best_resolution: Optional[str] = None  # Verse ID that best fills this gap

    # Scoring
    base_severity: float = 0.5  # How critical is this gap (0-1)
    contextual_severity: float = 0.5  # Severity adjusted by syntactic position
    confidence: float = 0.8  # Confidence in gap identification

    def compute_final_severity(self) -> float:
        """
        Compute final gap severity using weighted combination.

        Severity is higher when:
        - Gap is in main clause (not subordinate)
        - Gap is in subject/object position (not modifier)
        - Gap is in information focus position
        - Gap type is ENTITY or EVENT (not just CONCEPT)
        """
        weights = {
            'base': 0.40,
            'syntactic': 0.25,
            'focus': 0.20,
            'clause': 0.15
        }

        syntactic_bonus = {
            SyntacticRole.SUBJECT: 0.3,
            SyntacticRole.DIRECT_OBJECT: 0.25,
            SyntacticRole.INDIRECT_OBJECT: 0.15,
            SyntacticRole.PREDICATE: 0.2,
            SyntacticRole.MODIFIER: 0.0,
            SyntacticRole.ADJUNCT: -0.1
        }.get(self.syntactic_role, 0.0)

        clause_modifier = {
            ClauseType.MAIN: 0.3,
            ClauseType.COORDINATE: 0.2,
            ClauseType.SUBORDINATE: 0.0,
            ClauseType.CONDITIONAL: -0.1,
            ClauseType.RELATIVE: -0.05
        }.get(self.clause_type, 0.0)

        focus_bonus = 0.2 if self.is_focus else 0.0

        severity = (
            weights['base'] * self.base_severity +
            weights['syntactic'] * (0.5 + syntactic_bonus) +
            weights['focus'] * (0.5 + focus_bonus) +
            weights['clause'] * (0.5 + clause_modifier)
        )

        return min(1.0, max(0.0, severity))


@dataclass
class ResolutionCandidate:
    """A potential verse that could fill a semantic gap."""
    verse_id: str
    resolution_score: float  # How well this verse fills the gap (0-1)
    resolution_type: str  # "introduces", "defines", "narrates", "explains"
    evidence: List[str]  # Textual evidence for resolution
    confidence: float


class GapType(Enum):
    """Taxonomy of semantic gap types with detection signatures."""

    ENTITY_PERSON = "entity_person"
    # Named person assumed to be known
    # Signature: Proper noun without introduction
    # Example: "Abraham" in HEB.11.17

    ENTITY_PLACE = "entity_place"
    # Named place assumed to be known
    # Signature: Proper noun (toponym) without context
    # Example: "Mount Moriah" in 2CHR.3.1

    ENTITY_OBJECT = "entity_object"
    # Named object/artifact assumed known
    # Signature: Definite NP for specific object
    # Example: "the ark of the covenant"

    EVENT_HISTORICAL = "event_historical"
    # Reference to historical event not narrated
    # Signature: Temporal reference, past tense event
    # Example: "when he was tested" in HEB.11.17

    EVENT_RITUAL = "event_ritual"
    # Reference to ritual/liturgical event
    # Signature: Ritual terminology, feast names
    # Example: "the day of atonement"

    CONCEPT_THEOLOGICAL = "concept_theological"
    # Theological concept requiring background
    # Signature: Abstract theological noun
    # Example: "propitiation" (ἱλαστήριον)

    CONCEPT_COVENANTAL = "concept_covenantal"
    # Covenant-related concept
    # Signature: Covenant terminology
    # Example: "the promise to Abraham"

    TERM_TECHNICAL = "term_technical"
    # Technical term requiring definition
    # Signature: Specialized vocabulary
    # Example: "the Nazirite vow"

    QUOTATION_EXPLICIT = "quotation_explicit"
    # Explicit quotation requiring source
    # Signature: Citation formula
    # Example: "as it is written"

    QUOTATION_ALLUSION = "quotation_allusion"
    # Allusion to text without citation
    # Signature: Verbal parallel, echo
    # Example: "In the beginning was the Word"

    PRESUPPOSITION_FACTIVE = "presupposition_factive"
    # Factive verb presupposing truth of complement
    # Signature: "knew that", "remembered that"
    # Example: "knowing that God had sworn"

    PRESUPPOSITION_EXISTENTIAL = "presupposition_existential"
    # Existential presupposition (entity exists)
    # Signature: Definite NP in object position
    # Example: "his only son Isaac"


class SyntacticRole(Enum):
    """Syntactic roles for gap severity calculation."""
    SUBJECT = "subject"
    DIRECT_OBJECT = "direct_object"
    INDIRECT_OBJECT = "indirect_object"
    PREDICATE = "predicate"
    MODIFIER = "modifier"
    ADJUNCT = "adjunct"
    VOCATIVE = "vocative"
    APPOSITIVE = "appositive"


class ClauseType(Enum):
    """Clause types for context sensitivity."""
    MAIN = "main"
    COORDINATE = "coordinate"
    SUBORDINATE = "subordinate"
    CONDITIONAL = "conditional"
    RELATIVE = "relative"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    PURPOSE = "purpose"


class PresuppositionType(Enum):
    """Formal presupposition types from linguistic theory."""
    EXISTENTIAL = "existential"  # The X exists
    FACTIVE = "factive"  # Complement is true
    LEXICAL = "lexical"  # Word meaning presupposes
    STRUCTURAL = "structural"  # Construction presupposes
    COUNTERFACTUAL = "counterfactual"  # Contrary to fact
```

#### 2. `NecessityAnalysisResult` (Dataclass) - Enhanced

```python
@dataclass
class NecessityAnalysisResult:
    """
    Complete result of necessity analysis between two verses.

    This is the canonical output of the InterVerseNecessityCalculator,
    providing all metrics, evidence, and metadata for the necessity relationship.
    """

    # Primary identification
    analysis_id: str  # Unique ID for this analysis
    source_verse: str  # Verse A (the dependent verse)
    target_verse: str  # Verse B (the required verse)
    timestamp: str  # ISO timestamp of analysis

    # Primary metrics
    necessity_score: float  # Overall necessity (0-1)
    necessity_type: NecessityType  # Primary type of necessity
    strength: NecessityStrength  # Classification (ABSOLUTE, STRONG, etc.)
    confidence: float  # Confidence in analysis (0-1)

    # Statistical measures
    score_distribution: ScoreDistribution  # Full statistical distribution
    confidence_interval: Tuple[float, float]  # 95% CI for necessity score

    # Gap analysis
    semantic_gaps: List[SemanticGap]  # All identified gaps in source verse
    gaps_filled_by_target: int  # How many gaps target verse fills
    total_gaps: int  # Total gaps in source verse
    gap_coverage: float  # gaps_filled / total_gaps
    weighted_severity_filled: float  # Sum of filled gap severities

    # Presupposition analysis
    presuppositions: List[Presupposition]  # Formal presuppositions
    presuppositions_satisfied: int  # How many target satisfies

    # Reference analysis
    explicit_references: List[ExplicitReference]  # Direct textual references
    has_citation_formula: bool  # Contains "as it is written" etc.

    # Dependency graph context
    dependency_chain: List[str]  # Full chain A → ... → B
    chain_length: int  # Length of shortest dependency path
    is_direct_dependency: bool  # True if chain_length == 1

    # Bidirectionality analysis
    bidirectional: bool  # Does B also need A?
    reverse_necessity_score: float  # Necessity(B, A)
    mutual_necessity: bool  # Both need each other

    # Explanation
    reasoning: str  # Human-readable explanation
    evidence_summary: List[str]  # Key evidence points

    # Metadata
    computation_time_ms: float  # Time to compute
    cache_hit: bool  # Was this retrieved from cache
    model_version: str  # Version of necessity model used

    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j relationship properties."""
        return {
            "necessity_score": self.necessity_score,
            "necessity_type": self.necessity_type.value,
            "strength": self.strength.value,
            "confidence": self.confidence,
            "gaps_filled": self.gaps_filled_by_target,
            "has_citation": self.has_citation_formula,
            "is_direct": self.is_direct_dependency,
            "bidirectional": self.bidirectional,
            "computed_at": self.timestamp
        }


@dataclass
class ScoreDistribution:
    """Statistical distribution of necessity score."""
    mean: float
    std: float
    alpha: float  # Beta distribution alpha
    beta: float  # Beta distribution beta
    samples: int  # Number of evidence samples

    def confidence_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """Compute credible interval using Beta distribution."""
        lower = beta_dist.ppf((1 - level) / 2, self.alpha, self.beta)
        upper = beta_dist.ppf((1 + level) / 2, self.alpha, self.beta)
        return (lower, upper)


@dataclass
class Presupposition:
    """Formal linguistic presupposition."""
    presupposition_id: str
    ptype: PresuppositionType
    trigger: str  # The linguistic trigger
    content: str  # What is presupposed
    source_text: str  # Text in verse A
    satisfied_by: Optional[str] = None  # Verse ID that satisfies
    satisfaction_score: float = 0.0


@dataclass
class ExplicitReference:
    """An explicit textual reference in verse A."""
    reference_id: str
    reference_text: str  # The reference text in A
    formula_type: str  # "quotation", "allusion", "echo"
    target_verse: str  # The verse being referenced
    confidence: float
    verbal_overlap: float  # Lexical similarity
```

#### 3. `DependencyGraph` (Enhanced)

```python
@dataclass
class DependencyGraph:
    """
    A directed graph of necessity relationships between verses.

    This represents the web of inter-verse dependencies, allowing:
    - Shortest path queries (minimum context needed)
    - Strongly connected component detection (mutual dependencies)
    - PageRank-style importance scoring
    - Community detection (related verse clusters)
    """

    graph: nx.DiGraph  # NetworkX directed graph

    # Computed properties
    root_verses: List[str] = field(default_factory=list)  # No incoming necessity
    terminal_verses: List[str] = field(default_factory=list)  # No outgoing necessity
    strongly_connected: List[List[str]] = field(default_factory=list)  # SCCs

    # Statistics
    total_nodes: int = 0
    total_edges: int = 0
    average_in_degree: float = 0.0
    average_out_degree: float = 0.0
    max_chain_length: int = 0

    @classmethod
    def from_necessity_results(
        cls,
        results: List[NecessityAnalysisResult],
        min_score: float = 0.3
    ) -> "DependencyGraph":
        """Construct graph from necessity analysis results."""
        graph = nx.DiGraph()

        for result in results:
            if result.necessity_score >= min_score:
                graph.add_edge(
                    result.source_verse,
                    result.target_verse,
                    weight=result.necessity_score,
                    type=result.necessity_type.value,
                    strength=result.strength.value
                )

        instance = cls(graph=graph)
        instance._compute_properties()
        return instance

    def _compute_properties(self):
        """Compute derived graph properties."""
        self.total_nodes = self.graph.number_of_nodes()
        self.total_edges = self.graph.number_of_edges()

        if self.total_nodes > 0:
            self.average_in_degree = self.total_edges / self.total_nodes
            self.average_out_degree = self.total_edges / self.total_nodes

        # Find roots (verses that don't depend on anything)
        self.root_verses = [
            n for n in self.graph.nodes()
            if self.graph.out_degree(n) == 0
        ]

        # Find terminals (verses that nothing depends on)
        self.terminal_verses = [
            n for n in self.graph.nodes()
            if self.graph.in_degree(n) == 0
        ]

        # Find strongly connected components (mutual dependencies)
        self.strongly_connected = [
            list(scc) for scc in nx.strongly_connected_components(self.graph)
            if len(scc) > 1  # Only multi-verse SCCs
        ]

        # Compute max chain length
        if self.total_nodes > 0:
            try:
                self.max_chain_length = nx.dag_longest_path_length(self.graph)
            except nx.NetworkXError:
                # Graph has cycles, use different approach
                self.max_chain_length = self._estimate_max_chain()

    def _estimate_max_chain(self) -> int:
        """Estimate max chain length for cyclic graphs."""
        max_length = 0
        for source in self.terminal_verses[:10]:  # Sample from terminals
            for target in self.root_verses[:10]:  # Sample from roots
                try:
                    path = nx.shortest_path(self.graph, source, target)
                    max_length = max(max_length, len(path) - 1)
                except nx.NetworkXNoPath:
                    continue
        return max_length

    def find_necessity_chain(self, verse_a: str, verse_b: str) -> List[str]:
        """Find shortest necessity chain from A to B."""
        try:
            return nx.shortest_path(self.graph, verse_a, verse_b)
        except nx.NetworkXNoPath:
            return []

    def get_required_context(self, verse: str, depth: int = 3) -> Set[str]:
        """Get all verses required to understand a given verse."""
        required = set()
        current_layer = {verse}

        for _ in range(depth):
            next_layer = set()
            for v in current_layer:
                # Get all verses this verse depends on
                successors = self.graph.successors(v)
                next_layer.update(successors)
            required.update(next_layer)
            current_layer = next_layer

        return required

    def compute_importance_scores(self) -> Dict[str, float]:
        """Compute PageRank-style importance scores for verses."""
        # Reverse graph: edges point TO dependent verses
        reversed_graph = self.graph.reverse()

        # PageRank on reversed graph
        # High score = many verses depend on this one
        return nx.pagerank(reversed_graph, weight='weight')

    def to_cypher_statements(self) -> List[str]:
        """Generate Cypher statements to create this graph in Neo4j."""
        statements = []

        for source, target, data in self.graph.edges(data=True):
            statements.append(f"""
                MATCH (a:Verse {{id: '{source}'}})
                MATCH (b:Verse {{id: '{target}'}})
                MERGE (a)-[r:NECESSITATES]->(b)
                SET r.score = {data.get('weight', 0.5)},
                    r.type = '{data.get('type', 'unknown')}',
                    r.strength = '{data.get('strength', 'moderate')}'
            """)

        return statements
```

---

## Part 3: Necessity Detection Algorithms

### Algorithm 1: Advanced Reference Extraction

```python
class ReferenceExtractor:
    """
    Extracts explicit references from verse text using multiple detection strategies.

    Detection layers:
    1. Citation formula patterns (regex)
    2. Named entity linking to introduction verses
    3. Quotation detection via text alignment
    4. Verbal echo detection via embedding similarity
    """

    # Citation formulas in multiple languages
    CITATION_FORMULAS = {
        'greek': [
            r'καθὼς\s+γέγραπται',  # "as it is written"
            r'ἵνα\s+πληρωθῇ',  # "that it might be fulfilled"
            r'ὡς\s+εἶπεν',  # "as he said"
            r'τὸ\s+ῥηθὲν',  # "that which was spoken"
            r'λέγει\s+γὰρ\s+ἡ\s+γραφή',  # "for the scripture says"
            r'γέγραπται\s+γάρ',  # "for it is written"
            r'καθὼς\s+εἶπεν',  # "as he said"
            r'κατὰ\s+τὸ\s+εἰρημένον',  # "according to what was said"
            r'ἐν\s+τῷ\s+\w+\s+γέγραπται',  # "in [book] it is written"
        ],
        'english': [
            r'as\s+it\s+is\s+written',
            r'scripture\s+says?',
            r'it\s+is\s+written',
            r'according\s+to\s+(?:the\s+)?(?:scripture|prophet)',
            r'the\s+prophet\s+\w+\s+(?:said|wrote|spoke)',
            r'Moses\s+(?:said|wrote|commanded)',
            r'David\s+(?:said|wrote|spoke)',
            r'as\s+(?:Isaiah|Jeremiah|Ezekiel|Daniel|Hosea|Joel|Amos|Micah|Nahum|Habakkuk|Zephaniah|Haggai|Zechariah|Malachi)\s+(?:said|wrote|prophesied)',
            r'in\s+the\s+(?:law|prophets|writings|psalms)',
            r'that\s+(?:it\s+)?might\s+be\s+fulfilled',
            r'this\s+was\s+to\s+fulfill',
        ],
        'hebrew': [
            r'כַּכָּתוּב',  # "as it is written"
            r'כַּאֲשֶׁר\s+דִּבֶּר',  # "as he spoke"
            r'אֲשֶׁר\s+אָמַר',  # "which he said"
        ]
    }

    # Reference pronouns that create anaphoric gaps
    ANAPHORIC_TRIGGERS = {
        'demonstrative': [r'\bthis\b', r'\bthat\b', r'\bthese\b', r'\bthose\b'],
        'definite_article': [r'\bthe\s+\w+'],  # "the prophet", "the promise"
        'pronoun': [r'\bhe\b', r'\bshe\b', r'\bhim\b', r'\bher\b', r'\bhis\b'],
        'relative': [r'\bwho\b', r'\bwhom\b', r'\bwhich\b', r'\bwhat\b'],
    }

    async def extract_explicit_references(
        self,
        verse_text: str,
        verse_id: str,
        lang: str = 'greek'
    ) -> List[ExplicitReference]:
        """
        Extract all explicit references from verse text.

        Process:
        1. Scan for citation formulas
        2. Extract quoted material
        3. Match to source verses via text alignment
        4. Score by verbal overlap
        """
        references = []

        # Step 1: Find citation formulas
        formulas = self.CITATION_FORMULAS.get(lang, []) + self.CITATION_FORMULAS['english']

        for formula in formulas:
            matches = re.finditer(formula, verse_text, re.IGNORECASE)
            for match in matches:
                # Extract the material following the formula
                following_text = verse_text[match.end():match.end() + 200]

                # Find the best matching source verse
                source_match = await self._find_source_verse(following_text, verse_id)

                if source_match:
                    references.append(ExplicitReference(
                        reference_id=f"ref_{verse_id}_{len(references)}",
                        reference_text=match.group(0) + following_text[:50],
                        formula_type="quotation",
                        target_verse=source_match.verse_id,
                        confidence=source_match.confidence,
                        verbal_overlap=source_match.overlap
                    ))

        return references

    async def _find_source_verse(
        self,
        quoted_text: str,
        context_verse: str
    ) -> Optional[SourceMatch]:
        """Find the source verse being quoted using text alignment."""
        # Use embedding similarity + verbal overlap
        candidates = await self._get_quotation_candidates(quoted_text, context_verse)

        best_match = None
        best_score = 0.0

        for candidate in candidates:
            # Compute verbal overlap (shared vocabulary)
            overlap = self._compute_verbal_overlap(quoted_text, candidate.text)

            # Compute embedding similarity
            emb_sim = await self._embedding_similarity(quoted_text, candidate.text)

            # Combined score
            score = 0.6 * overlap + 0.4 * emb_sim

            if score > best_score:
                best_score = score
                best_match = SourceMatch(
                    verse_id=candidate.verse_id,
                    confidence=score,
                    overlap=overlap
                )

        return best_match if best_score > 0.5 else None

    def _compute_verbal_overlap(self, text_a: str, text_b: str) -> float:
        """Compute lexical overlap between two texts."""
        words_a = set(self._lemmatize(text_a))
        words_b = set(self._lemmatize(text_b))

        if not words_a or not words_b:
            return 0.0

        intersection = words_a & words_b
        union = words_a | words_b

        # Jaccard similarity
        return len(intersection) / len(union)
```

### Algorithm 2: Presupposition Detection Engine

```python
class PresuppositionDetector:
    """
    Detects linguistic presuppositions that create inter-verse dependencies.

    Based on formal semantics literature:
    - Existential presuppositions (definite descriptions)
    - Factive presuppositions (know, realize, remember)
    - Lexical presuppositions (stop, continue, again)
    - Structural presuppositions (cleft sentences, wh-questions)
    """

    # Factive verbs that presuppose truth of complement
    FACTIVE_VERBS = {
        'english': [
            'know', 'realize', 'discover', 'notice', 'remember',
            'forget', 'regret', 'be aware', 'be glad', 'be sorry'
        ],
        'greek': [
            'γινώσκω', 'οἶδα', 'μιμνῄσκομαι', 'ἐπιλανθάνομαι'
        ]
    }

    # Change-of-state verbs presupposing prior state
    CHANGE_VERBS = {
        'english': [
            'stop', 'continue', 'begin', 'start', 'resume',
            'return', 'again', 'still', 'anymore'
        ],
        'greek': [
            'παύομαι', 'ἄρχομαι', 'ἐπιστρέφω', 'πάλιν'
        ]
    }

    # Discourse markers indicating prior knowledge assumed
    PRIOR_KNOWLEDGE_MARKERS = [
        r'\bof course\b',
        r'\bas you know\b',
        r'\bremember\b',
        r'\byou recall\b',
        r'\bwhen\s+\w+\s+had\b',  # "when he had done"
        r'\bafter\s+\w+\s+had\b',  # "after he had said"
        r'\btherefore\b',
        r'\bthus\b',
        r'\bso\s+then\b',
        r'\bfor\s+this\s+reason\b',
    ]

    async def detect_presuppositions(
        self,
        verse_text: str,
        syntax_tree: Optional[SyntaxTree] = None
    ) -> List[Presupposition]:
        """
        Detect all presuppositions in verse text.

        Returns list of Presupposition objects with:
        - Type of presupposition
        - The presupposed content
        - The linguistic trigger
        """
        presuppositions = []

        # 1. Existential presuppositions from definite NPs
        definites = self._extract_definite_nps(verse_text, syntax_tree)
        for definite in definites:
            presuppositions.append(Presupposition(
                presupposition_id=f"pres_{hash(definite.text)}",
                ptype=PresuppositionType.EXISTENTIAL,
                trigger=definite.text,
                content=f"There exists a unique {definite.head_noun}",
                source_text=verse_text
            ))

        # 2. Factive presuppositions
        factives = self._detect_factive_verbs(verse_text)
        for factive in factives:
            presuppositions.append(Presupposition(
                presupposition_id=f"pres_{hash(factive.complement)}",
                ptype=PresuppositionType.FACTIVE,
                trigger=factive.verb,
                content=f"It is true that: {factive.complement}",
                source_text=verse_text
            ))

        # 3. Temporal/causal presuppositions
        temporals = self._detect_temporal_presuppositions(verse_text)
        for temporal in temporals:
            presuppositions.append(Presupposition(
                presupposition_id=f"pres_{hash(temporal.prior_event)}",
                ptype=PresuppositionType.STRUCTURAL,
                trigger=temporal.marker,
                content=f"Prior event occurred: {temporal.prior_event}",
                source_text=verse_text
            ))

        return presuppositions

    def _extract_definite_nps(
        self,
        text: str,
        syntax_tree: Optional[SyntaxTree]
    ) -> List[DefiniteNP]:
        """
        Extract definite noun phrases that presuppose existence.

        "the prophet" presupposes existence of a unique prophet in context
        "his son" presupposes existence of a son belonging to referent
        """
        definites = []

        # Pattern for definite article + noun
        # Exclude "the LORD" which is a name, not a description
        pattern = r'\b(the)\s+(?!LORD|Lord|God)(\w+(?:\s+\w+)?)\b'

        for match in re.finditer(pattern, text):
            definites.append(DefiniteNP(
                text=match.group(0),
                article=match.group(1),
                head_noun=match.group(2),
                span=(match.start(), match.end())
            ))

        # Possessives
        poss_pattern = r'\b(his|her|their|its)\s+(\w+)\b'
        for match in re.finditer(poss_pattern, text):
            definites.append(DefiniteNP(
                text=match.group(0),
                article=match.group(1),
                head_noun=match.group(2),
                span=(match.start(), match.end()),
                is_possessive=True
            ))

        return definites

    def _detect_factive_verbs(self, text: str) -> List[FactiveConstruction]:
        """Detect factive verb constructions."""
        factives = []

        for verb in self.FACTIVE_VERBS['english']:
            # Pattern: verb + that-clause or verb + wh-clause
            pattern = rf'\b{verb}(?:s|ed|ing)?\s+(?:that|what|how|why|when|where)\s+([^.]+)'

            for match in re.finditer(pattern, text, re.IGNORECASE):
                factives.append(FactiveConstruction(
                    verb=verb,
                    complement=match.group(1).strip(),
                    span=(match.start(), match.end())
                ))

        return factives
```

### Algorithm 3: Advanced Gap Severity Calculation

```python
class GapSeverityCalculator:
    """
    Calculates the severity of semantic gaps using information-theoretic measures.

    Severity is high when:
    1. Gap is in informationally prominent position (focus)
    2. Gap affects the main predication (not background)
    3. Gap involves discourse-new information expectation
    4. Gap creates logical incompleteness (broken arguments)
    """

    async def calculate_severity(
        self,
        gap: SemanticGap,
        verse_text: str,
        syntax_tree: Optional[SyntaxTree] = None,
        discourse_structure: Optional[DiscourseStructure] = None
    ) -> float:
        """
        Calculate gap severity with multi-factor weighting.

        Factors:
        1. Syntactic prominence (0.30)
        2. Information structure (0.25)
        3. Gap type importance (0.25)
        4. Discourse role (0.20)
        """

        # Factor 1: Syntactic prominence
        syntactic_score = self._syntactic_prominence(gap, syntax_tree)

        # Factor 2: Information structure (topic-focus)
        info_score = self._information_structure_score(gap, syntax_tree)

        # Factor 3: Gap type importance
        type_score = self._gap_type_importance(gap.gap_type)

        # Factor 4: Discourse role
        discourse_score = self._discourse_role_score(gap, discourse_structure)

        # Weighted combination
        severity = (
            0.30 * syntactic_score +
            0.25 * info_score +
            0.25 * type_score +
            0.20 * discourse_score
        )

        return min(1.0, max(0.0, severity))

    def _syntactic_prominence(
        self,
        gap: SemanticGap,
        syntax_tree: Optional[SyntaxTree]
    ) -> float:
        """Score based on syntactic position."""

        # Default scores by role
        role_scores = {
            SyntacticRole.SUBJECT: 0.9,  # Subjects are highly prominent
            SyntacticRole.PREDICATE: 0.85,  # Predicates are core
            SyntacticRole.DIRECT_OBJECT: 0.75,  # Objects are important
            SyntacticRole.INDIRECT_OBJECT: 0.6,
            SyntacticRole.MODIFIER: 0.4,  # Modifiers less critical
            SyntacticRole.ADJUNCT: 0.3,  # Adjuncts least critical
            SyntacticRole.VOCATIVE: 0.5,
            SyntacticRole.APPOSITIVE: 0.45,
        }

        base_score = role_scores.get(gap.syntactic_role, 0.5)

        # Boost for main clause
        clause_modifier = {
            ClauseType.MAIN: 0.15,
            ClauseType.COORDINATE: 0.10,
            ClauseType.SUBORDINATE: -0.05,
            ClauseType.CONDITIONAL: -0.10,
            ClauseType.RELATIVE: -0.05,
            ClauseType.TEMPORAL: 0.0,
            ClauseType.CAUSAL: 0.05,
            ClauseType.PURPOSE: 0.0,
        }.get(gap.clause_type, 0.0)

        return min(1.0, base_score + clause_modifier)

    def _information_structure_score(
        self,
        gap: SemanticGap,
        syntax_tree: Optional[SyntaxTree]
    ) -> float:
        """Score based on information structure (topic-focus-background)."""

        if gap.is_focus:
            # Focus position = new information = highly important if missing
            return 0.95

        # Estimate based on position
        # Sentence-initial = likely topic
        # Sentence-final = likely focus/rheme
        # Sentence-medial = likely background

        # Without full syntax tree, use heuristics
        return 0.6  # Default to moderate importance

    def _gap_type_importance(self, gap_type: GapType) -> float:
        """Intrinsic importance of different gap types."""

        importance = {
            # Entity gaps are critical - can't understand without knowing WHO
            GapType.ENTITY_PERSON: 0.95,
            GapType.ENTITY_PLACE: 0.75,
            GapType.ENTITY_OBJECT: 0.70,

            # Event gaps affect narrative comprehension
            GapType.EVENT_HISTORICAL: 0.90,
            GapType.EVENT_RITUAL: 0.70,

            # Concept gaps affect meaning depth
            GapType.CONCEPT_THEOLOGICAL: 0.85,
            GapType.CONCEPT_COVENANTAL: 0.80,

            # Term gaps affect precision
            GapType.TERM_TECHNICAL: 0.65,

            # Quotation gaps are explicit dependencies
            GapType.QUOTATION_EXPLICIT: 1.0,  # Explicit reference = must have source
            GapType.QUOTATION_ALLUSION: 0.60,

            # Presupposition gaps vary
            GapType.PRESUPPOSITION_FACTIVE: 0.75,
            GapType.PRESUPPOSITION_EXISTENTIAL: 0.70,
        }

        return importance.get(gap_type, 0.5)

    def _discourse_role_score(
        self,
        gap: SemanticGap,
        discourse_structure: Optional[DiscourseStructure]
    ) -> float:
        """Score based on discourse role of containing segment."""

        if discourse_structure is None:
            return 0.5  # Default

        # Gaps in claims/assertions are more critical than background
        role_scores = {
            'claim': 0.9,
            'evidence': 0.8,
            'conclusion': 0.85,
            'background': 0.4,
            'elaboration': 0.5,
            'contrast': 0.75,
        }

        return role_scores.get(discourse_structure.segment_role, 0.5)
```

### Algorithm 4: Comprehensive Necessity Score Computation

```python
class NecessityScoreComputer:
    """
    Computes the final necessity score using Bayesian combination of evidence.

    The necessity score represents:
    N(A, B) = P(Understand(A) | B present) - P(Understand(A) | B absent)
              ─────────────────────────────────────────────────────────────
                          P(Understand(A) | B present)

    This is the proportional comprehension loss when B is removed.
    """

    def __init__(self, config: NecessityCalculatorConfig):
        self.config = config

        # Weights for different evidence types
        self.evidence_weights = {
            'gap_coverage': 0.35,       # What % of gaps does B fill
            'severity_weighted': 0.25,  # Weighted by gap severity
            'type_multiplier': 0.15,    # Adjusted by necessity type
            'explicit_reference': 0.15, # Bonus for explicit citations
            'presupposition': 0.10,     # Presupposition satisfaction
        }

    async def compute_necessity_score(
        self,
        gaps: List[SemanticGap],
        gaps_filled: List[SemanticGap],
        presuppositions: List[Presupposition],
        presuppositions_satisfied: List[Presupposition],
        explicit_references: List[ExplicitReference],
        necessity_type: NecessityType
    ) -> Tuple[float, ScoreDistribution]:
        """
        Compute comprehensive necessity score with statistical distribution.

        Returns: (point_estimate, full_distribution)
        """

        # Component 1: Gap coverage
        if not gaps:
            gap_coverage = 0.0
        else:
            gap_coverage = len(gaps_filled) / len(gaps)

        # Component 2: Severity-weighted coverage
        if gaps:
            total_severity = sum(g.compute_final_severity() for g in gaps)
            filled_severity = sum(g.compute_final_severity() for g in gaps_filled)
            severity_weighted = filled_severity / total_severity if total_severity > 0 else 0.0
        else:
            severity_weighted = 0.0

        # Component 3: Type multiplier
        type_multiplier = self._get_type_multiplier(necessity_type)

        # Component 4: Explicit reference bonus
        explicit_bonus = 0.0
        if explicit_references:
            # Each explicit reference is strong evidence
            explicit_bonus = min(1.0, 0.3 * len(explicit_references))

        # Component 5: Presupposition satisfaction
        if presuppositions:
            pres_score = len(presuppositions_satisfied) / len(presuppositions)
        else:
            pres_score = 0.0

        # Weighted combination
        score = (
            self.evidence_weights['gap_coverage'] * gap_coverage +
            self.evidence_weights['severity_weighted'] * severity_weighted +
            self.evidence_weights['type_multiplier'] * type_multiplier +
            self.evidence_weights['explicit_reference'] * explicit_bonus +
            self.evidence_weights['presupposition'] * pres_score
        )

        # Compute statistical distribution
        # Using Beta distribution with evidence counts
        alpha = 1 + len(gaps_filled) + len(presuppositions_satisfied)
        beta_param = 1 + (len(gaps) - len(gaps_filled)) + (len(presuppositions) - len(presuppositions_satisfied))

        distribution = ScoreDistribution(
            mean=alpha / (alpha + beta_param),
            std=np.sqrt((alpha * beta_param) / ((alpha + beta_param)**2 * (alpha + beta_param + 1))),
            alpha=alpha,
            beta=beta_param,
            samples=len(gaps) + len(presuppositions)
        )

        # Final score with ceiling
        final_score = min(1.0, max(0.0, score))

        return final_score, distribution

    def _get_type_multiplier(self, necessity_type: NecessityType) -> float:
        """Get multiplier based on necessity type strength."""

        multipliers = {
            NecessityType.REFERENTIAL: 1.0,      # Explicit reference = highest
            NecessityType.PRESUPPOSITIONAL: 0.95,
            NecessityType.ARGUMENTATIVE: 0.92,
            NecessityType.COVENANTAL: 0.90,
            NecessityType.PROPHETIC: 0.88,
            NecessityType.DEFINITIONAL: 0.85,
            NecessityType.NARRATIVE: 0.82,
            NecessityType.LITURGICAL: 0.75,
            NecessityType.GENEALOGICAL: 0.70,
        }

        return multipliers.get(necessity_type, 0.75)

    def classify_strength(self, score: float) -> NecessityStrength:
        """Classify score into strength category."""

        if score >= self.config.absolute_threshold:
            return NecessityStrength.ABSOLUTE
        elif score >= self.config.strong_threshold:
            return NecessityStrength.STRONG
        elif score >= self.config.moderate_threshold:
            return NecessityStrength.MODERATE
        elif score >= self.config.weak_threshold:
            return NecessityStrength.WEAK
        else:
            return NecessityStrength.NONE
```

---

## Part 4: Main Calculator Implementation

### `InterVerseNecessityCalculator` Class

```python
class InterVerseNecessityCalculator:
    """
    The second Impossible Oracle: determines if verse B is NECESSARY to understand verse A.

    This engine goes beyond standard cross-reference discovery to identify
    LOGICAL DEPENDENCIES between verses - cases where verse A is semantically
    incomplete or incomprehensible without information from verse B.

    Architecture:
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                    InterVerseNecessityCalculator                          │
    │                                                                           │
    │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
    │  │ ReferenceExtractor │  │ PresupDetector │  │ SeverityCalc   │          │
    │  │                    │  │                │  │                │          │
    │  │ - Citation formulas│  │ - Existential  │  │ - Syntactic    │          │
    │  │ - Named entities   │  │ - Factive      │  │ - Info struct  │          │
    │  │ - Quotation detect │  │ - Lexical      │  │ - Discourse    │          │
    │  └────────┬───────────┘  └───────┬────────┘  └───────┬────────┘          │
    │           │                      │                   │                   │
    │           └──────────────────────┼───────────────────┘                   │
    │                                  ▼                                       │
    │                    ┌─────────────────────────┐                           │
    │                    │   GapAnalysisEngine     │                           │
    │                    │                         │                           │
    │                    │  - Identify all gaps    │                           │
    │                    │  - Score severities     │                           │
    │                    │  - Find gap fillers     │                           │
    │                    └───────────┬─────────────┘                           │
    │                                │                                         │
    │                                ▼                                         │
    │                    ┌─────────────────────────┐                           │
    │                    │ NecessityScoreComputer  │                           │
    │                    │                         │                           │
    │                    │  - Bayesian combination │                           │
    │                    │  - Statistical distrib  │                           │
    │                    │  - Strength classif     │                           │
    │                    └───────────┬─────────────┘                           │
    │                                │                                         │
    │                                ▼                                         │
    │                    ┌─────────────────────────┐                           │
    │                    │ NecessityAnalysisResult │                           │
    │                    └─────────────────────────┘                           │
    └──────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        omni_resolver: OmniContextualResolver,
        neo4j_client: Neo4jClient,
        redis_client: RedisClient,
        embedding_generator: EmbeddingGenerator,
        config: NecessityCalculatorConfig
    ):
        self.omni_resolver = omni_resolver
        self.neo4j = neo4j_client
        self.redis = redis_client
        self.embeddings = embedding_generator
        self.config = config

        # Initialize component engines
        self.reference_extractor = ReferenceExtractor()
        self.presupposition_detector = PresuppositionDetector()
        self.severity_calculator = GapSeverityCalculator()
        self.score_computer = NecessityScoreComputer(config)

        self.logger = logging.getLogger(__name__)

    async def calculate_necessity(
        self,
        verse_a: str,  # The dependent verse
        verse_b: str,  # The potential requirement
        force_recompute: bool = False
    ) -> NecessityAnalysisResult:
        """
        Main entry point: calculate necessity of B for understanding A.

        Process:
        1. Check cache for existing result
        2. Fetch verse texts and syntactic data
        3. Identify all semantic gaps in verse A
        4. Check if verse B fills any of those gaps
        5. Extract explicit references to B in A
        6. Detect presuppositions satisfied by B
        7. Compute weighted necessity score
        8. Classify and return result
        """
        start_time = asyncio.get_event_loop().time()

        # Step 1: Check cache
        if not force_recompute:
            cached = await self._get_cached_result(verse_a, verse_b)
            if cached:
                cached.cache_hit = True
                return cached

        # Step 2: Fetch verse data
        verse_a_data = await self._fetch_verse_data(verse_a)
        verse_b_data = await self._fetch_verse_data(verse_b)

        if not verse_a_data or not verse_b_data:
            raise ValueError(f"Could not fetch verse data for {verse_a} or {verse_b}")

        # Step 3: Identify semantic gaps in verse A
        gaps = await self.identify_semantic_gaps(
            verse_a_data.text,
            verse_a,
            verse_a_data.syntax_tree
        )

        # Step 4: Check if verse B fills gaps
        gaps_filled = await self._check_gaps_filled_by_verse(gaps, verse_b_data)

        # Step 5: Extract explicit references
        explicit_refs = await self.reference_extractor.extract_explicit_references(
            verse_a_data.text,
            verse_a,
            verse_a_data.language
        )
        # Filter to references to verse B
        refs_to_b = [r for r in explicit_refs if r.target_verse == verse_b]

        # Step 6: Detect and check presuppositions
        presuppositions = await self.presupposition_detector.detect_presuppositions(
            verse_a_data.text,
            verse_a_data.syntax_tree
        )
        pres_satisfied = await self._check_presuppositions_satisfied(
            presuppositions, verse_b_data
        )

        # Step 7: Determine necessity type
        necessity_type = self._determine_necessity_type(
            gaps_filled, refs_to_b, pres_satisfied
        )

        # Step 8: Compute necessity score
        score, distribution = await self.score_computer.compute_necessity_score(
            gaps=gaps,
            gaps_filled=gaps_filled,
            presuppositions=presuppositions,
            presuppositions_satisfied=pres_satisfied,
            explicit_references=refs_to_b,
            necessity_type=necessity_type
        )

        # Step 9: Classify strength
        strength = self.score_computer.classify_strength(score)

        # Step 10: Check bidirectionality (does B need A?)
        reverse_score = await self._quick_reverse_check(verse_b, verse_a)
        bidirectional = reverse_score > self.config.moderate_threshold

        # Step 11: Build result
        computation_time = (asyncio.get_event_loop().time() - start_time) * 1000

        result = NecessityAnalysisResult(
            analysis_id=self._generate_analysis_id(verse_a, verse_b),
            source_verse=verse_a,
            target_verse=verse_b,
            timestamp=datetime.utcnow().isoformat(),

            necessity_score=score,
            necessity_type=necessity_type,
            strength=strength,
            confidence=distribution.mean,

            score_distribution=distribution,
            confidence_interval=distribution.confidence_interval(0.95),

            semantic_gaps=gaps,
            gaps_filled_by_target=len(gaps_filled),
            total_gaps=len(gaps),
            gap_coverage=len(gaps_filled) / len(gaps) if gaps else 0.0,
            weighted_severity_filled=sum(g.compute_final_severity() for g in gaps_filled),

            presuppositions=presuppositions,
            presuppositions_satisfied=len(pres_satisfied),

            explicit_references=refs_to_b,
            has_citation_formula=any(
                r.formula_type == 'quotation' for r in refs_to_b
            ),

            dependency_chain=[verse_a, verse_b],
            chain_length=1,
            is_direct_dependency=True,

            bidirectional=bidirectional,
            reverse_necessity_score=reverse_score,
            mutual_necessity=bidirectional and score > self.config.strong_threshold,

            reasoning=self._generate_reasoning(
                gaps, gaps_filled, refs_to_b, pres_satisfied, score, strength
            ),
            evidence_summary=self._generate_evidence_summary(
                gaps_filled, refs_to_b, pres_satisfied
            ),

            computation_time_ms=computation_time,
            cache_hit=False,
            model_version="necessity_v2.0"
        )

        # Cache result
        await self._cache_result(result)

        return result

    async def identify_semantic_gaps(
        self,
        verse_text: str,
        verse_id: str,
        syntax_tree: Optional[SyntaxTree] = None
    ) -> List[SemanticGap]:
        """
        Identify all semantic gaps in verse text.

        A semantic gap is any element that requires external context to understand.

        Detection strategies:
        1. Named entities without introduction
        2. Definite NPs without antecedents
        3. Events referenced without narration
        4. Technical terms without definition
        5. Quotations without sources
        6. Presuppositions requiring prior knowledge
        """
        gaps = []

        # Strategy 1: Named entity analysis
        entity_gaps = await self._detect_entity_gaps(verse_text, verse_id)
        gaps.extend(entity_gaps)

        # Strategy 2: Definite NP analysis
        definite_gaps = await self._detect_definite_gaps(verse_text, syntax_tree)
        gaps.extend(definite_gaps)

        # Strategy 3: Event reference detection
        event_gaps = await self._detect_event_gaps(verse_text, verse_id)
        gaps.extend(event_gaps)

        # Strategy 4: Technical term detection
        term_gaps = await self._detect_term_gaps(verse_text, verse_id)
        gaps.extend(term_gaps)

        # Strategy 5: Quotation detection
        quote_gaps = await self._detect_quotation_gaps(verse_text, verse_id)
        gaps.extend(quote_gaps)

        # Calculate severities for all gaps
        for gap in gaps:
            gap.contextual_severity = await self.severity_calculator.calculate_severity(
                gap, verse_text, syntax_tree
            )

        return gaps

    async def _detect_entity_gaps(
        self,
        verse_text: str,
        verse_id: str
    ) -> List[SemanticGap]:
        """Detect references to entities not introduced in this verse."""
        gaps = []

        # Extract named entities
        entities = await self._extract_named_entities(verse_text)

        for entity in entities:
            # Check if this entity is introduced in this verse
            if not self._is_entity_introduced_here(entity, verse_text):
                # Find where this entity is introduced
                intro_verse = await self._find_entity_introduction(entity.text)

                gap = SemanticGap(
                    gap_id=f"gap_{verse_id}_{hash(entity.text)}",
                    gap_type=self._classify_entity_type(entity),
                    trigger_text=entity.text,
                    trigger_span=entity.span,
                    syntactic_role=entity.syntactic_role,
                    clause_type=entity.clause_type,
                    is_focus=entity.is_focus,
                    description=f"Reference to {entity.entity_type}: {entity.text}",
                    semantic_frame=entity.semantic_frame,
                    presupposition_type=PresuppositionType.EXISTENTIAL,
                    resolution_candidates=[
                        ResolutionCandidate(
                            verse_id=intro_verse,
                            resolution_score=0.9,
                            resolution_type="introduces",
                            evidence=[f"First canonical introduction of {entity.text}"],
                            confidence=0.85
                        )
                    ] if intro_verse else [],
                    best_resolution=intro_verse,
                    base_severity=0.85 if entity.entity_type == 'person' else 0.65
                )
                gaps.append(gap)

        return gaps

    async def build_dependency_graph(
        self,
        verse_ids: List[str],
        min_necessity: float = 0.3,
        max_pairs: int = 10000
    ) -> DependencyGraph:
        """
        Build a dependency graph for a set of verses.

        This creates a directed graph where edge (A, B) exists if
        verse A has necessity score >= min_necessity for verse B.

        Optimization: Uses batch processing and caching to avoid O(n²) computation.
        """
        self.logger.info(f"Building dependency graph for {len(verse_ids)} verses")

        # Step 1: Collect all existing necessity relationships from cache/DB
        existing_edges = await self._fetch_existing_necessities(verse_ids)

        # Step 2: Identify pairs that need computation
        all_pairs = set()
        for i, verse_a in enumerate(verse_ids):
            for verse_b in verse_ids[i+1:]:
                pair = (verse_a, verse_b)
                if pair not in existing_edges:
                    all_pairs.add(pair)
                    all_pairs.add((verse_b, verse_a))  # Check both directions

        # Limit pairs to avoid explosion
        pairs_to_compute = list(all_pairs)[:max_pairs]

        # Step 3: Batch compute necessities
        results = []
        for pair in pairs_to_compute:
            try:
                result = await self.calculate_necessity(pair[0], pair[1])
                if result.necessity_score >= min_necessity:
                    results.append(result)
            except Exception as e:
                self.logger.warning(f"Failed to compute necessity for {pair}: {e}")

        # Step 4: Combine with existing
        all_results = list(existing_edges.values()) + results

        # Step 5: Build graph
        graph = DependencyGraph.from_necessity_results(all_results, min_necessity)

        return graph

    def _determine_necessity_type(
        self,
        gaps_filled: List[SemanticGap],
        explicit_refs: List[ExplicitReference],
        pres_satisfied: List[Presupposition]
    ) -> NecessityType:
        """Determine the primary type of necessity relationship."""

        # Priority order for type determination

        # 1. Explicit quotation = REFERENTIAL
        if any(r.formula_type == 'quotation' for r in explicit_refs):
            return NecessityType.REFERENTIAL

        # 2. Gap types determine necessity type
        gap_types = [g.gap_type for g in gaps_filled]

        if GapType.QUOTATION_EXPLICIT in gap_types:
            return NecessityType.REFERENTIAL

        if GapType.EVENT_HISTORICAL in gap_types or GapType.ENTITY_PERSON in gap_types:
            return NecessityType.PRESUPPOSITIONAL

        if GapType.CONCEPT_COVENANTAL in gap_types:
            return NecessityType.COVENANTAL

        if GapType.CONCEPT_THEOLOGICAL in gap_types:
            return NecessityType.DEFINITIONAL

        if GapType.EVENT_RITUAL in gap_types:
            return NecessityType.LITURGICAL

        # 3. Presupposition types
        pres_types = [p.ptype for p in pres_satisfied]

        if PresuppositionType.FACTIVE in pres_types:
            return NecessityType.ARGUMENTATIVE

        # Default
        return NecessityType.PRESUPPOSITIONAL

    def _generate_reasoning(
        self,
        gaps: List[SemanticGap],
        gaps_filled: List[SemanticGap],
        refs: List[ExplicitReference],
        pres_satisfied: List[Presupposition],
        score: float,
        strength: NecessityStrength
    ) -> str:
        """Generate human-readable explanation of necessity relationship."""

        parts = []

        if refs:
            parts.append(
                f"Verse A explicitly cites verse B via {len(refs)} reference(s)."
            )

        if gaps_filled:
            gap_types = set(g.gap_type.value for g in gaps_filled)
            parts.append(
                f"Verse B fills {len(gaps_filled)}/{len(gaps)} semantic gaps "
                f"({', '.join(gap_types)})."
            )

        if pres_satisfied:
            parts.append(
                f"Verse B satisfies {len(pres_satisfied)} presupposition(s) in verse A."
            )

        parts.append(
            f"Necessity score: {score:.3f} ({strength.value})"
        )

        return " ".join(parts)
```

---

## Part 5: Integration Points

### Integration 1: OmniContextualResolver

```python
async def integrate_with_omni_resolver(
    self,
    verse_id: str
) -> List[NecessityRelation]:
    """
    Use OmniContextualResolver to find verses necessary for word disambiguation.

    When a word's meaning in verse A requires context from verse B to resolve,
    this creates a definitional necessity relationship.
    """
    # Get all polysemous words and their resolutions
    resolutions = await self.omni_resolver.resolve_verse_meanings(verse_id)

    necessity_relations = []

    for word_result in resolutions:
        if word_result.resolution_method == ResolutionMethod.INTERTEXTUAL:
            # This word required another verse for disambiguation
            for ctx_verse in word_result.required_context_verses:
                necessity_relations.append(NecessityRelation(
                    source_verse=verse_id,
                    target_verse=ctx_verse,
                    necessity_type=NecessityType.DEFINITIONAL,
                    evidence=f"Word '{word_result.word}' disambiguated by {ctx_verse}",
                    strength=word_result.confidence * 0.85
                ))

    return necessity_relations
```

### Integration 2: Neo4j SPIDERWEB Schema

```python
# Neo4j schema extension for necessity relationships

NECESSITY_SCHEMA = """
// Create constraint for NecessityRelationship
CREATE CONSTRAINT necessity_unique IF NOT EXISTS
FOR ()-[r:NECESSITATES]-()
REQUIRE (r.source_verse, r.target_verse) IS UNIQUE;

// Add indexes for necessity queries
CREATE INDEX necessity_score IF NOT EXISTS
FOR ()-[r:NECESSITATES]-()
ON (r.score);

CREATE INDEX necessity_type IF NOT EXISTS
FOR ()-[r:NECESSITATES]-()
ON (r.type);

// Example: Store necessity relationship
MATCH (a:Verse {id: $source_verse})
MATCH (b:Verse {id: $target_verse})
MERGE (a)-[r:NECESSITATES]->(b)
SET r.score = $necessity_score,
    r.type = $necessity_type,
    r.strength = $strength,
    r.gaps_filled = $gaps_filled,
    r.has_citation = $has_citation,
    r.bidirectional = $bidirectional,
    r.confidence = $confidence,
    r.computed_at = datetime();

// Query: Find all verses necessary to understand a target verse
MATCH path = (target:Verse {id: $verse_id})-[:NECESSITATES*1..5]->(required:Verse)
WHERE ALL(r IN relationships(path) WHERE r.score >= $min_score)
RETURN DISTINCT required.id AS required_verse,
       length(path) AS chain_length,
       [r IN relationships(path) | r.score] AS scores
ORDER BY chain_length;

// Query: Find foundational verses (most depended upon)
MATCH (v:Verse)<-[r:NECESSITATES]-(dependent:Verse)
WHERE r.score >= 0.7
WITH v, count(dependent) AS dependency_count, avg(r.score) AS avg_score
ORDER BY dependency_count DESC
LIMIT 100
RETURN v.id, dependency_count, avg_score;

// Query: Find mutual necessity clusters
MATCH (a:Verse)-[r1:NECESSITATES]->(b:Verse)-[r2:NECESSITATES]->(a)
WHERE r1.score >= 0.6 AND r2.score >= 0.6
RETURN a.id, b.id, r1.score AS a_needs_b, r2.score AS b_needs_a;
"""
```

### Integration 3: Pipeline Integration

```python
# In ml/inference/pipeline.py

class CrossReferencePipeline:
    """Extended to include necessity scoring."""

    async def _score_and_classify_candidates(
        self,
        source_verse: str,
        candidates: List[CrossReferenceCandidate]
    ) -> List[CrossReferenceCandidate]:
        """Score candidates with necessity analysis."""

        for candidate in candidates:
            if candidate.confidence >= self.config.necessity_threshold:
                # Calculate necessity for high-confidence candidates
                necessity_result = await self.necessity_calculator.calculate_necessity(
                    source_verse,
                    candidate.target_verse
                )

                # Enrich candidate with necessity data
                candidate.necessity_score = necessity_result.necessity_score
                candidate.necessity_type = necessity_result.necessity_type.value
                candidate.necessity_strength = necessity_result.strength.value
                candidate.is_essential = necessity_result.strength in [
                    NecessityStrength.ABSOLUTE,
                    NecessityStrength.STRONG
                ]

                # Boost confidence for essential connections
                if candidate.is_essential:
                    candidate.confidence = min(1.0, candidate.confidence * 1.15)

        return candidates
```

### Integration 4: Event Sourcing

```python
# Necessity events for event store

@dataclass
class NecessityDiscoveredEvent(DomainEvent):
    """Fired when a new necessity relationship is discovered."""
    event_type: str = "necessity.discovered"
    source_verse: str = ""
    target_verse: str = ""
    necessity_score: float = 0.0
    necessity_type: str = ""
    strength: str = ""
    gaps_filled: int = 0
    has_citation: bool = False
    confidence: float = 0.0

@dataclass
class NecessityUpdatedEvent(DomainEvent):
    """Fired when a necessity score is recalculated."""
    event_type: str = "necessity.updated"
    source_verse: str = ""
    target_verse: str = ""
    old_score: float = 0.0
    new_score: float = 0.0
    change_reason: str = ""

@dataclass
class DependencyGraphBuiltEvent(DomainEvent):
    """Fired when a dependency graph is constructed."""
    event_type: str = "dependency_graph.built"
    verse_count: int = 0
    edge_count: int = 0
    root_count: int = 0
    scc_count: int = 0  # Strongly connected components
```

---

## Part 6: Testing Specification

### Unit Tests: `tests/ml/engines/test_necessity_calculator.py`

```python
import pytest
from ml.engines.necessity_calculator import (
    InterVerseNecessityCalculator,
    NecessityType,
    NecessityStrength,
    SemanticGap,
    GapType
)

class TestNecessityCalculator:
    """Comprehensive tests for the InterVerseNecessityCalculator."""

    @pytest.fixture
    async def calculator(self, mock_services):
        """Create calculator with mock dependencies."""
        return InterVerseNecessityCalculator(
            omni_resolver=mock_services.omni_resolver,
            neo4j_client=mock_services.neo4j,
            redis_client=mock_services.redis,
            embedding_generator=mock_services.embeddings,
            config=NecessityCalculatorConfig()
        )

    # ==================== CANONICAL NECESSITY TESTS ====================

    @pytest.mark.asyncio
    async def test_hebrews_genesis_necessity_absolute(self, calculator):
        """
        HEB.11.17-19 → GEN.22.1-14 should be ABSOLUTE necessity.

        Hebrews passage is incomprehensible without Genesis Akedah:
        - "Abraham" - who is Abraham?
        - "tested" - what test?
        - "offered up Isaac" - what offering?
        - "seed promise" - what promise?
        - "resurrection reasoning" - why would he think this?
        """
        result = await calculator.calculate_necessity(
            verse_a="HEB.11.17",
            verse_b="GEN.22.1"
        )

        # Must be ABSOLUTE necessity
        assert result.necessity_score >= 0.90
        assert result.strength == NecessityStrength.ABSOLUTE

        # Must identify key gaps
        gap_types = {g.gap_type for g in result.semantic_gaps}
        assert GapType.ENTITY_PERSON in gap_types  # Abraham, Isaac
        assert GapType.EVENT_HISTORICAL in gap_types  # The testing

        # Should fill most gaps
        assert result.gap_coverage >= 0.8

        # Type should be presuppositional (no explicit citation)
        assert result.necessity_type in [
            NecessityType.PRESUPPOSITIONAL,
            NecessityType.NARRATIVE
        ]

    @pytest.mark.asyncio
    async def test_matthew_isaiah_quotation_strong(self, calculator):
        """
        MAT.1.23 → ISA.7.14 should be STRONG necessity.

        Matthew explicitly quotes Isaiah with citation formula:
        "All this took place to fulfill what the Lord had spoken
        by the prophet: 'Behold, the virgin shall conceive...'"
        """
        result = await calculator.calculate_necessity(
            verse_a="MAT.1.23",
            verse_b="ISA.7.14"
        )

        # Must be at least STRONG
        assert result.necessity_score >= 0.85
        assert result.strength in [NecessityStrength.ABSOLUTE, NecessityStrength.STRONG]

        # Must detect citation formula
        assert result.has_citation_formula == True
        assert len(result.explicit_references) >= 1

        # Type must be REFERENTIAL (explicit quotation)
        assert result.necessity_type == NecessityType.REFERENTIAL

    @pytest.mark.asyncio
    async def test_romans_leviticus_definitional(self, calculator):
        """
        ROM.3.25 → LEV.16 should be MODERATE-STRONG necessity.

        "propitiation" (ἱλαστήριον) requires sacrificial context:
        - The word literally means "mercy seat"
        - Full meaning requires Day of Atonement ritual
        """
        result = await calculator.calculate_necessity(
            verse_a="ROM.3.25",
            verse_b="LEV.16.15"
        )

        # At least MODERATE
        assert result.necessity_score >= 0.55
        assert result.strength in [
            NecessityStrength.STRONG,
            NecessityStrength.MODERATE
        ]

        # Should identify term gap
        term_gaps = [g for g in result.semantic_gaps if g.gap_type == GapType.TERM_TECHNICAL]
        assert len(term_gaps) >= 1

        # Type should be DEFINITIONAL
        assert result.necessity_type == NecessityType.DEFINITIONAL

    # ==================== NON-NECESSITY TESTS ====================

    @pytest.mark.asyncio
    async def test_no_necessity_thematic_only(self, calculator):
        """
        GEN.1.1 → PSA.19.1 should have WEAK/NONE necessity.

        Both discuss creation, but:
        - GEN.1.1 is complete without PSA.19.1
        - PSA.19.1 is complete without GEN.1.1
        - Thematic parallel, not logical dependency
        """
        result = await calculator.calculate_necessity(
            verse_a="GEN.1.1",
            verse_b="PSA.19.1"
        )

        # Must be WEAK or NONE
        assert result.necessity_score < 0.35
        assert result.strength in [NecessityStrength.WEAK, NecessityStrength.NONE]

        # Should identify as not essential
        assert result.is_direct_dependency == True  # Still direct check
        assert result.gap_coverage < 0.3  # Few gaps filled

    @pytest.mark.asyncio
    async def test_no_necessity_parallel_themes(self, calculator):
        """
        JHN.1.1 → GEN.1.1 - John echoes Genesis but doesn't require it.

        "In the beginning was the Word" is comprehensible standalone.
        Genesis enriches but doesn't complete the meaning.
        """
        result = await calculator.calculate_necessity(
            verse_a="JHN.1.1",
            verse_b="GEN.1.1"
        )

        # Should be at most MODERATE (literary echo, not dependency)
        assert result.necessity_score < 0.60

    # ==================== BIDIRECTIONAL TESTS ====================

    @pytest.mark.asyncio
    async def test_asymmetric_necessity(self, calculator):
        """
        HEB.11.17 needs GEN.22 (high), but GEN.22 doesn't need HEB.11.17 (low).

        The Old Testament is foundational; New Testament depends on it.
        """
        forward = await calculator.calculate_necessity("HEB.11.17", "GEN.22.1")
        reverse = await calculator.calculate_necessity("GEN.22.1", "HEB.11.17")

        # Forward should be high
        assert forward.necessity_score >= 0.8

        # Reverse should be low
        assert reverse.necessity_score < 0.3

        # Should not be mutual necessity
        assert forward.mutual_necessity == False

    @pytest.mark.asyncio
    async def test_mutual_necessity_rare(self, calculator):
        """
        True mutual necessity is rare - find if exists.

        Most relationships are directional (NT → OT).
        """
        # Example of potential mutual necessity: Chronicles/Kings parallel accounts
        # Both tell the same story, each has unique details
        forward = await calculator.calculate_necessity("2SAM.11.2", "1CHR.20.1")
        reverse = await calculator.calculate_necessity("1CHR.20.1", "2SAM.11.2")

        # At least one direction should have some necessity
        # (parallel accounts reference each other)
        max_score = max(forward.necessity_score, reverse.necessity_score)
        assert max_score >= 0.3

    # ==================== GAP DETECTION TESTS ====================

    @pytest.mark.asyncio
    async def test_gap_detection_entities(self, calculator):
        """Test detection of entity gaps."""
        gaps = await calculator.identify_semantic_gaps(
            verse_text="By faith Abraham offered up Isaac his son",
            verse_id="HEB.11.17"
        )

        # Should find Abraham and Isaac as entity gaps
        entity_gaps = [g for g in gaps if g.gap_type == GapType.ENTITY_PERSON]
        entity_names = {g.trigger_text.lower() for g in entity_gaps}

        assert 'abraham' in entity_names or any('abraham' in n for n in entity_names)
        assert 'isaac' in entity_names or any('isaac' in n for n in entity_names)

    @pytest.mark.asyncio
    async def test_gap_detection_events(self, calculator):
        """Test detection of event gaps."""
        gaps = await calculator.identify_semantic_gaps(
            verse_text="when he was tested",
            verse_id="HEB.11.17"
        )

        # Should find "tested" as an event gap
        event_gaps = [g for g in gaps if g.gap_type == GapType.EVENT_HISTORICAL]
        assert len(event_gaps) >= 1

    @pytest.mark.asyncio
    async def test_gap_severity_main_clause(self, calculator):
        """Gaps in main clause should have higher severity."""
        gaps = await calculator.identify_semantic_gaps(
            verse_text="Abraham offered Isaac",  # Main clause
            verse_id="test"
        )

        for gap in gaps:
            # Main clause gaps should be high severity
            assert gap.compute_final_severity() >= 0.6

    # ==================== DEPENDENCY GRAPH TESTS ====================

    @pytest.mark.asyncio
    async def test_dependency_graph_construction(self, calculator):
        """Test building dependency graph for Hebrews 11."""
        # Hebrews 11 "faith heroes" chapter
        verses = [f"HEB.11.{i}" for i in range(1, 40)]

        graph = await calculator.build_dependency_graph(
            verse_ids=verses,
            min_necessity=0.5
        )

        # Should have edges to OT source passages
        assert graph.total_edges > 0

        # Root verses should be OT foundations
        # (Hebrews depends on OT, not vice versa)
        # Note: roots in this graph = verses nothing depends on
        assert len(graph.root_verses) > 0

    @pytest.mark.asyncio
    async def test_dependency_chain_finding(self, calculator):
        """Test finding necessity chains."""
        graph = await calculator.build_dependency_graph(
            verse_ids=["HEB.11.17", "GEN.22.1", "GEN.12.1"],
            min_necessity=0.3
        )

        chain = graph.find_necessity_chain("HEB.11.17", "GEN.12.1")

        # Should find path through Genesis
        assert len(chain) >= 2  # At least HEB → GEN

    # ==================== PERFORMANCE TESTS ====================

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_single_calculation_performance(self, calculator, benchmark):
        """Single necessity calculation should be fast."""
        result = benchmark(
            lambda: asyncio.run(
                calculator.calculate_necessity("HEB.11.17", "GEN.22.1")
            )
        )

        # Should complete in < 500ms (cached)
        assert result.computation_time_ms < 500

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_batch_processing_performance(self, calculator):
        """Batch processing should achieve good throughput."""
        pairs = [
            ("MAT.1.23", "ISA.7.14"),
            ("ROM.3.25", "LEV.16.15"),
            ("HEB.11.17", "GEN.22.1"),
            ("GAL.3.6", "GEN.15.6"),
            ("1PE.2.6", "ISA.28.16"),
        ]

        start = asyncio.get_event_loop().time()

        results = await asyncio.gather(*[
            calculator.calculate_necessity(a, b) for a, b in pairs
        ])

        elapsed = asyncio.get_event_loop().time() - start

        # Should process at least 10 pairs/second
        rate = len(pairs) / elapsed
        assert rate >= 10.0
```

---

## Part 7: Configuration

### Add to `config.py`

```python
@dataclass
class NecessityCalculatorConfig:
    """Configuration for InterVerseNecessityCalculator."""

    # Strength thresholds
    absolute_threshold: float = 0.90   # Score for ABSOLUTE classification
    strong_threshold: float = 0.70     # Score for STRONG classification
    moderate_threshold: float = 0.50   # Score for MODERATE classification
    weak_threshold: float = 0.30       # Score for WEAK classification

    # Score component weights
    gap_coverage_weight: float = 0.35     # Weight of gap coverage
    severity_weight: float = 0.25         # Weight of severity-weighted coverage
    type_weight: float = 0.15             # Weight of necessity type
    explicit_ref_weight: float = 0.15     # Weight of explicit references
    presupposition_weight: float = 0.10   # Weight of presupposition satisfaction

    # Detection settings
    min_entity_confidence: float = 0.7    # Min confidence for entity detection
    min_quotation_overlap: float = 0.5    # Min verbal overlap for quotation
    min_presupposition_conf: float = 0.6  # Min confidence for presupposition

    # Graph settings
    max_chain_depth: int = 10             # Maximum necessity chain length
    max_graph_nodes: int = 1000           # Max nodes in dependency graph
    graph_min_score: float = 0.3          # Min score for graph edges

    # Caching
    cache_enabled: bool = True            # Enable result caching
    cache_ttl_seconds: int = 604800       # Cache TTL (1 week)

    # Performance
    parallel_gap_detection: bool = True   # Parallelize gap detection
    batch_size: int = 50                  # Batch size for bulk operations
    max_concurrent: int = 10              # Max concurrent calculations

    # Integration
    enrich_cross_references: bool = True  # Add necessity to cross-refs
    store_to_neo4j: bool = True           # Store results in Neo4j
    emit_events: bool = True              # Emit event sourcing events
```

---

## Part 8: Caching Strategy

### Multi-Level Caching Architecture

```python
class NecessityCacheManager:
    """
    Multi-level caching for necessity calculations.

    Level 1: In-memory LRU cache (fastest, smallest)
    Level 2: Redis cache (fast, medium size)
    Level 3: Neo4j persistent storage (slowest, permanent)
    """

    def __init__(self, redis: RedisClient, neo4j: Neo4jClient, config: NecessityCalculatorConfig):
        self.redis = redis
        self.neo4j = neo4j
        self.config = config

        # Level 1: In-memory LRU cache
        self.memory_cache: Dict[str, NecessityAnalysisResult] = {}
        self.memory_cache_max = 1000
        self.memory_cache_order: List[str] = []

    async def get(self, verse_a: str, verse_b: str) -> Optional[NecessityAnalysisResult]:
        """Get cached result using cache hierarchy."""
        key = self._make_key(verse_a, verse_b)

        # Level 1: Memory
        if key in self.memory_cache:
            self._touch_memory_cache(key)
            return self.memory_cache[key]

        # Level 2: Redis
        redis_result = await self._get_from_redis(key)
        if redis_result:
            self._add_to_memory_cache(key, redis_result)
            return redis_result

        # Level 3: Neo4j
        neo4j_result = await self._get_from_neo4j(verse_a, verse_b)
        if neo4j_result:
            await self._add_to_redis(key, neo4j_result)
            self._add_to_memory_cache(key, neo4j_result)
            return neo4j_result

        return None

    async def put(self, result: NecessityAnalysisResult):
        """Store result in all cache levels."""
        key = self._make_key(result.source_verse, result.target_verse)

        # Level 1: Memory
        self._add_to_memory_cache(key, result)

        # Level 2: Redis (async)
        asyncio.create_task(self._add_to_redis(key, result))

        # Level 3: Neo4j (async)
        asyncio.create_task(self._add_to_neo4j(result))

    def _make_key(self, verse_a: str, verse_b: str) -> str:
        """Create cache key."""
        return f"necessity:{verse_a}:{verse_b}"

    async def _get_from_redis(self, key: str) -> Optional[NecessityAnalysisResult]:
        """Get from Redis cache."""
        data = await self.redis.get(f"biblos:{key}")
        if data:
            return NecessityAnalysisResult.from_json(data)
        return None

    async def _add_to_redis(self, key: str, result: NecessityAnalysisResult):
        """Add to Redis cache with TTL."""
        await self.redis.setex(
            f"biblos:{key}",
            self.config.cache_ttl_seconds,
            result.to_json()
        )

    async def _get_from_neo4j(
        self,
        verse_a: str,
        verse_b: str
    ) -> Optional[NecessityAnalysisResult]:
        """Get from Neo4j persistent storage."""
        query = """
        MATCH (a:Verse {id: $verse_a})-[r:NECESSITATES]->(b:Verse {id: $verse_b})
        RETURN r.score AS score,
               r.type AS type,
               r.strength AS strength,
               r.confidence AS confidence,
               r.gaps_filled AS gaps_filled,
               r.has_citation AS has_citation,
               r.computed_at AS computed_at
        """
        result = await self.neo4j.query_single(query, {
            "verse_a": verse_a,
            "verse_b": verse_b
        })

        if result:
            return self._neo4j_record_to_result(verse_a, verse_b, result)
        return None

    async def _add_to_neo4j(self, result: NecessityAnalysisResult):
        """Persist to Neo4j."""
        query = """
        MATCH (a:Verse {id: $source_verse})
        MATCH (b:Verse {id: $target_verse})
        MERGE (a)-[r:NECESSITATES]->(b)
        SET r += $properties
        """
        await self.neo4j.execute(query, {
            "source_verse": result.source_verse,
            "target_verse": result.target_verse,
            "properties": result.to_neo4j_properties()
        })
```

---

## Part 9: Success Criteria

### Functional Requirements
- [ ] Correctly identifies explicit quotations via citation formula detection
- [ ] Detects presuppositional relationships using formal semantics
- [ ] Calculates gap severity with syntactic and information structure weighting
- [ ] Produces statistically valid necessity scores with confidence intervals
- [ ] Builds correct dependency graphs with SCC detection
- [ ] Distinguishes necessary from merely helpful connections (>95% accuracy)
- [ ] Integrates with OmniContextualResolver for word-level necessity
- [ ] Persists results to Neo4j with proper NECESSITATES relationships

### Theological Accuracy
- [ ] HEB.11.17-19 → GEN.22: ABSOLUTE necessity (score ≥ 0.90)
- [ ] MAT.1.23 → ISA.7.14: STRONG necessity with citation detection
- [ ] ROM.3.25 → LEV.16: MODERATE-STRONG definitional necessity
- [ ] GEN.1.1 ↔ PSA.19.1: WEAK/NONE (thematic, not logical)
- [ ] Asymmetric: NT → OT high, OT → NT low (generally)

### Performance Requirements
- [ ] Single pair calculation: < 200ms (cached), < 2s (cold)
- [ ] Batch processing: > 50 pairs/second
- [ ] Graph construction (100 verses): < 30 seconds
- [ ] Memory usage: < 500MB for dependency graph of 1000 verses

### Statistical Requirements
- [ ] Confidence intervals properly computed using Beta distribution
- [ ] Score distributions calibrated (predicted confidence matches actual)
- [ ] Inter-annotator agreement with human judgments: κ > 0.75

---

## Part 10: Implementation Order

1. **Core enums and dataclasses** - `NecessityType`, `NecessityStrength`, `SemanticGap`, `GapType`
2. **ReferenceExtractor** - Citation formula detection, quotation matching
3. **PresuppositionDetector** - Linguistic presupposition identification
4. **GapSeverityCalculator** - Multi-factor severity scoring
5. **NecessityScoreComputer** - Bayesian score combination
6. **InterVerseNecessityCalculator** - Main orchestration class
7. **DependencyGraph** - NetworkX-based graph operations
8. **NecessityCacheManager** - Multi-level caching
9. **Neo4j integration** - NECESSITATES relationship schema
10. **Pipeline integration** - Enrich cross-references with necessity
11. **Event sourcing** - NecessityDiscovered events
12. **Configuration** - `NecessityCalculatorConfig`
13. **Unit tests** - Canonical test cases
14. **Theological validation** - Human expert review

---

## Part 11: Dependencies on Other Sessions

### Depends On
- **SESSION 01**: Mutual Transformation Metric (semantic shift analysis)
- **SESSION 03**: Omni-Contextual Resolver (word meaning resolution)
- **SESSION 09**: Neo4j Graph Architecture (SPIDERWEB schema)
- **SESSION 10**: Vector DB (embedding similarity for quotation matching)

### Depended On By
- **SESSION 06**: Fractal Typology Engine (uses necessity for layer connections)
- **SESSION 07**: Prophetic Necessity Prover (builds on necessity calculations)
- **SESSION 11**: Pipeline Integration (orchestrates necessity scoring)
- **SESSION 12**: Testing (validates necessity accuracy)

### External Dependencies
- NetworkX for graph algorithms
- scipy.stats for Beta distribution
- Text-Fabric/Macula for syntactic analysis
- Verse database with full text and morphology

---

## Session Completion Checklist

```markdown
- [ ] `ml/engines/necessity_calculator.py` implemented
- [ ] All enums and dataclasses defined with comprehensive documentation
- [ ] ReferenceExtractor detects citation formulas in Greek/Hebrew/English
- [ ] PresuppositionDetector identifies formal linguistic presuppositions
- [ ] GapSeverityCalculator uses syntactic prominence and information structure
- [ ] NecessityScoreComputer produces Bayesian scores with confidence intervals
- [ ] DependencyGraph supports SCC detection and PageRank
- [ ] Multi-level caching (memory → Redis → Neo4j)
- [ ] Configuration added to config.py
- [ ] HEB.11.17 → GEN.22 test passing (ABSOLUTE, score ≥ 0.90)
- [ ] MAT.1.23 → ISA.7.14 test passing (STRONG, citation detected)
- [ ] GEN.1.1 ↔ PSA.19.1 test passing (WEAK/NONE)
- [ ] Performance tests passing (< 200ms cached)
- [ ] Neo4j NECESSITATES relationships stored correctly
- [ ] Event sourcing integration complete
- [ ] Documentation complete
```

**Next Session**: SESSION 05: LXX Christological Extractor
