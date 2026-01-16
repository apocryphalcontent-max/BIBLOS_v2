"""
BIBLOS v2 - ML Engines Module

"Five Impossible Oracles" - Superhuman analysis capabilities for biblical scholarship.

This module provides oracle engines that perform analysis beyond human cognitive
limits by leveraging computational access to the complete biblical canon.
"""

from ml.engines.omnicontext_resolver import (
    # Enums
    EliminationReason,
    SemanticDomain,
    GrammaticalCategory,
    VerbSemantic,
    PatristicEra,
    ConciliarAuthority,
    LXXDivergenceType,
    ConfidenceLevel,
    # Dataclasses
    EliminationStep,
    SemanticFieldEntry,
    CompatibilityResult,
    LXXMTDivergence,
    PatristicWitness as OmniPatristicWitness,  # Alias to avoid conflict with lxx_extractor
    ConciliarDefinition,
    OccurrenceData,
    AbsoluteMeaningResult,
    # Main class
    OmniContextualResolver,
)

from ml.engines.necessity_calculator import (
    # Enums
    NecessityType,
    NecessityStrength,
    GapType,
    PresuppositionType,
    SyntacticRole,
    ClauseType,
    # Dataclasses
    SemanticGap,
    Presupposition,
    ExplicitReference,
    ResolutionCandidate,
    ScoreDistribution,
    NecessityAnalysisResult,
    VerseData,
    # Component classes
    ReferenceExtractor,
    PresuppositionDetector,
    GapSeverityCalculator,
    NecessityScoreComputer,
    DependencyGraph,
    # Main class
    InterVerseNecessityCalculator,
)

from ml.engines.lxx_extractor import (
    # Enums
    ManuscriptPriority,
    DivergenceType,
    ChristologicalCategory,
    # Dataclasses
    ManuscriptWitness,
    NTQuotation,
    PatristicWitness,
    LXXDivergence,
    LXXAnalysisResult,
    # Main class
    LXXChristologicalExtractor,
)

from ml.engines.fractal_typology import (
    # Enums
    TypologyLayer,
    TypeAntitypeRelation,
    CorrespondenceType,
    PatristicConfidence,
    CovenantPhase,
    # Dataclasses
    TypePattern,
    LayerConnection,
    CovenantArc,
    SelfSimilarityAnalysis,
    FractalTypologyResult,
    FractalTypologyConfig,
    # Main class
    HyperFractalTypologyEngine,
)

from ml.engines.prophetic_prover import (
    # Enums
    FulfillmentType,
    IndependenceLevel,
    SpecificityFactor,
    EvidenceStrength,
    # Dataclasses
    ProbabilityEstimation,
    ProphecyFulfillmentPair,
    IndependenceAnalysis,
    BayesianResult,
    PropheticProofResult,
    PropheticProverConfig,
    # Main class
    PropheticNecessityProver,
)

__all__ = [
    # OmniContextualResolver exports - Enums
    "EliminationReason",
    "SemanticDomain",
    "GrammaticalCategory",
    "VerbSemantic",
    "PatristicEra",
    "ConciliarAuthority",
    "LXXDivergenceType",
    "ConfidenceLevel",
    # OmniContextualResolver exports - Dataclasses
    "EliminationStep",
    "SemanticFieldEntry",
    "CompatibilityResult",
    "LXXMTDivergence",
    "OmniPatristicWitness",
    "ConciliarDefinition",
    "OccurrenceData",
    "AbsoluteMeaningResult",
    # OmniContextualResolver exports - Main class
    "OmniContextualResolver",
    # NecessityCalculator enums
    "NecessityType",
    "NecessityStrength",
    "GapType",
    "PresuppositionType",
    "SyntacticRole",
    "ClauseType",
    # NecessityCalculator dataclasses
    "SemanticGap",
    "Presupposition",
    "ExplicitReference",
    "ResolutionCandidate",
    "ScoreDistribution",
    "NecessityAnalysisResult",
    "VerseData",
    # NecessityCalculator component classes
    "ReferenceExtractor",
    "PresuppositionDetector",
    "GapSeverityCalculator",
    "NecessityScoreComputer",
    "DependencyGraph",
    # NecessityCalculator main class
    "InterVerseNecessityCalculator",
    # LXXChristologicalExtractor enums
    "ManuscriptPriority",
    "DivergenceType",
    "ChristologicalCategory",
    # LXXChristologicalExtractor dataclasses
    "ManuscriptWitness",
    "NTQuotation",
    "PatristicWitness",
    "LXXDivergence",
    "LXXAnalysisResult",
    # LXXChristologicalExtractor main class
    "LXXChristologicalExtractor",
    # FractalTypologyEngine enums
    "TypologyLayer",
    "TypeAntitypeRelation",
    "CorrespondenceType",
    "PatristicConfidence",
    "CovenantPhase",
    # FractalTypologyEngine dataclasses
    "TypePattern",
    "LayerConnection",
    "CovenantArc",
    "SelfSimilarityAnalysis",
    "FractalTypologyResult",
    "FractalTypologyConfig",
    # FractalTypologyEngine main class
    "HyperFractalTypologyEngine",
    # PropheticNecessityProver enums
    "FulfillmentType",
    "IndependenceLevel",
    "SpecificityFactor",
    "EvidenceStrength",
    # PropheticNecessityProver dataclasses
    "ProbabilityEstimation",
    "ProphecyFulfillmentPair",
    "IndependenceAnalysis",
    "BayesianResult",
    "PropheticProofResult",
    "PropheticProverConfig",
    # PropheticNecessityProver main class
    "PropheticNecessityProver",
]
