"""
BIBLOS v2 - ML Engines Module

"Five Impossible Oracles" - Superhuman analysis capabilities for biblical scholarship.

This module provides oracle engines that perform analysis beyond human cognitive
limits by leveraging computational access to the complete biblical canon.
"""

from ml.engines.omnicontext_resolver import (
    # Enums
    EliminationReason,
    # Dataclasses
    EliminationStep,
    SemanticFieldEntry,
    CompatibilityResult,
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
    TransformationType,
    PatternType,
    # Dataclasses
    FractalTypologyConfig,
    TypologyConnection,
    FractalTypologyResult,
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
    # OmniContextualResolver exports
    "EliminationReason",
    "EliminationStep",
    "SemanticFieldEntry",
    "CompatibilityResult",
    "AbsoluteMeaningResult",
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
    "TransformationType",
    "PatternType",
    # FractalTypologyEngine dataclasses
    "FractalTypologyConfig",
    "TypologyConnection",
    "FractalTypologyResult",
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
