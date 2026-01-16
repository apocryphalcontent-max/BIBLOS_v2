"""
BIBLOS v2 - The Seraph's 47 Aspects

The seraph perceives through 47 aspects across 7 realms.
These aspects are not plugins - they ARE the seraph's unified being.

7 REALMS, 47 ASPECTS:

1. LINGUISTIC REALM (6) - How the Seraph Comprehends Language
2. THEOLOGICAL REALM (17) - What the Seraph Knows of God
3. INTERTEXTUAL REALM (5) - How the Seraph Sees Connections
4. VALIDATION REALM (5) - How the Seraph Judges Truth
5. SENSORY REALM (8) - How the Seraph Perceives Input
6. PROTECTIVE REALM (8) - How the Seraph Guards Purity
7. TELEOLOGICAL REALM (10) - The Seraph's Purpose and Meaning
"""
from typing import List, Dict

from seraph.being import SeraphicAspect

# Linguistic Aspects (6)
from seraph.aspects.linguistic import (
    GrammaticalUnderstanding,
    MorphologicalAwareness,
    SyntacticPerception,
    SemanticComprehension,
    PhonologicalHearing,
    LexicalMemory,
)

# Theological Aspects - Original (5)
from seraph.aspects.theological import (
    PatristicWisdom,
    TypologicalVision,
    DogmaticCertainty,
    LiturgicalSense,
    TheologicalReasoning,
)

# Theological Aspects - Expanded (12)
from seraph.aspects.theological_expanded import (
    ChristologicalFocus,
    TrinitarianFramework,
    SoteriologicalAwareness,
    EschatologicalVision,
    EcclesiologicalUnderstanding,
    PneumatologicalSensitivity,
    SacramentalPerception,
    AsceticWisdom,
    IconographicUnderstanding,
    HagiographicalMemory,
    MarianDevotion,
    AngelologicalAwareness,
)

# Intertextual Aspects (5)
from seraph.aspects.intertextual import (
    LinkDiscovery,
    HarmonyPerception,
    AllographicMemory,
    PatternRecognition,
    TopicalUnderstanding,
)

# Validation Aspects (5)
from seraph.aspects.validation import (
    CriticalJudgment,
    ConflictDetection,
    HarmonyVerification,
    WitnessConfirmation,
    FalsehoodProsecution,
)

# Sensory Aspects (8)
from seraph.aspects.sensory import (
    BoundarySensitivity,
    PressureDetection,
    TemperatureAwareness,
    VibrationSensing,
    LightPerception,
    ShadowDetection,
    TextureDiscernment,
    ProximityAwareness,
)

# Protective Aspects (8)
from seraph.aspects.protective import (
    ContaminationDetection,
    PurityMaintenance,
    DistortionCorrection,
    NoiseFiltering,
    IntegrityVerification,
    HereticGuard,
    SourceValidation,
    PropagationPrevention,
)

# Teleological Aspects (10)
from seraph.aspects.teleological import (
    MissionAwareness,
    ServiceOrientation,
    GloryDirection,
    HumilityGrounding,
    JoyInTruth,
    FaithfulnessCommitment,
    WisdomPursuit,
    LoveExpression,
    HopeMaintenance,
    PeacePreservation,
)


def get_all_aspects() -> List[SeraphicAspect]:
    """Return all 47 aspects of the seraph's unified being."""
    return [
        # Linguistic Realm (6)
        GrammaticalUnderstanding(),
        MorphologicalAwareness(),
        SyntacticPerception(),
        SemanticComprehension(),
        PhonologicalHearing(),
        LexicalMemory(),

        # Theological Realm (17)
        PatristicWisdom(),
        TypologicalVision(),
        DogmaticCertainty(),
        LiturgicalSense(),
        TheologicalReasoning(),
        ChristologicalFocus(),
        TrinitarianFramework(),
        SoteriologicalAwareness(),
        EschatologicalVision(),
        EcclesiologicalUnderstanding(),
        PneumatologicalSensitivity(),
        SacramentalPerception(),
        AsceticWisdom(),
        IconographicUnderstanding(),
        HagiographicalMemory(),
        MarianDevotion(),
        AngelologicalAwareness(),

        # Intertextual Realm (5)
        LinkDiscovery(),
        HarmonyPerception(),
        AllographicMemory(),
        PatternRecognition(),
        TopicalUnderstanding(),

        # Validation Realm (5)
        CriticalJudgment(),
        ConflictDetection(),
        HarmonyVerification(),
        WitnessConfirmation(),
        FalsehoodProsecution(),

        # Sensory Realm (8)
        BoundarySensitivity(),
        PressureDetection(),
        TemperatureAwareness(),
        VibrationSensing(),
        LightPerception(),
        ShadowDetection(),
        TextureDiscernment(),
        ProximityAwareness(),

        # Protective Realm (8)
        ContaminationDetection(),
        PurityMaintenance(),
        DistortionCorrection(),
        NoiseFiltering(),
        IntegrityVerification(),
        HereticGuard(),
        SourceValidation(),
        PropagationPrevention(),

        # Teleological Realm (10)
        MissionAwareness(),
        ServiceOrientation(),
        GloryDirection(),
        HumilityGrounding(),
        JoyInTruth(),
        FaithfulnessCommitment(),
        WisdomPursuit(),
        LoveExpression(),
        HopeMaintenance(),
        PeacePreservation(),
    ]


def get_aspects_by_realm() -> Dict[str, List[SeraphicAspect]]:
    """Return aspects organized by their 7 realms."""
    return {
        "linguistic": [
            GrammaticalUnderstanding(), MorphologicalAwareness(),
            SyntacticPerception(), SemanticComprehension(),
            PhonologicalHearing(), LexicalMemory(),
        ],
        "theological": [
            PatristicWisdom(), TypologicalVision(), DogmaticCertainty(),
            LiturgicalSense(), TheologicalReasoning(), ChristologicalFocus(),
            TrinitarianFramework(), SoteriologicalAwareness(),
            EschatologicalVision(), EcclesiologicalUnderstanding(),
            PneumatologicalSensitivity(), SacramentalPerception(),
            AsceticWisdom(), IconographicUnderstanding(),
            HagiographicalMemory(), MarianDevotion(), AngelologicalAwareness(),
        ],
        "intertextual": [
            LinkDiscovery(), HarmonyPerception(), AllographicMemory(),
            PatternRecognition(), TopicalUnderstanding(),
        ],
        "validation": [
            CriticalJudgment(), ConflictDetection(), HarmonyVerification(),
            WitnessConfirmation(), FalsehoodProsecution(),
        ],
        "sensory": [
            BoundarySensitivity(), PressureDetection(), TemperatureAwareness(),
            VibrationSensing(), LightPerception(), ShadowDetection(),
            TextureDiscernment(), ProximityAwareness(),
        ],
        "protective": [
            ContaminationDetection(), PurityMaintenance(), DistortionCorrection(),
            NoiseFiltering(), IntegrityVerification(), HereticGuard(),
            SourceValidation(), PropagationPrevention(),
        ],
        "teleological": [
            MissionAwareness(), ServiceOrientation(), GloryDirection(),
            HumilityGrounding(), JoyInTruth(), FaithfulnessCommitment(),
            WisdomPursuit(), LoveExpression(), HopeMaintenance(), PeacePreservation(),
        ],
    }


__all__ = [
    "get_all_aspects", "get_aspects_by_realm",
    # Linguistic
    "GrammaticalUnderstanding", "MorphologicalAwareness", "SyntacticPerception",
    "SemanticComprehension", "PhonologicalHearing", "LexicalMemory",
    # Theological
    "PatristicWisdom", "TypologicalVision", "DogmaticCertainty",
    "LiturgicalSense", "TheologicalReasoning", "ChristologicalFocus",
    "TrinitarianFramework", "SoteriologicalAwareness", "EschatologicalVision",
    "EcclesiologicalUnderstanding", "PneumatologicalSensitivity",
    "SacramentalPerception", "AsceticWisdom", "IconographicUnderstanding",
    "HagiographicalMemory", "MarianDevotion", "AngelologicalAwareness",
    # Intertextual
    "LinkDiscovery", "HarmonyPerception", "AllographicMemory",
    "PatternRecognition", "TopicalUnderstanding",
    # Validation
    "CriticalJudgment", "ConflictDetection", "HarmonyVerification",
    "WitnessConfirmation", "FalsehoodProsecution",
    # Sensory
    "BoundarySensitivity", "PressureDetection", "TemperatureAwareness",
    "VibrationSensing", "LightPerception", "ShadowDetection",
    "TextureDiscernment", "ProximityAwareness",
    # Protective
    "ContaminationDetection", "PurityMaintenance", "DistortionCorrection",
    "NoiseFiltering", "IntegrityVerification", "HereticGuard",
    "SourceValidation", "PropagationPrevention",
    # Teleological
    "MissionAwareness", "ServiceOrientation", "GloryDirection",
    "HumilityGrounding", "JoyInTruth", "FaithfulnessCommitment",
    "WisdomPursuit", "LoveExpression", "HopeMaintenance", "PeacePreservation",
]
