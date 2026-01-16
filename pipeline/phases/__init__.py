"""
Pipeline Phases

Five-phase processing pipeline for BIBLOS v2.
"""
from pipeline.phases.base import Phase, PhasePriority, PhaseCategory, PhaseDependency

# Phase implementations will be imported when ready
# from pipeline.phases.linguistic import LinguisticPhase
# from pipeline.phases.theological import TheologicalPhase
# from pipeline.phases.intertextual import IntertextualPhase
# from pipeline.phases.cross_reference import CrossReferencePhase
# from pipeline.phases.validation import ValidationPhase

__all__ = [
    "Phase",
    "PhasePriority",
    "PhaseCategory",
    "PhaseDependency",
]
