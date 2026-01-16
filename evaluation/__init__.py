"""
BIBLOS v2 - Agent Evaluation Framework

Comprehensive evaluation system for SDES extraction agents.
"""
from evaluation.framework import (
    AgentEvaluator,
    EvaluationConfig,
    EvaluationResult,
    EvaluationMetrics,
)
from evaluation.metrics import (
    compute_precision,
    compute_recall,
    compute_f1,
    compute_cross_reference_accuracy,
)
from evaluation.test_cases import (
    TestCase,
    TestSuite,
    load_test_suite,
)

__all__ = [
    "AgentEvaluator",
    "EvaluationConfig",
    "EvaluationResult",
    "EvaluationMetrics",
    "compute_precision",
    "compute_recall",
    "compute_f1",
    "compute_cross_reference_accuracy",
    "TestCase",
    "TestSuite",
    "load_test_suite",
]
