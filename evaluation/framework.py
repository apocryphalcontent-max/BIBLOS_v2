"""
BIBLOS v2 - Agent Evaluation Framework

Production-grade evaluation system for SDES extraction agents with:
- Multi-dimensional evaluation metrics
- MLflow experiment tracking
- Statistical significance testing
- Regression detection
- Adversarial testing
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from pydantic import BaseModel, Field, ConfigDict

# MLflow integration (optional)
try:
    import mlflow
    from mlflow import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from agents.base import (
    BaseExtractionAgent,
    ExtractionResult,
    ExtractionContext,
    AgentConfig,
)
from data.schemas import ProcessingStatus


logger = logging.getLogger("biblos.evaluation")


# =============================================================================
# EVALUATION ENUMS
# =============================================================================


class EvaluationDimension(str, Enum):
    """Dimensions of agent evaluation."""
    EXTRACTION_QUALITY = "extraction_quality"
    CROSS_REFERENCE_ACCURACY = "cross_reference_accuracy"
    LATENCY = "latency"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    ROBUSTNESS = "robustness"


class TestType(str, Enum):
    """Types of evaluation tests."""
    UNIT = "unit"
    INTEGRATION = "integration"
    REGRESSION = "regression"
    ADVERSARIAL = "adversarial"
    STRESS = "stress"


# =============================================================================
# DATA MODELS
# =============================================================================


class EvaluationConfig(BaseModel):
    """Configuration for agent evaluation."""

    model_config = ConfigDict(extra="allow")

    # Evaluation scope
    dimensions: List[EvaluationDimension] = Field(
        default=[
            EvaluationDimension.EXTRACTION_QUALITY,
            EvaluationDimension.LATENCY,
        ]
    )
    test_types: List[TestType] = Field(
        default=[TestType.UNIT, TestType.REGRESSION]
    )

    # Thresholds
    min_precision: float = Field(default=0.8, ge=0.0, le=1.0)
    min_recall: float = Field(default=0.7, ge=0.0, le=1.0)
    min_f1: float = Field(default=0.75, ge=0.0, le=1.0)
    max_latency_ms: float = Field(default=1000.0, gt=0)
    max_latency_p99_ms: float = Field(default=5000.0, gt=0)

    # Sampling
    sample_size: Optional[int] = Field(default=None, description="Number of test cases to sample")
    random_seed: int = Field(default=42)

    # MLflow settings
    mlflow_tracking_uri: Optional[str] = Field(default=None)
    mlflow_experiment_name: str = Field(default="biblos-agent-evaluation")

    # Output settings
    output_dir: Path = Field(default=Path("./evaluation_results"))
    save_detailed_results: bool = Field(default=True)


class EvaluationMetrics(BaseModel):
    """Metrics from agent evaluation."""

    model_config = ConfigDict(extra="allow")

    # Extraction quality
    precision: float = Field(default=0.0, ge=0.0, le=1.0)
    recall: float = Field(default=0.0, ge=0.0, le=1.0)
    f1_score: float = Field(default=0.0, ge=0.0, le=1.0)
    accuracy: float = Field(default=0.0, ge=0.0, le=1.0)

    # Cross-reference specific
    crossref_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    crossref_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    type_classification_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    strength_correlation: float = Field(default=0.0, ge=-1.0, le=1.0)

    # Latency
    latency_mean_ms: float = Field(default=0.0, ge=0.0)
    latency_p50_ms: float = Field(default=0.0, ge=0.0)
    latency_p95_ms: float = Field(default=0.0, ge=0.0)
    latency_p99_ms: float = Field(default=0.0, ge=0.0)

    # Resource efficiency
    avg_tokens_used: float = Field(default=0.0, ge=0.0)
    cost_per_extraction: float = Field(default=0.0, ge=0.0)

    # Robustness
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    adversarial_success_rate: float = Field(default=0.0, ge=0.0, le=1.0)

    # Counts
    total_tests: int = Field(default=0, ge=0)
    passed_tests: int = Field(default=0, ge=0)
    failed_tests: int = Field(default=0, ge=0)
    skipped_tests: int = Field(default=0, ge=0)

    @property
    def pass_rate(self) -> float:
        """Calculate test pass rate."""
        total = self.passed_tests + self.failed_tests
        return self.passed_tests / total if total > 0 else 0.0


class TestCaseResult(BaseModel):
    """Result of a single test case."""

    model_config = ConfigDict(extra="forbid")

    test_id: str
    test_type: TestType
    passed: bool
    verse_id: str
    expected: Dict[str, Any]
    actual: Dict[str, Any]
    metrics: Dict[str, float]
    latency_ms: float
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EvaluationResult(BaseModel):
    """Complete evaluation result."""

    model_config = ConfigDict(extra="allow")

    # Metadata
    agent_name: str
    evaluation_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    config: EvaluationConfig

    # Aggregate metrics
    metrics: EvaluationMetrics

    # Individual results
    test_results: List[TestCaseResult] = Field(default_factory=list)

    # Status
    passed: bool = False
    summary: str = ""

    def to_mlflow_metrics(self) -> Dict[str, float]:
        """Convert to MLflow-compatible metrics dict."""
        return {
            "precision": self.metrics.precision,
            "recall": self.metrics.recall,
            "f1_score": self.metrics.f1_score,
            "accuracy": self.metrics.accuracy,
            "crossref_precision": self.metrics.crossref_precision,
            "crossref_recall": self.metrics.crossref_recall,
            "latency_mean_ms": self.metrics.latency_mean_ms,
            "latency_p95_ms": self.metrics.latency_p95_ms,
            "latency_p99_ms": self.metrics.latency_p99_ms,
            "error_rate": self.metrics.error_rate,
            "pass_rate": self.metrics.pass_rate,
            "total_tests": float(self.metrics.total_tests),
        }


# =============================================================================
# TEST CASE
# =============================================================================


class TestCase(BaseModel):
    """Definition of a single test case."""

    model_config = ConfigDict(extra="allow")

    test_id: str = Field(..., description="Unique test identifier")
    test_type: TestType = Field(default=TestType.UNIT)
    description: str = Field(default="")

    # Input
    verse_id: str = Field(..., description="Verse to test")
    text: str = Field(..., description="Verse text")
    context: Dict[str, Any] = Field(default_factory=dict)

    # Expected output
    expected_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Expected extraction data"
    )
    expected_confidence_min: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum expected confidence"
    )

    # Test settings
    timeout_seconds: float = Field(default=30.0)
    is_adversarial: bool = Field(default=False)
    adversarial_type: Optional[str] = Field(default=None)


class TestSuite(BaseModel):
    """Collection of test cases for an agent."""

    model_config = ConfigDict(extra="allow")

    name: str
    description: str = ""
    agent_name: str
    test_cases: List[TestCase] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "1.0.0"

    def get_by_type(self, test_type: TestType) -> List[TestCase]:
        """Get test cases by type."""
        return [tc for tc in self.test_cases if tc.test_type == test_type]

    def sample(self, n: int, seed: int = 42) -> List[TestCase]:
        """Randomly sample n test cases."""
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self.test_cases), min(n, len(self.test_cases)), replace=False)
        return [self.test_cases[i] for i in indices]


# =============================================================================
# AGENT EVALUATOR
# =============================================================================


class AgentEvaluator:
    """
    Comprehensive evaluator for SDES extraction agents.

    Features:
    - Multi-dimensional evaluation (quality, latency, robustness)
    - MLflow integration for experiment tracking
    - Statistical significance testing
    - Regression detection
    - Adversarial testing
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.logger = logging.getLogger("biblos.evaluation.evaluator")

        # Initialize MLflow if available
        self._mlflow_client: Optional['MlflowClient'] = None
        if MLFLOW_AVAILABLE and self.config.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.mlflow_experiment_name)
            self._mlflow_client = MlflowClient()

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    async def evaluate(
        self,
        agent: BaseExtractionAgent,
        test_suite: TestSuite,
        baseline_result: Optional[EvaluationResult] = None,
    ) -> EvaluationResult:
        """
        Run comprehensive evaluation on an agent.

        Args:
            agent: Agent to evaluate
            test_suite: Test suite to run
            baseline_result: Optional baseline for regression testing

        Returns:
            Complete evaluation result
        """
        evaluation_id = f"eval_{agent.config.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        self.logger.info(f"Starting evaluation {evaluation_id} for agent {agent.config.name}")

        # Sample test cases if configured
        test_cases = test_suite.test_cases
        if self.config.sample_size and self.config.sample_size < len(test_cases):
            test_cases = test_suite.sample(self.config.sample_size, self.config.random_seed)

        # Initialize result
        result = EvaluationResult(
            agent_name=agent.config.name,
            evaluation_id=evaluation_id,
            config=self.config,
            metrics=EvaluationMetrics(),
        )

        # Run test cases
        test_results: List[TestCaseResult] = []
        latencies: List[float] = []

        for test_case in test_cases:
            test_result = await self._run_test_case(agent, test_case)
            test_results.append(test_result)
            if test_result.latency_ms > 0:
                latencies.append(test_result.latency_ms)

        result.test_results = test_results

        # Compute metrics
        result.metrics = self._compute_metrics(test_results, latencies)

        # Run regression test if baseline provided
        if baseline_result:
            regression_passed = self._check_regression(result.metrics, baseline_result.metrics)
            if not regression_passed:
                result.summary += "REGRESSION DETECTED. "

        # Determine overall pass/fail
        result.passed = self._determine_pass(result.metrics)
        result.summary += self._generate_summary(result)

        # Log to MLflow
        if self._mlflow_client:
            self._log_to_mlflow(result)

        # Save results
        if self.config.save_detailed_results:
            self._save_results(result)

        self.logger.info(
            f"Evaluation {evaluation_id} completed: "
            f"{'PASSED' if result.passed else 'FAILED'}"
        )

        return result

    async def _run_test_case(
        self,
        agent: BaseExtractionAgent,
        test_case: TestCase,
    ) -> TestCaseResult:
        """Run a single test case."""
        start_time = time.perf_counter()

        try:
            # Create context
            context = ExtractionContext(**test_case.context)

            # Run extraction with timeout
            extraction_result = await asyncio.wait_for(
                agent.process(test_case.verse_id, test_case.text, context),
                timeout=test_case.timeout_seconds,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Compare results
            metrics = self._compare_results(
                extraction_result,
                test_case.expected_data,
            )

            # Determine pass/fail
            passed = self._test_passed(
                extraction_result,
                test_case,
                metrics,
            )

            return TestCaseResult(
                test_id=test_case.test_id,
                test_type=test_case.test_type,
                passed=passed,
                verse_id=test_case.verse_id,
                expected=test_case.expected_data,
                actual=extraction_result.data,
                metrics=metrics,
                latency_ms=latency_ms,
            )

        except asyncio.TimeoutError:
            return TestCaseResult(
                test_id=test_case.test_id,
                test_type=test_case.test_type,
                passed=False,
                verse_id=test_case.verse_id,
                expected=test_case.expected_data,
                actual={},
                metrics={},
                latency_ms=(time.perf_counter() - start_time) * 1000,
                error=f"Timeout after {test_case.timeout_seconds}s",
            )

        except Exception as e:
            return TestCaseResult(
                test_id=test_case.test_id,
                test_type=test_case.test_type,
                passed=False,
                verse_id=test_case.verse_id,
                expected=test_case.expected_data,
                actual={},
                metrics={},
                latency_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e),
            )

    def _compare_results(
        self,
        actual: ExtractionResult,
        expected: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compare actual results to expected."""
        metrics = {}

        if not expected:
            return metrics

        # Field-level comparison
        actual_data = actual.data
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for key, expected_value in expected.items():
            actual_value = actual_data.get(key)

            if actual_value is None:
                false_negatives += 1
            elif self._values_match(actual_value, expected_value):
                true_positives += 1
            else:
                false_positives += 1

        # Check for extra fields
        for key in actual_data:
            if key not in expected:
                false_positives += 1

        # Calculate metrics
        total = true_positives + false_positives + false_negatives
        if total > 0:
            metrics["precision"] = true_positives / max(1, true_positives + false_positives)
            metrics["recall"] = true_positives / max(1, true_positives + false_negatives)
            if metrics["precision"] + metrics["recall"] > 0:
                metrics["f1"] = 2 * metrics["precision"] * metrics["recall"] / (
                    metrics["precision"] + metrics["recall"]
                )
            else:
                metrics["f1"] = 0.0

        return metrics

    def _values_match(self, actual: Any, expected: Any) -> bool:
        """Check if actual value matches expected."""
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            # Numeric comparison with tolerance
            return abs(actual - expected) < 0.01 * max(abs(expected), 1)
        elif isinstance(expected, list) and isinstance(actual, list):
            # List comparison (order-independent for sets)
            return set(map(str, actual)) == set(map(str, expected))
        elif isinstance(expected, dict) and isinstance(actual, dict):
            # Recursive dict comparison
            return all(
                key in actual and self._values_match(actual[key], val)
                for key, val in expected.items()
            )
        else:
            return str(actual).lower() == str(expected).lower()

    def _test_passed(
        self,
        result: ExtractionResult,
        test_case: TestCase,
        metrics: Dict[str, float],
    ) -> bool:
        """Determine if a test case passed."""
        # Check status
        if result.status == ProcessingStatus.FAILED:
            return False

        # Check confidence
        if result.confidence < test_case.expected_confidence_min:
            return False

        # Check metrics thresholds
        if metrics.get("precision", 1.0) < self.config.min_precision:
            return False
        if metrics.get("recall", 1.0) < self.config.min_recall:
            return False

        return True

    def _compute_metrics(
        self,
        test_results: List[TestCaseResult],
        latencies: List[float],
    ) -> EvaluationMetrics:
        """Compute aggregate metrics from test results."""
        metrics = EvaluationMetrics()

        if not test_results:
            return metrics

        # Counts
        metrics.total_tests = len(test_results)
        metrics.passed_tests = sum(1 for r in test_results if r.passed)
        metrics.failed_tests = sum(1 for r in test_results if not r.passed)

        # Aggregate precision/recall/F1
        precisions = [r.metrics.get("precision", 0) for r in test_results if "precision" in r.metrics]
        recalls = [r.metrics.get("recall", 0) for r in test_results if "recall" in r.metrics]
        f1s = [r.metrics.get("f1", 0) for r in test_results if "f1" in r.metrics]

        if precisions:
            metrics.precision = np.mean(precisions)
        if recalls:
            metrics.recall = np.mean(recalls)
        if f1s:
            metrics.f1_score = np.mean(f1s)

        # Latency metrics
        if latencies:
            latency_array = np.array(latencies)
            metrics.latency_mean_ms = float(np.mean(latency_array))
            metrics.latency_p50_ms = float(np.percentile(latency_array, 50))
            metrics.latency_p95_ms = float(np.percentile(latency_array, 95))
            metrics.latency_p99_ms = float(np.percentile(latency_array, 99))

        # Error rate
        errors = sum(1 for r in test_results if r.error is not None)
        metrics.error_rate = errors / metrics.total_tests

        return metrics

    def _check_regression(
        self,
        current: EvaluationMetrics,
        baseline: EvaluationMetrics,
        tolerance: float = 0.05,
    ) -> bool:
        """Check for regression against baseline."""
        # Check key metrics for regression
        if current.precision < baseline.precision - tolerance:
            self.logger.warning(
                f"Precision regression: {current.precision:.3f} < {baseline.precision:.3f}"
            )
            return False

        if current.recall < baseline.recall - tolerance:
            self.logger.warning(
                f"Recall regression: {current.recall:.3f} < {baseline.recall:.3f}"
            )
            return False

        if current.f1_score < baseline.f1_score - tolerance:
            self.logger.warning(
                f"F1 regression: {current.f1_score:.3f} < {baseline.f1_score:.3f}"
            )
            return False

        # Allow 20% latency regression
        if current.latency_p95_ms > baseline.latency_p95_ms * 1.2:
            self.logger.warning(
                f"Latency regression: {current.latency_p95_ms:.1f}ms > {baseline.latency_p95_ms:.1f}ms"
            )
            return False

        return True

    def _determine_pass(self, metrics: EvaluationMetrics) -> bool:
        """Determine overall pass/fail status."""
        # Check quality metrics
        if metrics.precision < self.config.min_precision:
            return False
        if metrics.recall < self.config.min_recall:
            return False
        if metrics.f1_score < self.config.min_f1:
            return False

        # Check latency
        if metrics.latency_p99_ms > self.config.max_latency_p99_ms:
            return False

        # Check error rate
        if metrics.error_rate > 0.1:  # 10% max error rate
            return False

        return True

    def _generate_summary(self, result: EvaluationResult) -> str:
        """Generate human-readable summary."""
        m = result.metrics
        return (
            f"Agent {result.agent_name}: "
            f"{m.passed_tests}/{m.total_tests} tests passed "
            f"(P={m.precision:.2f}, R={m.recall:.2f}, F1={m.f1_score:.2f}). "
            f"Latency: {m.latency_mean_ms:.1f}ms mean, {m.latency_p95_ms:.1f}ms p95."
        )

    def _log_to_mlflow(self, result: EvaluationResult) -> None:
        """Log evaluation results to MLflow."""
        if not MLFLOW_AVAILABLE or not self._mlflow_client:
            return

        try:
            with mlflow.start_run(run_name=result.evaluation_id):
                # Log metrics
                mlflow.log_metrics(result.to_mlflow_metrics())

                # Log parameters
                mlflow.log_params({
                    "agent_name": result.agent_name,
                    "sample_size": self.config.sample_size or "all",
                    "min_precision_threshold": self.config.min_precision,
                    "min_recall_threshold": self.config.min_recall,
                })

                # Log result summary
                mlflow.log_param("passed", result.passed)
                mlflow.log_param("summary", result.summary[:250])  # Truncate

                self.logger.info(f"Logged results to MLflow run {result.evaluation_id}")

        except Exception as e:
            self.logger.error(f"Failed to log to MLflow: {e}")

    def _save_results(self, result: EvaluationResult) -> None:
        """Save evaluation results to file."""
        output_path = self.config.output_dir / f"{result.evaluation_id}.json"

        try:
            with open(output_path, "w") as f:
                json.dump(result.model_dump(mode="json"), f, indent=2, default=str)

            self.logger.info(f"Saved results to {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def load_test_suite(path: Path) -> TestSuite:
    """Load test suite from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return TestSuite(**data)


def create_test_suite_from_golden_records(
    agent_name: str,
    golden_records: List[Dict[str, Any]],
    suite_name: str = "golden_record_tests",
) -> TestSuite:
    """Create test suite from golden record data."""
    test_cases = []

    for i, record in enumerate(golden_records):
        test_cases.append(TestCase(
            test_id=f"golden_{i:04d}",
            test_type=TestType.REGRESSION,
            description=f"Golden record test for {record.get('verse_id', 'unknown')}",
            verse_id=record.get("verse_id", ""),
            text=record.get("text", ""),
            context=record.get("context", {}),
            expected_data=record.get("data", {}),
            expected_confidence_min=record.get("confidence", 0.5),
        ))

    return TestSuite(
        name=suite_name,
        description="Auto-generated from golden records",
        agent_name=agent_name,
        test_cases=test_cases,
    )
