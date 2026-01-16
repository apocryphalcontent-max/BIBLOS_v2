"""
Property-Based Tests for Pipeline Invariants

Tests pipeline execution invariants, golden record validation, and phase consistency.
"""
import pytest
from hypothesis import given, strategies as st, settings, assume
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant, precondition
import time

from data.schemas import (
    PipelineResultSchema,
    PhaseResultSchema,
    GoldenRecordSchema,
    ProcessingStatus,
)
from tests.property.strategies import (
    verse_id_strategy,
    processing_status_strategy,
    phase_name_strategy,
    pipeline_metrics_strategy,
    golden_record_schema_strategy,
)


class TestPipelineResultInvariants:
    """Property-based tests for pipeline result invariants."""

    @given(
        verse_id_strategy(valid_only=True),
        processing_status_strategy(),
        st.floats(min_value=0.0, max_value=3600.0, allow_nan=False),
        st.floats(min_value=0.0, max_value=3600.0, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_pipeline_result_time_consistency(self, verse_id, status, start_time, duration):
        """End time should be greater than or equal to start time."""
        end_time = start_time + duration

        result = PipelineResultSchema(
            verse_id=verse_id,
            status=status,
            start_time=start_time,
            end_time=end_time,
        )

        # Duration should be non-negative
        assert result.duration >= 0
        assert result.end_time >= result.start_time

    @given(
        verse_id_strategy(valid_only=True),
        processing_status_strategy(),
    )
    @settings(max_examples=200)
    def test_status_is_valid_enum(self, verse_id, status):
        """Pipeline status must be valid enum value."""
        result = PipelineResultSchema(
            verse_id=verse_id,
            status=status,
        )

        valid_statuses = [e.value for e in ProcessingStatus]
        assert result.status in valid_statuses

    @given(verse_id_strategy(valid_only=True))
    @settings(max_examples=200)
    def test_empty_phase_results_valid(self, verse_id):
        """Pipeline with no phase results should be valid."""
        result = PipelineResultSchema(
            verse_id=verse_id,
            status="pending",
            phase_results={},
        )

        assert isinstance(result.phase_results, dict)
        assert len(result.phase_results) == 0

    @given(
        verse_id_strategy(valid_only=True),
        st.lists(st.text(min_size=1, max_size=100), max_size=10),
    )
    @settings(max_examples=200)
    def test_errors_list_handling(self, verse_id, errors):
        """Errors list should handle arbitrary text."""
        result = PipelineResultSchema(
            verse_id=verse_id,
            status="failed",
            errors=errors,
        )

        assert isinstance(result.errors, list)
        assert len(result.errors) == len(errors)


class TestPhaseResultInvariants:
    """Property-based tests for phase result invariants."""

    @given(
        phase_name_strategy(),
        processing_status_strategy(),
        st.floats(min_value=0.0, max_value=3600.0, allow_nan=False),
        st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_phase_duration_consistency(self, phase_name, status, start_time, duration):
        """Phase duration should be calculated correctly."""
        end_time = start_time + duration

        phase = PhaseResultSchema(
            phase_name=phase_name,
            status=status,
            start_time=start_time,
            end_time=end_time,
        )

        # Duration property should match calculation
        assert phase.duration == end_time - start_time
        assert phase.duration >= 0

    @given(
        phase_name_strategy(),
        st.dictionaries(
            st.text(min_size=1, max_size=50),
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            max_size=20
        )
    )
    @settings(max_examples=200)
    def test_metrics_dict_handling(self, phase_name, metrics):
        """Metrics should handle arbitrary float dictionaries."""
        phase = PhaseResultSchema(
            phase_name=phase_name,
            status="completed",
            metrics=metrics,
        )

        assert isinstance(phase.metrics, dict)
        assert len(phase.metrics) == len(metrics)

        for key, value in metrics.items():
            assert phase.metrics[key] == value

    @given(phase_name_strategy())
    @settings(max_examples=100)
    def test_phase_with_error_has_failed_status(self, phase_name):
        """Phase with error message should typically have failed status."""
        phase = PhaseResultSchema(
            phase_name=phase_name,
            status="failed",
            error="Something went wrong",
        )

        # Error should be set
        assert phase.error is not None
        assert len(phase.error) > 0

        # Status should reflect failure
        assert phase.status == "failed"


class TestGoldenRecordInvariants:
    """Property-based tests for golden record invariants."""

    @given(golden_record_schema_strategy())
    @settings(max_examples=200)
    def test_golden_record_structure(self, record):
        """Golden records should have valid structure."""
        assert record.verse_id is not None
        assert isinstance(record.verse_id, str)
        assert isinstance(record.certification, dict)
        assert isinstance(record.data, dict)
        assert isinstance(record.phases_executed, list)

    @given(golden_record_schema_strategy())
    @settings(max_examples=200)
    def test_certification_structure(self, record):
        """Certification should have expected fields."""
        cert = record.certification
        assert "level" in cert
        assert "score" in cert
        assert "validation_passed" in cert
        assert "quality_passed" in cert

    @given(golden_record_schema_strategy())
    @settings(max_examples=200)
    def test_agent_count_non_negative(self, record):
        """Agent count should be non-negative."""
        assert record.agent_count >= 0

    @given(golden_record_schema_strategy())
    @settings(max_examples=200)
    def test_processing_time_non_negative(self, record):
        """Total processing time should be non-negative."""
        assert record.total_processing_time >= 0.0


class TestPipelinePhaseConsistency:
    """Tests for consistency between pipeline and phase results."""

    @given(
        verse_id_strategy(valid_only=True),
        st.lists(phase_name_strategy(), min_size=1, max_size=4, unique=True),
        st.floats(min_value=0.0, max_value=1000.0, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_phase_times_sum_to_pipeline_time(self, verse_id, phase_names, base_time):
        """Sum of phase durations should not exceed pipeline duration."""
        # Create phases with times
        phase_results = {}
        phase_start = base_time
        total_phase_duration = 0.0

        for i, phase_name in enumerate(phase_names):
            phase_duration = 1.0 + (i * 0.5)  # Incremental durations
            phase_end = phase_start + phase_duration

            phase_results[phase_name] = PhaseResultSchema(
                phase_name=phase_name,
                status="completed",
                start_time=phase_start,
                end_time=phase_end,
            )

            total_phase_duration += phase_duration
            phase_start = phase_end

        # Create pipeline result
        pipeline_start = base_time
        pipeline_end = phase_start  # Phases run sequentially

        pipeline = PipelineResultSchema(
            verse_id=verse_id,
            status="completed",
            phase_results=phase_results,
            start_time=pipeline_start,
            end_time=pipeline_end,
        )

        # Pipeline duration should encompass all phases
        assert pipeline.duration >= total_phase_duration - 0.01  # Allow small floating point error

    @given(
        verse_id_strategy(valid_only=True),
        st.lists(phase_name_strategy(), min_size=1, max_size=4, unique=True),
    )
    @settings(max_examples=200)
    def test_all_phases_have_valid_status(self, verse_id, phase_names):
        """All phases should have valid status values."""
        phase_results = {}

        for phase_name in phase_names:
            status = "completed"
            phase_results[phase_name] = PhaseResultSchema(
                phase_name=phase_name,
                status=status,
            )

        pipeline = PipelineResultSchema(
            verse_id=verse_id,
            status="completed",
            phase_results=phase_results,
        )

        valid_statuses = [e.value for e in ProcessingStatus]

        for phase_name, phase in pipeline.phase_results.items():
            assert phase.status in valid_statuses

    @given(
        verse_id_strategy(valid_only=True),
        st.lists(phase_name_strategy(), min_size=1, max_size=4, unique=True),
    )
    @settings(max_examples=200)
    def test_completed_pipeline_has_completed_phases(self, verse_id, phase_names):
        """Completed pipeline should have all phases completed or skipped."""
        phase_results = {}

        for phase_name in phase_names:
            # All phases completed
            phase_results[phase_name] = PhaseResultSchema(
                phase_name=phase_name,
                status="completed",
            )

        pipeline = PipelineResultSchema(
            verse_id=verse_id,
            status="completed",
            phase_results=phase_results,
        )

        # Check consistency
        for phase_name, phase in pipeline.phase_results.items():
            assert phase.status in ["completed", "skipped"]


class TestGoldenRecordValidation:
    """Tests for golden record validation logic."""

    @given(
        verse_id_strategy(valid_only=True),
        st.lists(phase_name_strategy(), min_size=1, max_size=4, unique=True),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_golden_record_from_completed_pipeline(self, verse_id, phase_names, cert_score):
        """Golden record from completed pipeline should be valid."""
        golden = GoldenRecordSchema(
            verse_id=verse_id,
            text="Sample verse text",
            certification={
                "level": "gold" if cert_score > 0.9 else "silver" if cert_score > 0.7 else "bronze",
                "score": cert_score,
                "validation_passed": True,
                "quality_passed": cert_score > 0.7,
            },
            phases_executed=phase_names,
            agent_count=len(phase_names) * 6,  # Assume 6 agents per phase
        )

        assert golden.verse_id == verse_id
        assert len(golden.phases_executed) == len(phase_names)
        assert golden.certification["score"] == cert_score

    @given(
        verse_id_strategy(valid_only=True),
        st.integers(min_value=0, max_value=30),
        st.floats(min_value=0.0, max_value=3600.0, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_golden_record_metrics_consistency(self, verse_id, agent_count, processing_time):
        """Golden record metrics should be consistent."""
        golden = GoldenRecordSchema(
            verse_id=verse_id,
            text="Sample text",
            agent_count=agent_count,
            total_processing_time=processing_time,
        )

        # If we have agents, we should have processing time
        if agent_count > 0:
            # Processing time could be zero for very fast agents
            assert processing_time >= 0.0

        # Metrics should be accessible
        assert golden.agent_count == agent_count
        assert golden.total_processing_time == processing_time


# =============================================================================
# STATEFUL TESTING - PIPELINE EXECUTION
# =============================================================================

class PipelineStateMachine(RuleBasedStateMachine):
    """Stateful testing for pipeline execution."""

    def __init__(self):
        super().__init__()
        self.verse_id = "GEN.1.1"
        self.pipeline = PipelineResultSchema(
            verse_id=self.verse_id,
            status="pending",
            start_time=time.time(),
        )
        self.phases = ["linguistic", "theological", "intertextual", "validation"]
        self.current_phase_index = 0

    @precondition(lambda self: self.current_phase_index < len(self.phases))
    @rule()
    def execute_next_phase(self):
        """Execute the next phase."""
        phase_name = self.phases[self.current_phase_index]
        start = time.time()
        end = start + 0.001  # Tiny duration

        phase = PhaseResultSchema(
            phase_name=phase_name,
            status="completed",
            start_time=start,
            end_time=end,
        )

        self.pipeline.phase_results[phase_name] = phase
        self.current_phase_index += 1

        # Update pipeline status
        if self.current_phase_index >= len(self.phases):
            self.pipeline.status = "completed"
            self.pipeline.end_time = time.time()

    @rule()
    def check_status(self):
        """Check pipeline status."""
        valid_statuses = [e.value for e in ProcessingStatus]
        assert self.pipeline.status in valid_statuses

    @invariant()
    def status_valid(self):
        """Status should always be valid."""
        valid_statuses = [e.value for e in ProcessingStatus]
        assert self.pipeline.status in valid_statuses

    @invariant()
    def verse_id_unchanged(self):
        """Verse ID should never change."""
        assert self.pipeline.verse_id == self.verse_id

    @invariant()
    def phase_count_consistent(self):
        """Phase count should match executed phases."""
        assert len(self.pipeline.phase_results) == self.current_phase_index


TestPipelineExecution = PipelineStateMachine.TestCase
