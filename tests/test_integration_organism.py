"""
BIBLOS v2 - Integration Tests: The Organism Breathes

These tests verify that the seraphic architecture functions as a unified whole.
Each test doesn't check isolated parts but rather the interpenetration of
all components working together.

Test Philosophy:
    We don't test "does the database work" and "does the mediator work" separately.
    We test "can the organism contemplate scripture" - which requires all parts
    functioning in harmony.

    A seraph doesn't have "working wings" separate from "working eyes" - it either
    IS a seraph (fully functional) or it is not. Our tests reflect this unity.
"""

import asyncio
import pytest
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch


# ============================================================================
# Test Fixtures - Preparing the Sacred Space
# ============================================================================

@pytest.fixture
def biblos_config():
    """Create a test configuration for the organism."""
    from biblos import BiblosConfig

    return BiblosConfig(
        environment="test",
        instance_id="biblos-test",
        enable_ml_inference=False,  # Disable for faster tests
        enable_graph_analysis=False,
        enable_patristic_integration=False,
        max_concurrent_verses=5,
        processing_timeout_seconds=30.0,
    )


@pytest.fixture
def mock_mediator():
    """Create a mock mediator for testing without full infrastructure."""
    mediator = MagicMock()
    mediator.send = AsyncMock(return_value={
        "linguistic": {"morphology": "analyzed"},
        "theological": {"themes": ["creation", "divine action"]},
    })
    return mediator


# ============================================================================
# Tests: The Organism's Existence
# ============================================================================

class TestOrganismExistence:
    """Tests for the organism's basic existence and lifecycle."""

    def test_cannot_construct_directly(self):
        """The organism should be awakened, not constructed."""
        from biblos import BIBLOS

        with pytest.raises(RuntimeError, match="awakened"):
            BIBLOS()

    @pytest.mark.asyncio
    async def test_can_create_via_factory(self, biblos_config):
        """The organism can be created via the factory method."""
        from biblos import BIBLOS, SystemState

        # Create without auto-awakening
        organism = await BIBLOS.create(biblos_config, auto_awaken=False)

        assert organism is not None
        assert organism.state == SystemState.DORMANT

    @pytest.mark.asyncio
    async def test_singleton_tracking(self, biblos_config):
        """The organism tracks itself as a singleton."""
        from biblos import BIBLOS, get_biblos

        organism = await BIBLOS.create(biblos_config, auto_awaken=False)

        assert get_biblos() is organism

    @pytest.mark.asyncio
    async def test_context_manager_lifecycle(self, biblos_config):
        """The organism manages its lifecycle via context manager."""
        from biblos import BIBLOS, SystemState

        # Patch the awakening to avoid full bootstrap
        with patch.object(BIBLOS, '_awaken', new_callable=AsyncMock) as mock_awaken, \
             patch.object(BIBLOS, '_ascend', new_callable=AsyncMock) as mock_ascend:

            mock_awaken.side_effect = lambda: setattr(
                mock_awaken._mock_self, '_state', SystemState.ALIVE
            )

            organism = await BIBLOS.create(biblos_config, auto_awaken=False)
            organism._state = SystemState.DORMANT

            async with organism:
                mock_awaken.assert_called_once()

            mock_ascend.assert_called_once()


# ============================================================================
# Tests: The Organism's Health
# ============================================================================

class TestOrganismHealth:
    """Tests for the organism's holographic health system."""

    @pytest.mark.asyncio
    async def test_health_reflects_state(self, biblos_config):
        """Health status reflects the organism's state."""
        from biblos import BIBLOS, SystemState

        organism = await BIBLOS.create(biblos_config, auto_awaken=False)

        # Dormant organism should not be healthy
        health = organism.health
        assert health.state == SystemState.DORMANT
        assert not health.healthy  # Not alive = not healthy

    @pytest.mark.asyncio
    async def test_health_contains_all_organs(self, biblos_config):
        """Health report includes all critical organs."""
        from biblos import BIBLOS

        organism = await BIBLOS.create(biblos_config, auto_awaken=False)
        health = organism.health

        # Should have organs
        assert len(health.organs) > 0

        # Should have critical organs
        organ_names = {o.name for o in health.organs}
        assert "domain" in organ_names
        assert "container" in organ_names
        assert "application" in organ_names

    @pytest.mark.asyncio
    async def test_health_is_holographic(self, biblos_config):
        """Health is holographic - each part reflects the whole."""
        from biblos import BIBLOS

        organism = await BIBLOS.create(biblos_config, auto_awaken=False)

        # Get specific organ health
        health = organism.health
        domain_health = health.organ("domain")

        assert domain_health is not None
        # Domain not initialized in dormant state
        assert not domain_health.healthy


# ============================================================================
# Tests: The Organism's Sacred Purpose
# ============================================================================

class TestOrganismPurpose:
    """Tests for the organism's primary function: processing scripture."""

    @pytest.mark.asyncio
    async def test_can_process_verse(self, biblos_config, mock_mediator):
        """The organism can process a verse."""
        from biblos import BIBLOS, SystemState

        organism = await BIBLOS.create(biblos_config, auto_awaken=False)
        organism._state = SystemState.ALIVE
        organism._mediator = mock_mediator

        result = await organism.process_verse("GEN.1.1")

        assert result.verse_id == "GEN.1.1"
        assert result.success
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_processing_uses_mediator(self, biblos_config, mock_mediator):
        """Processing uses the mediator - the organism's consciousness."""
        from biblos import BIBLOS, SystemState

        organism = await BIBLOS.create(biblos_config, auto_awaken=False)
        organism._state = SystemState.ALIVE
        organism._mediator = mock_mediator

        await organism.process_verse("GEN.1.1")

        # Mediator should have been called
        mock_mediator.send.assert_called()

    @pytest.mark.asyncio
    async def test_processing_returns_analysis(self, biblos_config, mock_mediator):
        """Processing returns the fruit of contemplation."""
        from biblos import BIBLOS, SystemState

        organism = await BIBLOS.create(biblos_config, auto_awaken=False)
        organism._state = SystemState.ALIVE
        organism._mediator = mock_mediator

        result = await organism.process_verse(
            "GEN.1.1",
            include_linguistic=True,
            include_theological=True,
        )

        assert "morphology" in result.linguistic_analysis
        assert "themes" in result.theological_insights

    @pytest.mark.asyncio
    async def test_graceful_degradation_without_mediator(self, biblos_config):
        """Organism degrades gracefully if mediator unavailable."""
        from biblos import BIBLOS, SystemState

        organism = await BIBLOS.create(biblos_config, auto_awaken=False)
        organism._state = SystemState.ALIVE
        organism._mediator = None

        # Should not raise, should return partial result
        result = await organism.process_verse("GEN.1.1")

        assert result.verse_id == "GEN.1.1"
        # Success may be true even without mediator - graceful degradation


# ============================================================================
# Tests: Cross-Reference Discovery
# ============================================================================

class TestCrossReferenceDiscovery:
    """Tests for the organism's perception of the SPIDERWEB."""

    @pytest.mark.asyncio
    async def test_can_discover_references(self, biblos_config):
        """The organism can discover cross-references."""
        from biblos import BIBLOS, SystemState

        mock_mediator = MagicMock()
        mock_mediator.send = AsyncMock(return_value={
            "cross_references": [
                {"target": "JHN.1.1", "strength": "strong", "type": "typological"},
                {"target": "HEB.1.1", "strength": "moderate", "type": "thematic"},
            ]
        })

        organism = await BIBLOS.create(biblos_config, auto_awaken=False)
        organism._state = SystemState.ALIVE
        organism._mediator = mock_mediator

        refs = await organism.discover_cross_references("GEN.1.1", top_k=5)

        assert len(refs) == 2
        assert refs[0]["target"] == "JHN.1.1"

    @pytest.mark.asyncio
    async def test_empty_refs_without_mediator(self, biblos_config):
        """Returns empty list gracefully without mediator."""
        from biblos import BIBLOS, SystemState

        organism = await BIBLOS.create(biblos_config, auto_awaken=False)
        organism._state = SystemState.ALIVE
        organism._mediator = None

        refs = await organism.discover_cross_references("GEN.1.1")

        assert refs == []


# ============================================================================
# Tests: Golden Record Certification
# ============================================================================

class TestGoldenRecordCertification:
    """Tests for the organism's witness to certified truth."""

    @pytest.mark.asyncio
    async def test_can_certify_golden_record(self, biblos_config):
        """The organism can certify a golden record."""
        from biblos import BIBLOS, SystemState

        mock_mediator = MagicMock()
        mock_mediator.send = AsyncMock(return_value={
            "verse_id": "GEN.1.1",
            "certified": True,
            "certified_at": 1234567890.0,
            "quality_tier": "platinum",
        })

        organism = await BIBLOS.create(biblos_config, auto_awaken=False)
        organism._state = SystemState.ALIVE
        organism._mediator = mock_mediator

        golden = await organism.certify_golden_record("GEN.1.1")

        assert golden["verse_id"] == "GEN.1.1"
        assert golden["certified"] is True

    @pytest.mark.asyncio
    async def test_uncertified_without_mediator(self, biblos_config):
        """Returns uncertified status without mediator."""
        from biblos import BIBLOS, SystemState

        organism = await BIBLOS.create(biblos_config, auto_awaken=False)
        organism._state = SystemState.ALIVE
        organism._mediator = None

        golden = await organism.certify_golden_record("GEN.1.1")

        assert golden["certified"] is False


# ============================================================================
# Tests: Seraphic Interpenetration
# ============================================================================

class TestSeraphicInterpenetration:
    """
    Tests that verify the seraphic principle: parts contain the whole.

    These tests check that accessing any part of the system gives you
    a view of the whole, not just that part. Like how each of a seraph's
    eyes sees the same reality from a different angle.
    """

    @pytest.mark.asyncio
    async def test_processing_includes_all_aspects(self, biblos_config):
        """Processing a verse touches all aspects of the organism."""
        from biblos import BIBLOS, SystemState

        mock_mediator = MagicMock()
        mock_mediator.send = AsyncMock(return_value={
            "linguistic": {"parsed": True},
            "theological": {"analyzed": True},
        })

        organism = await BIBLOS.create(biblos_config, auto_awaken=False)
        organism._state = SystemState.ALIVE
        organism._mediator = mock_mediator

        result = await organism.process_verse(
            "GEN.1.1",
            include_linguistic=True,
            include_theological=True,
            include_cross_references=True,
        )

        # Result should contain linguistic analysis
        assert result.linguistic_analysis

        # Result should contain theological insights
        assert result.theological_insights

        # Cross-references were requested
        # (may be empty if mediator doesn't provide them)
        assert isinstance(result.cross_references, list)

    @pytest.mark.asyncio
    async def test_state_changes_are_consistent(self, biblos_config):
        """State changes are consistent across the whole organism."""
        from biblos import BIBLOS, SystemState

        organism = await BIBLOS.create(biblos_config, auto_awaken=False)

        # Initial state
        assert organism.state == SystemState.DORMANT
        assert organism.health.state == SystemState.DORMANT

        # Manually change state (simulating awakening)
        organism._state = SystemState.ALIVE

        # Both should reflect the change
        assert organism.state == SystemState.ALIVE
        assert organism.health.state == SystemState.ALIVE


# ============================================================================
# Module Import Tests
# ============================================================================

class TestModuleImports:
    """Tests that verify all module exports are properly wired."""

    def test_core_exports(self):
        """Core module exports all expected components."""
        # This will fail if imports are broken
        from core import (
            # Errors
            BiblosError,
            # Resilience
            CircuitBreaker,
            # Async
            AsyncTaskGroup,
            # Types
            VerseId,
        )

    def test_di_exports(self):
        """DI module exports all expected components."""
        from di import (
            Container,
            ServiceLifetime,
            IServiceProvider,
        )

    def test_domain_exports(self):
        """Domain module exports all expected components."""
        from domain import (
            VerseAggregate,
            DomainEvent,
            Command,
            Mediator,
        )

    def test_biblos_exports(self):
        """Biblos module exports all expected components."""
        from biblos import (
            BIBLOS,
            SystemState,
            BiblosConfig,
            create_biblos,
        )


# ============================================================================
# Performance Tests (Optional)
# ============================================================================

class TestPerformance:
    """Basic performance tests for the organism."""

    @pytest.mark.asyncio
    async def test_rapid_health_checks(self, biblos_config):
        """Health checks should be fast."""
        import time
        from biblos import BIBLOS

        organism = await BIBLOS.create(biblos_config, auto_awaken=False)

        start = time.time()
        for _ in range(100):
            _ = organism.health
        elapsed = time.time() - start

        # 100 health checks should complete in under 100ms
        assert elapsed < 0.1, f"Health checks too slow: {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_concurrent_processing(self, biblos_config, mock_mediator):
        """Multiple verses can be processed concurrently."""
        from biblos import BIBLOS, SystemState

        organism = await BIBLOS.create(biblos_config, auto_awaken=False)
        organism._state = SystemState.ALIVE
        organism._mediator = mock_mediator
        organism._processing_semaphore = asyncio.Semaphore(5)

        # Process multiple verses concurrently
        verses = [f"GEN.1.{i}" for i in range(1, 6)]
        tasks = [organism.process_verse(v) for v in verses]

        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(r.success for r in results)
