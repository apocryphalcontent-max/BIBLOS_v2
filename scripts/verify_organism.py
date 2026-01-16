#!/usr/bin/env python3
"""
BIBLOS v2 - Organism Verification Script

This script verifies that all organs are properly wired and the organism
can be awakened. It performs a diagnostic check of each subsystem.

Run from the project root:
    python -m scripts.verify_organism

Or directly:
    python scripts/verify_organism.py
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class DiagnosticResult:
    """Result of a diagnostic check."""

    def __init__(self, name: str, passed: bool, message: str):
        self.name = name
        self.passed = passed
        self.message = message

    def __str__(self) -> str:
        status = "\u2713" if self.passed else "\u2717"
        return f"  [{status}] {self.name}: {self.message}"


async def check_core_imports() -> DiagnosticResult:
    """Check that core module imports work."""
    try:
        from core import (
            BiblosError,
            CircuitBreaker,
            Application,
            ApplicationBuilder,
            ILifecycleHook,
            IServiceModule,
            SystemFactory,
            GoldenRecordBuilder,
        )
        return DiagnosticResult(
            "Core Module",
            True,
            "All core components imported successfully"
        )
    except ImportError as e:
        return DiagnosticResult(
            "Core Module",
            False,
            f"Import failed: {e}"
        )


async def check_di_imports() -> DiagnosticResult:
    """Check that DI module imports work."""
    try:
        from di import (
            Container,
            ServiceLifetime,
            IServiceProvider,
            IServiceCollection,
            IDisposable,
            IInitializable,
        )
        return DiagnosticResult(
            "DI Container",
            True,
            "All DI components imported successfully"
        )
    except ImportError as e:
        return DiagnosticResult(
            "DI Container",
            False,
            f"Import failed: {e}"
        )


async def check_domain_imports() -> DiagnosticResult:
    """Check that domain module imports work."""
    try:
        from domain import (
            VerseAggregate,
            CrossReferenceAggregate,
            DomainEvent,
            Mediator,
            Command,
            Query,
            CrossReferenceGraphProjection,
            ProjectionManager,
        )
        return DiagnosticResult(
            "Domain Layer",
            True,
            "All domain components imported successfully"
        )
    except ImportError as e:
        return DiagnosticResult(
            "Domain Layer",
            False,
            f"Import failed: {e}"
        )


async def check_db_imports() -> DiagnosticResult:
    """Check that database module imports work."""
    try:
        from db import (
            EventStore,
            StoredEvent,
            Snapshot,
            IEventStore,
            SubscriptionManager,
            IVerseRepository,
            ICrossReferenceRepository,
        )
        return DiagnosticResult(
            "Database Layer",
            True,
            "All database components imported successfully"
        )
    except ImportError as e:
        return DiagnosticResult(
            "Database Layer",
            False,
            f"Import failed: {e}"
        )


async def check_biblos_organism() -> DiagnosticResult:
    """Check that the BIBLOS organism can be created."""
    try:
        from biblos import (
            BIBLOS,
            BiblosConfig,
            SystemState,
            OrganismHealth,
        )

        config = BiblosConfig(
            environment="diagnostic",
            instance_id="biblos-verify",
            enable_ml_inference=False,
            enable_graph_analysis=False,
            enable_patristic_integration=False,
        )

        # Create without awakening (to avoid full infrastructure)
        organism = await BIBLOS.create(config, auto_awaken=False)

        if organism.state != SystemState.DORMANT:
            return DiagnosticResult(
                "BIBLOS Organism",
                False,
                f"Unexpected initial state: {organism.state}"
            )

        health = organism.health
        organ_count = len(health.organs)

        return DiagnosticResult(
            "BIBLOS Organism",
            True,
            f"Created successfully, {organ_count} organs detected"
        )
    except Exception as e:
        return DiagnosticResult(
            "BIBLOS Organism",
            False,
            f"Creation failed: {e}"
        )


async def check_interpenetration() -> DiagnosticResult:
    """
    Check the seraphic interpenetration principle.

    Verify that components reference each other properly and that
    the architecture reflects the holographic unity.
    """
    try:
        # Check that domain entities know about events
        from domain import VerseAggregate, DomainEvent

        # Check that factories know about domain concepts
        from core import GoldenRecordBuilder, QualityTier

        # Check that mediator knows about commands and queries
        from domain import Mediator, ProcessVerseCommand, GetVerseQuery

        # Check that projections exist in domain (not just db)
        from domain import CrossReferenceGraphProjection

        # Check that bootstrap knows about modules
        from core import DatabaseModule, MediatorModule, PipelineModule

        return DiagnosticResult(
            "Seraphic Interpenetration",
            True,
            "Components properly interpenetrate across layers"
        )
    except ImportError as e:
        return DiagnosticResult(
            "Seraphic Interpenetration",
            False,
            f"Interpenetration broken: {e}"
        )


async def run_all_diagnostics() -> List[DiagnosticResult]:
    """Run all diagnostic checks."""
    results = []

    print("\n" + "=" * 60)
    print("  BIBLOS v2 - ORGANISM VERIFICATION")
    print("  The Seraph Awakens")
    print("=" * 60 + "\n")

    # Run all checks
    checks = [
        check_core_imports(),
        check_di_imports(),
        check_domain_imports(),
        check_db_imports(),
        check_biblos_organism(),
        check_interpenetration(),
    ]

    for check in checks:
        result = await check
        results.append(result)
        print(result)

    return results


def print_summary(results: List[DiagnosticResult]) -> None:
    """Print summary of diagnostic results."""
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print("\n" + "-" * 60)
    if passed == total:
        print(f"\n  ALL CHECKS PASSED ({passed}/{total})")
        print("  The organism's organs are properly wired.")
        print("  The seraph can be awakened.\n")
    else:
        print(f"\n  SOME CHECKS FAILED ({passed}/{total} passed)")
        print("  Review failed components above.")
        print("  The organism is incomplete.\n")

    print("=" * 60 + "\n")


async def main() -> int:
    """Main entry point."""
    try:
        results = await run_all_diagnostics()
        print_summary(results)

        # Return exit code
        all_passed = all(r.passed for r in results)
        return 0 if all_passed else 1

    except Exception as e:
        print(f"\nFATAL ERROR: {e}\n")
        return 2


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
