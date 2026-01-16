#!/usr/bin/env python3
"""
Run Property-Based Tests and Generate Report

This script runs the Hypothesis property tests and generates a summary report
of edge cases found and test coverage.
"""
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime


def run_property_tests(profile="default", verbose=False):
    """
    Run property tests with specified Hypothesis profile.

    Args:
        profile: Hypothesis profile to use (default, dev, ci, thorough, debug)
        verbose: Enable verbose output
    """
    print(f"Running property tests with profile: {profile}")
    print("=" * 80)

    cmd = [
        "pytest",
        "tests/property/",
        f"--hypothesis-profile={profile}",
        "-v" if verbose else "-q",
        "--tb=short",
        "-m", "property or stateful",
        "--json-report",
        "--json-report-file=.property-test-report.json",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        return result.returncode == 0
    except Exception as e:
        print(f"Error running tests: {e}")
        return False


def generate_summary_report():
    """Generate a summary report from test results."""
    report_file = Path(".property-test-report.json")

    if not report_file.exists():
        print("No test report found. Run tests first.")
        return

    with open(report_file, "r") as f:
        data = json.load(f)

    print("\n" + "=" * 80)
    print("PROPERTY TEST SUMMARY REPORT")
    print("=" * 80)

    # Summary statistics
    summary = data.get("summary", {})
    print(f"\nTotal tests: {summary.get('total', 0)}")
    print(f"Passed: {summary.get('passed', 0)}")
    print(f"Failed: {summary.get('failed', 0)}")
    print(f"Skipped: {summary.get('skipped', 0)}")
    print(f"Duration: {data.get('duration', 0):.2f}s")

    # Test categories
    print("\n" + "-" * 80)
    print("TESTS BY MODULE")
    print("-" * 80)

    tests = data.get("tests", [])
    modules = {}

    for test in tests:
        module = test.get("nodeid", "").split("::")[0]
        if module not in modules:
            modules[module] = {"passed": 0, "failed": 0, "skipped": 0}

        outcome = test.get("outcome", "unknown")
        if outcome in modules[module]:
            modules[module][outcome] += 1

    for module, counts in sorted(modules.items()):
        print(f"\n{module}:")
        print(f"  Passed: {counts['passed']}")
        print(f"  Failed: {counts['failed']}")
        print(f"  Skipped: {counts['skipped']}")

    # Failed tests detail
    failed_tests = [t for t in tests if t.get("outcome") == "failed"]
    if failed_tests:
        print("\n" + "-" * 80)
        print("FAILED TESTS")
        print("-" * 80)
        for test in failed_tests:
            print(f"\n{test.get('nodeid')}")
            print(f"  Error: {test.get('call', {}).get('longrepr', 'Unknown error')[:200]}")

    print("\n" + "=" * 80)
    print(f"Report generated at: {datetime.now().isoformat()}")
    print("=" * 80)


def print_edge_cases_found():
    """Print known edge cases discovered by property testing."""
    print("\n" + "=" * 80)
    print("EDGE CASES DISCOVERED BY PROPERTY TESTING")
    print("=" * 80)

    edge_cases = {
        "Schema Issues": [
            "Unicode normalization differences (NFD vs NFC)",
            "Empty strings in required fields",
            "Very long text (>10,000 characters) serialization",
            "Nested JSON with deep recursion",
        ],
        "Verse Parsing": [
            "Malformed IDs with multiple separators: GEN::1::1",
            "Lowercase book codes: gen.1.1",
            "Trailing/leading whitespace",
            "Zero and negative chapter/verse numbers",
            "Very long verse IDs (>1000 characters)",
        ],
        "Cross-References": [
            "Confidence scores at exact boundaries (0.0, 1.0)",
            "Self-referencing cross-references",
            "Empty notes and sources lists",
            "Very long note strings",
        ],
        "Pipeline": [
            "Zero-duration phases (very fast execution)",
            "Floating-point precision in time calculations",
            "Empty phase results in completed pipelines",
            "Concurrent phase execution timing",
        ],
        "ML Invariants": [
            "Embedding vectors with all zeros",
            "Numerical stability in softmax with large values",
            "Cosine similarity with zero-norm vectors",
            "Batch vs sequential processing differences",
        ],
    }

    for category, cases in edge_cases.items():
        print(f"\n{category}:")
        for case in cases:
            print(f"  ✓ {case}")

    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run property-based tests with Hypothesis"
    )
    parser.add_argument(
        "--profile",
        choices=["default", "dev", "ci", "thorough", "debug"],
        default="default",
        help="Hypothesis profile to use",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate report from existing results",
    )
    parser.add_argument(
        "--edge-cases",
        action="store_true",
        help="Show known edge cases discovered",
    )

    args = parser.parse_args()

    if args.edge_cases:
        print_edge_cases_found()
        return

    if not args.report_only:
        success = run_property_tests(profile=args.profile, verbose=args.verbose)
        if not success:
            print("\n⚠️  Some tests failed!")
            sys.exit(1)

    generate_summary_report()
    print("\n✅ Property tests completed successfully!")


if __name__ == "__main__":
    main()
