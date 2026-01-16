#!/usr/bin/env python3
"""
BIBLOS v2 - Stream Pipeline Runner

Example script demonstrating how to run the event-driven stream pipeline.

Usage:
    # Start the full pipeline (orchestrator + all phase consumers)
    python scripts/run_stream_pipeline.py --mode full

    # Start only phase consumers (for horizontal scaling)
    python scripts/run_stream_pipeline.py --mode consumers

    # Start only the orchestrator
    python scripts/run_stream_pipeline.py --mode orchestrator

    # Process a single verse
    python scripts/run_stream_pipeline.py --mode single --verse "GEN.1.1" --text "In the beginning..."

    # Process a batch from file
    python scripts/run_stream_pipeline.py --mode batch --input verses.json

    # Run recovery/DLQ operations
    python scripts/run_stream_pipeline.py --mode recovery --action list-dlq
"""
import argparse
import asyncio
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import (
    # Event bus
    get_event_bus,
    shutdown_event_bus,
    StreamTopic,
    # Stream orchestrator
    StreamOrchestrator,
    StreamOrchestratorConfig,
    create_stream_orchestrator,
    # Stream consumers
    StreamConsumerConfig,
    StreamConsumerManager,
    start_all_phase_consumers,
    StreamPhaseFactory,
    # Recovery
    get_recovery_service,
    shutdown_recovery_service,
)
from config import get_config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("biblos.stream_runner")


# =============================================================================
# SIGNAL HANDLING
# =============================================================================

shutdown_event = asyncio.Event()


def handle_shutdown(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    shutdown_event.set()


# =============================================================================
# RUNNING MODES
# =============================================================================

async def run_full_pipeline():
    """Run the complete stream pipeline (orchestrator + consumers)."""
    logger.info("Starting full stream pipeline")

    orchestrator = None
    consumer_manager = None

    try:
        # Start phase consumers
        logger.info("Starting phase consumers...")
        consumer_manager = await start_all_phase_consumers()

        # Start orchestrator
        logger.info("Starting orchestrator...")
        orchestrator = await create_stream_orchestrator()
        await orchestrator.start()

        logger.info("Stream pipeline is running. Press Ctrl+C to stop.")

        # Wait for shutdown signal
        await shutdown_event.wait()

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Shutting down...")

        if orchestrator:
            await orchestrator.shutdown()

        if consumer_manager:
            await consumer_manager.shutdown()

        await shutdown_event_bus()

        logger.info("Shutdown complete")


async def run_orchestrator_only():
    """Run only the orchestrator (consumers run separately)."""
    logger.info("Starting orchestrator only")

    orchestrator = None

    try:
        orchestrator = await create_stream_orchestrator()
        await orchestrator.start()

        logger.info("Orchestrator is running. Press Ctrl+C to stop.")
        await shutdown_event.wait()

    finally:
        if orchestrator:
            await orchestrator.shutdown()
        await shutdown_event_bus()


async def run_consumers_only(phases: Optional[List[str]] = None):
    """Run only phase consumers (for horizontal scaling)."""
    logger.info(f"Starting phase consumers: {phases or 'all'}")

    consumer_manager = None

    try:
        consumer_manager = await start_all_phase_consumers(phases=phases)

        logger.info("Phase consumers are running. Press Ctrl+C to stop.")
        await shutdown_event.wait()

    finally:
        if consumer_manager:
            await consumer_manager.shutdown()
        await shutdown_event_bus()


async def process_single_verse(verse_id: str, text: str):
    """Process a single verse through the stream pipeline."""
    logger.info(f"Processing single verse: {verse_id}")

    orchestrator = None
    consumer_manager = None

    try:
        # Start consumers
        consumer_manager = await start_all_phase_consumers()

        # Start orchestrator
        orchestrator = await create_stream_orchestrator()
        await orchestrator.start()

        # Give time for startup
        await asyncio.sleep(1)

        # Ingest verse
        correlation_id = await orchestrator.ingest_verse(verse_id, text)
        logger.info(f"Ingested verse with correlation_id: {correlation_id}")

        # Wait for completion (poll status)
        timeout = 300  # 5 minutes
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            status = await orchestrator.get_verse_status(correlation_id)

            if status:
                if status["status"] == "completed":
                    logger.info(f"Verse completed successfully!")
                    logger.info(f"Phases executed: {status['completed_phases']}")
                    print(json.dumps(status, indent=2, default=str))
                    return

                elif status["status"] == "failed":
                    logger.error(f"Verse processing failed!")
                    logger.error(f"Errors: {status['errors']}")
                    return

            await asyncio.sleep(2)

        logger.error("Processing timed out")

    finally:
        if orchestrator:
            await orchestrator.shutdown()
        if consumer_manager:
            await consumer_manager.shutdown()
        await shutdown_event_bus()


async def process_batch(input_file: str, output_file: Optional[str] = None):
    """Process a batch of verses from a JSON file."""
    logger.info(f"Processing batch from: {input_file}")

    # Load verses
    with open(input_file, "r", encoding="utf-8") as f:
        verses = json.load(f)

    logger.info(f"Loaded {len(verses)} verses")

    orchestrator = None
    consumer_manager = None
    results = []

    try:
        # Start consumers
        consumer_manager = await start_all_phase_consumers()

        # Start orchestrator
        orchestrator = await create_stream_orchestrator()
        await orchestrator.start()

        await asyncio.sleep(1)

        # Ingest all verses
        correlation_ids = await orchestrator.ingest_batch(verses)
        logger.info(f"Ingested {len(correlation_ids)} verses")

        # Wait for all to complete
        timeout = 600  # 10 minutes
        start_time = asyncio.get_event_loop().time()
        completed = set()

        while len(completed) < len(correlation_ids):
            if asyncio.get_event_loop().time() - start_time > timeout:
                logger.warning("Batch processing timed out")
                break

            for cid in correlation_ids:
                if cid in completed:
                    continue

                status = await orchestrator.get_verse_status(cid)
                if status and status["status"] in ("completed", "failed"):
                    completed.add(cid)
                    results.append(status)
                    logger.info(
                        f"[{len(completed)}/{len(correlation_ids)}] "
                        f"{status['verse_id']}: {status['status']}"
                    )

            await asyncio.sleep(2)

        # Save results
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to: {output_file}")

        # Summary
        completed_count = sum(1 for r in results if r["status"] == "completed")
        failed_count = sum(1 for r in results if r["status"] == "failed")
        logger.info(f"Batch complete: {completed_count} completed, {failed_count} failed")

    finally:
        if orchestrator:
            await orchestrator.shutdown()
        if consumer_manager:
            await consumer_manager.shutdown()
        await shutdown_event_bus()


async def run_recovery_operations(action: str, **kwargs):
    """Run recovery/DLQ operations."""
    logger.info(f"Running recovery action: {action}")

    try:
        recovery = await get_recovery_service()

        if action == "list-dlq":
            entries = await recovery.get_dlq_entries(count=kwargs.get("count", 50))
            print(f"\nDead Letter Queue ({len(entries)} entries):")
            print("-" * 80)
            for entry in entries:
                print(f"  ID: {entry.message_id}")
                print(f"  Type: {entry.original_event_type}")
                print(f"  Error: {entry.error[:100]}")
                print(f"  Topic: {entry.source_topic}")
                print(f"  Time: {entry.timestamp}")
                print("-" * 40)

        elif action == "dlq-summary":
            summary = await recovery.get_dlq_summary()
            print("\nDLQ Summary:")
            print(json.dumps(summary, indent=2))

        elif action == "reprocess":
            message_id = kwargs.get("message_id")
            if not message_id:
                logger.error("Message ID required for reprocess")
                return
            result = await recovery.reprocess_dlq_entry(message_id)
            if result:
                print(f"Reprocessed: {message_id} -> {result}")
            else:
                print(f"Failed to reprocess: {message_id}")

        elif action == "reprocess-all":
            count = await recovery.reprocess_dlq_by_filter(max_count=kwargs.get("count", 10))
            print(f"Reprocessed {count} DLQ entries")

        elif action == "cleanup-stale":
            results = await recovery.cleanup_all_stale()
            print(f"Cleaned up stale messages: {results}")

        elif action == "list-checkpoints":
            checkpoints = await recovery.list_checkpoints()
            print(f"\nCheckpoints ({len(checkpoints)}):")
            for cp in checkpoints:
                print(f"  {cp.checkpoint_id}: {cp.last_message_id} ({cp.processed_count} processed)")

        elif action == "health":
            health = await recovery.health_check()
            print("\nRecovery Service Health:")
            print(json.dumps(health, indent=2))

        else:
            logger.error(f"Unknown action: {action}")

    finally:
        await shutdown_recovery_service()
        await shutdown_event_bus()


async def show_pipeline_status():
    """Show current pipeline status."""
    logger.info("Fetching pipeline status...")

    try:
        event_bus = await get_event_bus()
        health = await event_bus.health_check()

        print("\nEvent Bus Status:")
        print(json.dumps(health, indent=2))

        # Show stream info for key topics
        print("\nStream Information:")
        for topic in [StreamTopic.VERSE_INGESTED, StreamTopic.VERSE_COMPLETED, StreamTopic.DEAD_LETTER_QUEUE]:
            info = await event_bus.get_stream_info(topic)
            print(f"  {topic.value}: {info.get('length', 0)} messages")

    finally:
        await shutdown_event_bus()


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="BIBLOS v2 Stream Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--mode",
        choices=["full", "orchestrator", "consumers", "single", "batch", "recovery", "status"],
        default="full",
        help="Running mode",
    )

    # Single verse options
    parser.add_argument("--verse", help="Verse ID for single mode")
    parser.add_argument("--text", help="Verse text for single mode")

    # Batch options
    parser.add_argument("--input", help="Input JSON file for batch mode")
    parser.add_argument("--output", help="Output JSON file for batch results")

    # Consumer options
    parser.add_argument(
        "--phases",
        help="Comma-separated list of phases (default: all)",
    )

    # Recovery options
    parser.add_argument(
        "--action",
        choices=["list-dlq", "dlq-summary", "reprocess", "reprocess-all", "cleanup-stale", "list-checkpoints", "health"],
        default="dlq-summary",
        help="Recovery action",
    )
    parser.add_argument("--message-id", help="Message ID for reprocess action")
    parser.add_argument("--count", type=int, default=50, help="Count limit")

    # General options
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Run based on mode
    if args.mode == "full":
        await run_full_pipeline()

    elif args.mode == "orchestrator":
        await run_orchestrator_only()

    elif args.mode == "consumers":
        phases = args.phases.split(",") if args.phases else None
        await run_consumers_only(phases)

    elif args.mode == "single":
        if not args.verse or not args.text:
            logger.error("--verse and --text required for single mode")
            return
        await process_single_verse(args.verse, args.text)

    elif args.mode == "batch":
        if not args.input:
            logger.error("--input required for batch mode")
            return
        await process_batch(args.input, args.output)

    elif args.mode == "recovery":
        await run_recovery_operations(
            args.action,
            message_id=args.message_id,
            count=args.count,
        )

    elif args.mode == "status":
        await show_pipeline_status()


if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted")
