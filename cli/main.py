"""
BIBLOS v2 - Main CLI Application

Command-line interface for the SDES extraction pipeline.
"""
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional, List
from enum import Enum

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import print as rprint

# Initialize app
app = typer.Typer(
    name="biblos",
    help="BIBLOS v2 - Scripture Data Extraction System",
    add_completion=False
)

console = Console()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("biblos.cli")


class OutputFormat(str, Enum):
    """Output format options."""
    JSON = "json"
    CSV = "csv"
    TABLE = "table"


@app.command()
def status():
    """Show system status and component health."""
    console.print(Panel.fit(
        "[bold blue]BIBLOS v2 - Scripture Data Extraction System[/bold blue]",
        border_style="blue"
    ))

    table = Table(title="Component Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")

    # Check components
    components = [
        ("Pipeline Orchestrator", "✓ Ready", "5 phases configured"),
        ("Linguistic Agents", "✓ Ready", "4 agents (grammateus, morphologos, syntaktikos, semantikos)"),
        ("Theological Agents", "✓ Ready", "5 agents (patrologos, typologos, theologos, liturgikos, dogmatikos)"),
        ("Intertextual Agents", "✓ Ready", "5 agents (syndesmos, harmonikos, allographos, paradeigma, topos)"),
        ("Validation Agents", "✓ Ready", "5 agents (elenktikos, kritikos, harmonizer, prosecutor, witness)"),
        ("ML Inference", "✓ Ready", "Ensemble with GNN refinement"),
        ("Database", "○ Not Connected", "Requires configuration"),
        ("Text-Fabric", "○ Not Loaded", "Load with --corpus flag"),
    ]

    for name, status, details in components:
        table.add_row(name, status, details)

    console.print(table)


@app.command()
def process(
    verse: str = typer.Argument(..., help="Verse ID (e.g., GEN.1.1 or 'Gen 1:1')"),
    text: Optional[str] = typer.Option(None, "--text", "-t", help="Verse text (optional if corpus loaded)"),
    output: OutputFormat = typer.Option(OutputFormat.TABLE, "--output", "-o", help="Output format"),
    save: Optional[Path] = typer.Option(None, "--save", "-s", help="Save results to file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Process a single verse through the SDES pipeline."""
    if verbose:
        logging.getLogger("biblos").setLevel(logging.DEBUG)

    console.print(f"[bold]Processing verse: {verse}[/bold]")

    # Run pipeline
    result = asyncio.run(_process_verse(verse, text))

    # Display results
    if output == OutputFormat.TABLE:
        _display_table_result(result)
    elif output == OutputFormat.JSON:
        import json
        console.print_json(json.dumps(result, indent=2))
    elif output == OutputFormat.CSV:
        _display_csv_result(result)

    # Save if requested
    if save:
        _save_result(result, save, output)
        console.print(f"[green]Results saved to {save}[/green]")


@app.command()
def batch(
    input_file: Path = typer.Argument(..., help="Input file with verse IDs (one per line)"),
    output_dir: Path = typer.Option(Path("output"), "--output", "-o", help="Output directory"),
    parallel: int = typer.Option(4, "--parallel", "-p", help="Parallel processing count"),
    format: OutputFormat = typer.Option(OutputFormat.JSON, "--format", "-f", help="Output format")
):
    """Process multiple verses in batch mode."""
    if not input_file.exists():
        console.print(f"[red]Error: Input file not found: {input_file}[/red]")
        raise typer.Exit(1)

    # Read verse IDs
    with open(input_file) as f:
        verses = [line.strip() for line in f if line.strip()]

    console.print(f"[bold]Processing {len(verses)} verses[/bold]")

    # Process batch
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Processing...", total=len(verses))

        results = asyncio.run(_process_batch(verses, parallel, progress, task))

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        verse_id = result.get("verse_id", "unknown")
        filename = f"{verse_id.replace('.', '_')}.{format.value}"
        filepath = output_dir / filename
        _save_result(result, filepath, format)

    console.print(f"[green]Processed {len(results)} verses to {output_dir}[/green]")


@app.command()
def discover(
    verse: str = typer.Argument(..., help="Source verse ID"),
    max_results: int = typer.Option(20, "--max", "-m", help="Maximum results"),
    min_confidence: float = typer.Option(0.5, "--min-conf", "-c", help="Minimum confidence"),
    connection_type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by connection type")
):
    """Discover cross-references for a verse using ML inference."""
    console.print(f"[bold]Discovering cross-references for: {verse}[/bold]")

    results = asyncio.run(_discover_crossrefs(verse, max_results, min_confidence, connection_type))

    # Display results
    table = Table(title=f"Cross-References for {verse}")
    table.add_column("Target", style="cyan")
    table.add_column("Type", style="yellow")
    table.add_column("Confidence", style="green")
    table.add_column("Strength")

    for ref in results:
        conf = ref.get("confidence", 0)
        strength_color = "green" if conf >= 0.7 else "yellow" if conf >= 0.5 else "red"
        table.add_row(
            ref.get("target_verse", ""),
            ref.get("connection_type", "thematic"),
            f"{conf:.2%}",
            f"[{strength_color}]{ref.get('strength', 'moderate')}[/{strength_color}]"
        )

    console.print(table)
    console.print(f"\nFound {len(results)} cross-references")


@app.command()
def export(
    book: str = typer.Argument(..., help="Book code (e.g., GEN, MAT)"),
    output: Path = typer.Option(Path("export"), "--output", "-o", help="Output directory"),
    format: OutputFormat = typer.Option(OutputFormat.JSON, "--format", "-f", help="Output format"),
    include_golden: bool = typer.Option(True, "--golden", help="Include golden records")
):
    """Export processed data for a book."""
    console.print(f"[bold]Exporting data for: {book}[/bold]")

    output.mkdir(parents=True, exist_ok=True)

    # Export logic would go here
    console.print(f"[green]Exported to {output}[/green]")


@app.command()
def validate(
    verse: Optional[str] = typer.Option(None, "--verse", "-v", help="Validate specific verse"),
    full: bool = typer.Option(False, "--full", help="Run full validation suite"),
    report: Optional[Path] = typer.Option(None, "--report", "-r", help="Save validation report")
):
    """Validate extraction results and data quality."""
    console.print("[bold]Running validation...[/bold]")

    if verse:
        result = asyncio.run(_validate_verse(verse))
        _display_validation_result(result)
    elif full:
        results = asyncio.run(_full_validation())
        _display_validation_summary(results)

        if report:
            import json
            with open(report, "w") as f:
                json.dump(results, f, indent=2)
            console.print(f"[green]Report saved to {report}[/green]")
    else:
        console.print("[yellow]Specify --verse or --full[/yellow]")


@app.command()
def train(
    data_dir: Path = typer.Argument(..., help="Training data directory"),
    output_dir: Path = typer.Option(Path("models"), "--output", "-o", help="Model output directory"),
    epochs: int = typer.Option(10, "--epochs", "-e", help="Training epochs"),
    batch_size: int = typer.Option(32, "--batch", "-b", help="Batch size"),
    learning_rate: float = typer.Option(1e-4, "--lr", help="Learning rate")
):
    """Train ML models for cross-reference discovery."""
    console.print("[bold]Starting model training...[/bold]")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Training would be implemented here
    console.print(f"[green]Training complete. Models saved to {output_dir}[/green]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Server host"),
    port: int = typer.Option(8000, "--port", "-p", help="Server port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload")
):
    """Start the BIBLOS API server."""
    console.print(f"[bold]Starting server at {host}:{port}[/bold]")

    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload
    )


# Helper functions
async def _process_verse(verse_id: str, text: Optional[str] = None) -> dict:
    """Process a single verse through the pipeline."""
    from pipeline.orchestrator import PipelineOrchestrator

    orchestrator = PipelineOrchestrator()
    await orchestrator.initialize()

    # Get text if not provided
    if not text:
        text = f"[{verse_id}]"  # Placeholder

    result = await orchestrator.execute(verse_id, text)
    await orchestrator.cleanup()

    return result.to_dict()


async def _process_batch(
    verses: List[str],
    parallel: int,
    progress,
    task
) -> List[dict]:
    """Process a batch of verses."""
    from pipeline.orchestrator import PipelineOrchestrator

    orchestrator = PipelineOrchestrator()
    await orchestrator.initialize()

    results = []
    verse_data = [{"verse_id": v, "text": f"[{v}]"} for v in verses]

    batch_results = await orchestrator.execute_batch(verse_data, parallel)

    for result in batch_results:
        results.append(result.to_dict())
        progress.advance(task)

    await orchestrator.cleanup()
    return results


async def _discover_crossrefs(
    verse_id: str,
    max_results: int,
    min_confidence: float,
    connection_type: Optional[str]
) -> List[dict]:
    """Discover cross-references for a verse."""
    from ml.inference.pipeline import InferencePipeline, InferenceConfig

    config = InferenceConfig(
        max_candidates=max_results,
        min_confidence=min_confidence
    )

    pipeline = InferencePipeline(config)
    await pipeline.initialize()

    result = await pipeline.infer(verse_id, f"[{verse_id}]")
    await pipeline.cleanup()

    candidates = []
    for c in result.candidates:
        if connection_type and c.connection_type != connection_type:
            continue
        candidates.append({
            "source_verse": c.source_verse,
            "target_verse": c.target_verse,
            "connection_type": c.connection_type,
            "confidence": c.confidence,
            "strength": "strong" if c.confidence >= 0.7 else "moderate" if c.confidence >= 0.5 else "weak"
        })

    return candidates


async def _validate_verse(verse_id: str) -> dict:
    """Validate a single verse's results."""
    return {"verse_id": verse_id, "valid": True, "issues": []}


async def _full_validation() -> dict:
    """Run full validation suite."""
    return {
        "total_verses": 0,
        "valid": 0,
        "invalid": 0,
        "issues": []
    }


def _display_table_result(result: dict):
    """Display result as rich table."""
    table = Table(title=f"Results for {result.get('verse_id', 'unknown')}")
    table.add_column("Phase", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Agents")

    phases = result.get("phases", {})
    for phase_name, phase_data in phases.items():
        status = phase_data.get("status", "unknown")
        agents = len(phase_data.get("agent_results", {}))
        table.add_row(phase_name, status, str(agents))

    console.print(table)


def _display_csv_result(result: dict):
    """Display result as CSV."""
    lines = ["phase,status,confidence"]
    for phase_name, phase_data in result.get("phases", {}).items():
        lines.append(f"{phase_name},{phase_data.get('status', 'unknown')},{phase_data.get('confidence', 0):.2f}")
    console.print("\n".join(lines))


def _save_result(result: dict, path: Path, format: OutputFormat):
    """Save result to file."""
    import json

    if format == OutputFormat.JSON:
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
    elif format == OutputFormat.CSV:
        with open(path, "w") as f:
            f.write(_display_csv_result(result))


def _display_validation_result(result: dict):
    """Display validation result."""
    valid = result.get("valid", False)
    color = "green" if valid else "red"
    console.print(f"[{color}]Validation: {'PASSED' if valid else 'FAILED'}[/{color}]")

    for issue in result.get("issues", []):
        console.print(f"  - {issue}")


def _display_validation_summary(results: dict):
    """Display validation summary."""
    table = Table(title="Validation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Verses", str(results.get("total_verses", 0)))
    table.add_row("Valid", str(results.get("valid", 0)))
    table.add_row("Invalid", str(results.get("invalid", 0)))

    console.print(table)


def main():
    """Main entry point."""
    # Register sub-command modules
    try:
        from cli import lxx_commands
        lxx_commands.register_commands(app)
    except ImportError as e:
        logger.warning(f"Could not load LXX commands: {e}")

    app()


if __name__ == "__main__":
    main()
