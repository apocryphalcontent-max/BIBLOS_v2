#!/usr/bin/env python3
"""
BIBLOS v2 - Patristic Text Population Script

Ingests, cleans, and normalizes patristic texts using the scraper agents.
Integrates with the Polars loader pipeline for high-performance data access.

Features:
- Smart garbage filtering
- Author detection
- Scripture reference extraction
- Quality assessment
- Deduplication
- Polars DataFrame output
"""
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

# Local imports
try:
    from agents.scrapers import (
        PatristicScraperAgent,
        TextCleanerAgent,
        ScraperConfig,
        ContentQuality,
    )
    from data.loaders import create_polars_patristic_loader
    from data.schemas import PatristicTextSchema
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from agents.scrapers import (
        PatristicScraperAgent,
        TextCleanerAgent,
        ScraperConfig,
        ContentQuality,
    )
    from data.loaders import create_polars_patristic_loader
    from data.schemas import PatristicTextSchema


console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("biblos.scripts.populate_patristics")


@click.group()
def cli():
    """BIBLOS v2 Patristic Text Population Tools."""
    pass


@cli.command()
@click.option("--source", "-s", type=click.Path(exists=True), required=True,
              help="Source directory containing patristic texts")
@click.option("--output", "-o", type=click.Path(), default="data/patristics",
              help="Output directory for processed data")
@click.option("--format", "-f", type=click.Choice(["json", "jsonl", "parquet"]), default="json",
              help="Output format")
@click.option("--min-quality", "-q", type=float, default=0.4,
              help="Minimum quality score (0.0-1.0)")
@click.option("--deduplicate/--no-deduplicate", default=True,
              help="Remove duplicate content")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def ingest(source: str, output: str, format: str, min_quality: float,
           deduplicate: bool, verbose: bool):
    """
    Ingest patristic texts from source directory.

    Scans the source directory for text files, processes them through
    the PatristicScraperAgent with garbage filtering, and outputs
    normalized JSON/Parquet files ready for the Polars loader.
    """
    console.print("[bold blue]BIBLOS v2 - Patristic Text Ingestion[/bold blue]\n")

    source_path = Path(source)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Run async ingestion
    stats = asyncio.run(_ingest_patristics(
        source_path, output_path, format, min_quality, deduplicate, verbose
    ))

    # Display results
    _display_stats(stats)


async def _ingest_patristics(
    source_path: Path,
    output_path: Path,
    output_format: str,
    min_quality: float,
    deduplicate: bool,
    verbose: bool
) -> Dict[str, Any]:
    """Async ingestion pipeline."""

    # Configure scraper
    config = ScraperConfig(
        name="patristic_ingestion",
        source_type="patristic",
        batch_size=50,
        min_quality_score=min_quality,
        deduplicate=deduplicate,
        output_format=output_format,
    )

    scraper = PatristicScraperAgent(config)
    await scraper.initialize()

    # Get source files
    files = scraper.get_source_files(source_path)
    total_files = len(files)

    console.print(f"Found [cyan]{total_files}[/cyan] text files in {source_path}\n")

    if total_files == 0:
        console.print("[yellow]No files found to process[/yellow]")
        return {"total_files": 0}

    # Process with progress bar
    results = []
    quality_counts = {q: 0 for q in ContentQuality}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing files...", total=total_files)

        for file_path in files:
            result = await scraper.process(str(file_path))
            results.append(result)

            quality_counts[result.quality] += 1

            if verbose and result.status.value != "completed":
                console.print(f"  [dim]{file_path.name}: {result.status.value}[/dim]")

            progress.advance(task)

    # Convert usable results to schemas
    schemas = []
    for result in results:
        if result.is_usable():
            schema = await scraper.convert_to_schema(result)
            if schema:
                schemas.append(schema)

    console.print(f"\nConverted [green]{len(schemas)}[/green] texts to schemas")

    # Save output
    output_file = output_path / f"patristic_texts.{output_format}"

    if output_format == "json":
        _save_json(schemas, output_file)
    elif output_format == "jsonl":
        _save_jsonl(schemas, output_file)
    elif output_format == "parquet":
        _save_parquet(schemas, output_file)

    # Create index file
    index_file = output_path / "patristic_index.json"
    _create_index(schemas, index_file)

    await scraper.shutdown()

    return {
        "total_files": total_files,
        "processed": len(results),
        "usable": len(schemas),
        "quality_counts": {q.value: c for q, c in quality_counts.items()},
        "output_file": str(output_file),
        "index_file": str(index_file),
    }


def _save_json(schemas: List[PatristicTextSchema], output_file: Path) -> None:
    """Save as JSON array."""
    data = [s.to_dict() for s in schemas]
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    console.print(f"Saved to [cyan]{output_file}[/cyan]")


def _save_jsonl(schemas: List[PatristicTextSchema], output_file: Path) -> None:
    """Save as JSON Lines."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for schema in schemas:
            f.write(json.dumps(schema.to_dict(), ensure_ascii=False, default=str) + '\n')
    console.print(f"Saved to [cyan]{output_file}[/cyan]")


def _save_parquet(schemas: List[PatristicTextSchema], output_file: Path) -> None:
    """Save as Parquet using Polars."""
    try:
        import polars as pl
        from data.polars_schemas import PATRISTIC_TEXT

        data = [s.to_dict() for s in schemas]
        df = pl.DataFrame(data)
        df.write_parquet(output_file)
        console.print(f"Saved to [cyan]{output_file}[/cyan]")
    except ImportError:
        console.print("[yellow]Polars not available, falling back to JSON[/yellow]")
        json_file = output_file.with_suffix('.json')
        _save_json(schemas, json_file)


def _create_index(schemas: List[PatristicTextSchema], index_file: Path) -> None:
    """Create index file for quick lookups."""
    index = {
        "total_texts": len(schemas),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "authors": {},
        "themes": {},
    }

    # Index by author
    for schema in schemas:
        author = schema.author
        if author not in index["authors"]:
            index["authors"][author] = {"count": 0, "text_ids": []}
        index["authors"][author]["count"] += 1
        index["authors"][author]["text_ids"].append(schema.text_id)

        # Index by theme
        for theme in schema.themes:
            if theme not in index["themes"]:
                index["themes"][theme] = {"count": 0, "text_ids": []}
            index["themes"][theme]["count"] += 1
            index["themes"][theme]["text_ids"].append(schema.text_id)

    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    console.print(f"Created index at [cyan]{index_file}[/cyan]")


def _display_stats(stats: Dict[str, Any]) -> None:
    """Display processing statistics."""
    console.print("\n[bold]Processing Statistics[/bold]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Total Files", str(stats.get("total_files", 0)))
    table.add_row("Processed", str(stats.get("processed", 0)))
    table.add_row("Usable Texts", str(stats.get("usable", 0)))

    quality_counts = stats.get("quality_counts", {})
    for quality, count in quality_counts.items():
        if count > 0:
            table.add_row(f"Quality: {quality}", str(count))

    console.print(table)

    if "output_file" in stats:
        console.print(f"\n[green]Output:[/green] {stats['output_file']}")


@cli.command()
@click.option("--data-dir", "-d", type=click.Path(exists=True), default="data/patristics",
              help="Directory containing processed patristic data")
@click.option("--author", "-a", type=str, help="Filter by author name")
@click.option("--verse", "-v", type=str, help="Filter by verse reference")
@click.option("--search", "-s", type=str, help="Search content")
@click.option("--limit", "-l", type=int, default=10, help="Maximum results")
def query(data_dir: str, author: Optional[str], verse: Optional[str],
          search: Optional[str], limit: int):
    """
    Query processed patristic texts.

    Uses the Polars loader for high-performance queries against
    the processed patristic data.
    """
    console.print("[bold blue]BIBLOS v2 - Patristic Text Query[/bold blue]\n")

    try:
        import polars as pl

        loader = create_polars_patristic_loader(data_dir)
        df = loader.scan().collect()

        console.print(f"Loaded [cyan]{len(df)}[/cyan] texts\n")

        if author:
            df = loader.get_by_author(author)
            console.print(f"Filtered by author '{author}': [cyan]{len(df)}[/cyan] texts")

        if verse:
            df = loader.get_by_verse(verse)
            console.print(f"Filtered by verse '{verse}': [cyan]{len(df)}[/cyan] texts")

        if search:
            df = loader.search_content(search)
            console.print(f"Search for '{search}': [cyan]{len(df)}[/cyan] texts")

        # Display results
        if len(df) > 0:
            display_df = df.head(limit)

            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("ID", style="cyan", max_width=20)
            table.add_column("Author", max_width=25)
            table.add_column("Title", max_width=30)
            table.add_column("Content Preview", max_width=50)

            for row in display_df.iter_rows(named=True):
                content = row.get("content_clean", "")[:100] + "..."
                table.add_row(
                    row.get("text_id", ""),
                    row.get("author", ""),
                    row.get("title", ""),
                    content
                )

            console.print(table)
        else:
            console.print("[yellow]No results found[/yellow]")

    except ImportError:
        console.print("[red]Polars not installed. Install with: pip install polars[/red]")


@cli.command()
@click.option("--data-dir", "-d", type=click.Path(exists=True), default="data/patristics",
              help="Directory containing processed patristic data")
def stats(data_dir: str):
    """Display statistics about processed patristic data."""
    console.print("[bold blue]BIBLOS v2 - Patristic Data Statistics[/bold blue]\n")

    data_path = Path(data_dir)

    # Check for index file
    index_file = data_path / "patristic_index.json"
    if index_file.exists():
        with open(index_file, 'r', encoding='utf-8') as f:
            index = json.load(f)

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Total Texts", str(index.get("total_texts", 0)))
        table.add_row("Authors", str(len(index.get("authors", {}))))
        table.add_row("Themes", str(len(index.get("themes", {}))))
        table.add_row("Created", index.get("created_at", "Unknown"))

        console.print(table)

        # Author breakdown
        console.print("\n[bold]Authors:[/bold]")
        author_table = Table(show_header=True, header_style="bold cyan")
        author_table.add_column("Author")
        author_table.add_column("Texts", justify="right")

        authors = index.get("authors", {})
        for author, info in sorted(authors.items(), key=lambda x: -x[1]["count"])[:10]:
            author_table.add_row(author, str(info["count"]))

        console.print(author_table)

        # Theme breakdown
        console.print("\n[bold]Themes:[/bold]")
        theme_table = Table(show_header=True, header_style="bold cyan")
        theme_table.add_column("Theme")
        theme_table.add_column("Texts", justify="right")

        themes = index.get("themes", {})
        for theme, info in sorted(themes.items(), key=lambda x: -x[1]["count"])[:10]:
            theme_table.add_row(theme, str(info["count"]))

        console.print(theme_table)

    else:
        console.print("[yellow]No index file found. Run 'ingest' first.[/yellow]")


@cli.command()
@click.option("--source", "-s", type=click.Path(exists=True), required=True,
              help="Source directory or file to clean")
@click.option("--output", "-o", type=click.Path(),
              help="Output path (defaults to source with .clean suffix)")
def clean(source: str, output: Optional[str]):
    """
    Clean and normalize text files.

    Applies the TextCleanerAgent to fix encoding issues,
    normalize whitespace, and improve text quality.
    """
    console.print("[bold blue]BIBLOS v2 - Text Cleaning[/bold blue]\n")

    asyncio.run(_clean_texts(Path(source), Path(output) if output else None))


async def _clean_texts(source_path: Path, output_path: Optional[Path]) -> None:
    """Clean text files."""
    cleaner = TextCleanerAgent()
    await cleaner.initialize()

    if source_path.is_file():
        files = [source_path]
    else:
        files = cleaner.get_source_files(source_path)

    console.print(f"Cleaning [cyan]{len(files)}[/cyan] files\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Cleaning...", total=len(files))

        for file_path in files:
            result = await cleaner.process(str(file_path))

            if result.is_usable():
                # Determine output path
                if output_path:
                    if output_path.is_dir():
                        out_file = output_path / file_path.name
                    else:
                        out_file = output_path
                else:
                    out_file = file_path.with_suffix('.clean' + file_path.suffix)

                with open(out_file, 'w', encoding='utf-8') as f:
                    f.write(result.content_clean)

            progress.advance(task)

    await cleaner.shutdown()
    console.print("\n[green]Cleaning complete![/green]")


if __name__ == "__main__":
    cli()
