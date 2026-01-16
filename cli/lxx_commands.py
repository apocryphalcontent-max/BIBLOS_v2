"""
BIBLOS v2 - LXX Christological Analysis CLI Commands

CLI commands for the LXX Christological Extractor (Third Impossible Oracle).
Provides analysis, comparison, and scanning capabilities for LXX-MT divergences.
"""
import asyncio
import json
import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from ml.engines.lxx_extractor import (
    LXXChristologicalExtractor,
    ChristologicalCategory
)
from config import get_config
from db.neo4j_client import Neo4jClient
from db.redis_client import RedisClient
from integrations.lxx_corpus import LXXCorpusClient
from integrations.text_fabric import TextFabricClient

app = typer.Typer(help="LXX Christological analysis commands")
console = Console()


def get_extractor() -> LXXChristologicalExtractor:
    """Create configured extractor instance."""
    config = get_config()

    # Get corpus paths from config or use defaults
    lxx_corpus_path = config.lxx_extractor.lxx_corpus_path
    mt_corpus_path = config.lxx_extractor.mt_corpus_path

    return LXXChristologicalExtractor(
        lxx_client=LXXCorpusClient(lxx_corpus_path),
        mt_client=TextFabricClient(mt_corpus_path),
        neo4j=Neo4jClient(config.database.neo4j_uri),
        redis=RedisClient(config.database.redis_url),
        config=config.lxx_extractor
    )


@app.command("analyze")
def analyze_verse(
    verse: str = typer.Argument(..., help="Verse ID (e.g., ISA.7.14)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force recompute"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output")
):
    """Analyze a single verse for LXX Christological content."""

    async def _run():
        extractor = get_extractor()
        await extractor.lxx.initialize()
        await extractor.mt.initialize()

        with console.status(f"Analyzing {verse}..."):
            result = await extractor.extract_christological_content(verse, force)

        # Display results
        if result.christological_divergence_count > 0:
            console.print(f"\n[bold green]✓ Christological Content Found[/]")
            console.print(f"  Category: [cyan]{result.christological_category.value}[/]")
            console.print(f"  Insight: {result.primary_christological_insight}")
            console.print(f"  Significance: [yellow]{result.overall_significance:.2f}[/]")
            console.print(f"  NT Support: [yellow]{result.nt_support_strength:.0%}[/]")
            console.print(f"  Patristic Unanimity: [yellow]{result.patristic_unanimity:.0%}[/]")

            if verbose:
                # Divergences table
                table = Table(title="Divergences")
                table.add_column("Type")
                table.add_column("MT (Hebrew)")
                table.add_column("LXX (Greek)")
                table.add_column("Score")

                for div in result.divergences:
                    table.add_row(
                        div.divergence_type.value,
                        div.mt_text_hebrew[:30] + "..." if len(div.mt_text_hebrew) > 30 else div.mt_text_hebrew,
                        div.lxx_text_greek[:30] + "..." if len(div.lxx_text_greek) > 30 else div.lxx_text_greek,
                        f"{div.composite_score:.2f}"
                    )
                console.print(table)

                # NT Quotations
                if any(d.nt_quotations for d in result.divergences):
                    console.print("\n[bold]NT Quotations:[/]")
                    for div in result.divergences:
                        for q in div.nt_quotations:
                            pref = "LXX" if q.follows_lxx else "MT" if q.follows_mt else "neither"
                            console.print(f"  • {q.nt_verse} follows [cyan]{pref}[/] (LXX: {q.verbal_agreement_lxx:.0%}, MT: {q.verbal_agreement_mt:.0%})")

                # Manuscript evidence
                if any(d.manuscript_witnesses for d in result.divergences):
                    console.print("\n[bold]Manuscript Evidence:[/]")
                    for div in result.divergences:
                        if div.oldest_witness:
                            console.print(f"  Oldest: {div.oldest_witness.manuscript_id} ({div.oldest_witness.date_range})")
                            console.print(f"  Supports: [cyan]{div.oldest_supports}[/]")
        else:
            console.print(f"\n[dim]No significant Christological divergence in {verse}[/]")

        await extractor.lxx.cleanup()
        await extractor.mt.cleanup()

    asyncio.run(_run())


@app.command("scan")
def scan_book(
    book: str = typer.Argument(..., help="Book code (e.g., ISA, PSA)"),
    min_score: float = typer.Option(0.5, "--min-score", "-m", help="Minimum significance"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON file")
):
    """Scan an entire book for Christological LXX divergences."""

    async def _run():
        extractor = get_extractor()
        await extractor.lxx.initialize()
        await extractor.mt.initialize()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Scanning {book}...", total=None)

            results = await extractor.scan_book_for_christological(book, min_score)

        console.print(f"\n[bold]Found {len(results)} significant verses in {book}[/]")

        # Display table
        table = Table(title=f"Christological LXX Divergences in {book}")
        table.add_column("Verse")
        table.add_column("Category")
        table.add_column("Score")
        table.add_column("NT Support")
        table.add_column("Patristic")

        for r in results[:20]:  # Show top 20
            table.add_row(
                r.verse_id,
                r.christological_category.value if r.christological_category else "-",
                f"{r.overall_significance:.2f}",
                "✓" if r.nt_support_strength > 0.5 else "○",
                "✓" if r.patristic_unanimity > 0.5 else "○"
            )

        console.print(table)

        if len(results) > 20:
            console.print(f"\n[dim]Showing top 20 of {len(results)} results[/]")

        # Save to file if requested
        if output:
            output_data = [
                {
                    "verse_id": r.verse_id,
                    "category": r.christological_category.value if r.christological_category else None,
                    "significance": r.overall_significance,
                    "nt_support": r.nt_support_strength,
                    "patristic_unanimity": r.patristic_unanimity,
                    "insight": r.primary_christological_insight,
                    "divergence_count": r.christological_divergence_count
                }
                for r in results
            ]
            with open(output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            console.print(f"\n[green]Results saved to {output}[/]")

        await extractor.lxx.cleanup()
        await extractor.mt.cleanup()

    asyncio.run(_run())


@app.command("known")
def list_known():
    """List all known Christological LXX divergences."""
    console.print(Panel.fit(
        "[bold cyan]Known Christological LXX Divergences[/bold cyan]",
        border_style="cyan"
    ))

    table = Table(title="Catalog of Known Christological Verses")
    table.add_column("Verse", style="cyan")
    table.add_column("Category", style="green")
    table.add_column("Description")

    descriptions = {
        "ISA.7.14": "almah → parthenos (virgin)",
        "PSA.40.6": "ears → body prepared",
        "GEN.3.15": "it/she → he (seed promise)",
        "PSA.22.16": "like a lion → they pierced",
        "ISA.53.8": "Generation declaration",
        "PSA.16.10": "Holy One not see corruption",
        "ISA.9.6": "Wonderful Counselor titles",
        "MIC.5.2": "Origin from eternity",
        "ZEC.12.10": "Look on pierced one",
        "PSA.110.1": "Lord says to my Lord",
        "DAN.7.13": "Son of Man coming",
        "ISA.61.1": "Anointed to preach",
        "MAL.3.1": "Messenger of the covenant",
        "PSA.2.7": "You are my Son",
        "ISA.11.1": "Branch from Jesse",
    }

    for verse, category in LXXChristologicalExtractor.KNOWN_CHRISTOLOGICAL_VERSES.items():
        table.add_row(
            verse,
            category.value,
            descriptions.get(verse, "")
        )

    console.print(table)
    console.print(f"\n[dim]{len(LXXChristologicalExtractor.KNOWN_CHRISTOLOGICAL_VERSES)} known verses cataloged[/]")


@app.command("compare")
def compare_texts(
    verse: str = typer.Argument(..., help="Verse ID to compare")
):
    """Display MT and LXX texts side by side."""

    async def _run():
        extractor = get_extractor()
        await extractor.lxx.initialize()
        await extractor.mt.initialize()

        # Get MT and LXX verse IDs (handle numbering differences)
        mt_verse_id, lxx_verse_id = await extractor._normalize_verse_ids(verse)

        mt_data = await extractor.mt.get_verse(mt_verse_id)
        lxx_data = await extractor.lxx.get_verse(lxx_verse_id)

        console.print(f"\n[bold]Text Comparison: {verse}[/]\n")

        if mt_verse_id != lxx_verse_id:
            console.print(f"[dim]MT: {mt_verse_id} | LXX: {lxx_verse_id}[/]\n")

        console.print("[cyan]Masoretic Text (Hebrew):[/]")
        console.print(f"  {mt_data.get('text', 'Not found')}")

        console.print("\n[green]Septuagint (Greek):[/]")
        console.print(f"  {lxx_data.get('text', 'Not found')}")

        # Word-by-word comparison
        if mt_data.get("words") and lxx_data.get("words"):
            console.print("\n[bold]Word Comparison:[/]")
            table = Table()
            table.add_column("MT Word")
            table.add_column("MT Gloss")
            table.add_column("LXX Word")
            table.add_column("LXX Gloss")

            mt_words = mt_data["words"]
            lxx_words = lxx_data["words"]

            max_len = max(len(mt_words), len(lxx_words))
            for i in range(max_len):
                mt_w = mt_words[i] if i < len(mt_words) else {}
                lxx_w = lxx_words[i] if i < len(lxx_words) else {}
                table.add_row(
                    mt_w.get("text", "-"),
                    mt_w.get("gloss", "-"),
                    lxx_w.get("text", "-"),
                    lxx_w.get("gloss", "-")
                )

            console.print(table)

        await extractor.lxx.cleanup()
        await extractor.mt.cleanup()

    asyncio.run(_run())


@app.command("search")
def search_greek(
    pattern: str = typer.Argument(..., help="Greek text pattern to search"),
    book: Optional[str] = typer.Option(None, "--book", "-b", help="Limit to book (e.g., ISA)"),
    limit: int = typer.Option(50, "--limit", "-l", help="Maximum results")
):
    """Search for Greek text patterns in the LXX."""

    async def _run():
        config = get_config()
        lxx = LXXCorpusClient(config.lxx_extractor.lxx_corpus_path)
        await lxx.initialize()

        console.print(f"[bold]Searching LXX for: {pattern}[/]")
        if book:
            console.print(f"  Limited to book: {book}")

        results = await lxx.search_greek(pattern, book, limit)

        table = Table(title=f"Search Results for '{pattern}'")
        table.add_column("Verse", style="cyan")
        table.add_column("Text")

        for result in results:
            verse_id = result.get("verse_id", "")
            text = result.get("text", "")
            # Highlight the pattern in the text
            highlighted = text.replace(pattern, f"[yellow]{pattern}[/yellow]")
            table.add_row(verse_id, highlighted)

        console.print(table)
        console.print(f"\nFound {len(results)} results")

        await lxx.cleanup()

    asyncio.run(_run())


def register_commands(main_app: typer.Typer):
    """Register LXX commands with main CLI."""
    main_app.add_typer(app, name="lxx", help="LXX Christological analysis")
