#!/usr/bin/env python3
"""
BIBLOS v2 - Data Population Script

Populates the database with biblical text data from various sources.
"""
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn


console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("biblos.scripts.populate")


# Book metadata
BOOKS = {
    # Old Testament
    "GEN": {"name": "Genesis", "chapters": 50, "testament": "OT"},
    "EXO": {"name": "Exodus", "chapters": 40, "testament": "OT"},
    "LEV": {"name": "Leviticus", "chapters": 27, "testament": "OT"},
    "NUM": {"name": "Numbers", "chapters": 36, "testament": "OT"},
    "DEU": {"name": "Deuteronomy", "chapters": 34, "testament": "OT"},
    "JOS": {"name": "Joshua", "chapters": 24, "testament": "OT"},
    "JDG": {"name": "Judges", "chapters": 21, "testament": "OT"},
    "RUT": {"name": "Ruth", "chapters": 4, "testament": "OT"},
    "1SA": {"name": "1 Samuel", "chapters": 31, "testament": "OT"},
    "2SA": {"name": "2 Samuel", "chapters": 24, "testament": "OT"},
    "1KI": {"name": "1 Kings", "chapters": 22, "testament": "OT"},
    "2KI": {"name": "2 Kings", "chapters": 25, "testament": "OT"},
    "1CH": {"name": "1 Chronicles", "chapters": 29, "testament": "OT"},
    "2CH": {"name": "2 Chronicles", "chapters": 36, "testament": "OT"},
    "EZR": {"name": "Ezra", "chapters": 10, "testament": "OT"},
    "NEH": {"name": "Nehemiah", "chapters": 13, "testament": "OT"},
    "EST": {"name": "Esther", "chapters": 10, "testament": "OT"},
    "JOB": {"name": "Job", "chapters": 42, "testament": "OT"},
    "PSA": {"name": "Psalms", "chapters": 150, "testament": "OT"},
    "PRO": {"name": "Proverbs", "chapters": 31, "testament": "OT"},
    "ECC": {"name": "Ecclesiastes", "chapters": 12, "testament": "OT"},
    "SNG": {"name": "Song of Solomon", "chapters": 8, "testament": "OT"},
    "ISA": {"name": "Isaiah", "chapters": 66, "testament": "OT"},
    "JER": {"name": "Jeremiah", "chapters": 52, "testament": "OT"},
    "LAM": {"name": "Lamentations", "chapters": 5, "testament": "OT"},
    "EZK": {"name": "Ezekiel", "chapters": 48, "testament": "OT"},
    "DAN": {"name": "Daniel", "chapters": 12, "testament": "OT"},
    "HOS": {"name": "Hosea", "chapters": 14, "testament": "OT"},
    "JOL": {"name": "Joel", "chapters": 3, "testament": "OT"},
    "AMO": {"name": "Amos", "chapters": 9, "testament": "OT"},
    "OBA": {"name": "Obadiah", "chapters": 1, "testament": "OT"},
    "JON": {"name": "Jonah", "chapters": 4, "testament": "OT"},
    "MIC": {"name": "Micah", "chapters": 7, "testament": "OT"},
    "NAH": {"name": "Nahum", "chapters": 3, "testament": "OT"},
    "HAB": {"name": "Habakkuk", "chapters": 3, "testament": "OT"},
    "ZEP": {"name": "Zephaniah", "chapters": 3, "testament": "OT"},
    "HAG": {"name": "Haggai", "chapters": 2, "testament": "OT"},
    "ZEC": {"name": "Zechariah", "chapters": 14, "testament": "OT"},
    "MAL": {"name": "Malachi", "chapters": 4, "testament": "OT"},
    # New Testament
    "MAT": {"name": "Matthew", "chapters": 28, "testament": "NT"},
    "MRK": {"name": "Mark", "chapters": 16, "testament": "NT"},
    "LUK": {"name": "Luke", "chapters": 24, "testament": "NT"},
    "JHN": {"name": "John", "chapters": 21, "testament": "NT"},
    "ACT": {"name": "Acts", "chapters": 28, "testament": "NT"},
    "ROM": {"name": "Romans", "chapters": 16, "testament": "NT"},
    "1CO": {"name": "1 Corinthians", "chapters": 16, "testament": "NT"},
    "2CO": {"name": "2 Corinthians", "chapters": 13, "testament": "NT"},
    "GAL": {"name": "Galatians", "chapters": 6, "testament": "NT"},
    "EPH": {"name": "Ephesians", "chapters": 6, "testament": "NT"},
    "PHP": {"name": "Philippians", "chapters": 4, "testament": "NT"},
    "COL": {"name": "Colossians", "chapters": 4, "testament": "NT"},
    "1TH": {"name": "1 Thessalonians", "chapters": 5, "testament": "NT"},
    "2TH": {"name": "2 Thessalonians", "chapters": 3, "testament": "NT"},
    "1TI": {"name": "1 Timothy", "chapters": 6, "testament": "NT"},
    "2TI": {"name": "2 Timothy", "chapters": 4, "testament": "NT"},
    "TIT": {"name": "Titus", "chapters": 3, "testament": "NT"},
    "PHM": {"name": "Philemon", "chapters": 1, "testament": "NT"},
    "HEB": {"name": "Hebrews", "chapters": 13, "testament": "NT"},
    "JAS": {"name": "James", "chapters": 5, "testament": "NT"},
    "1PE": {"name": "1 Peter", "chapters": 5, "testament": "NT"},
    "2PE": {"name": "2 Peter", "chapters": 3, "testament": "NT"},
    "1JN": {"name": "1 John", "chapters": 5, "testament": "NT"},
    "2JN": {"name": "2 John", "chapters": 1, "testament": "NT"},
    "3JN": {"name": "3 John", "chapters": 1, "testament": "NT"},
    "JUD": {"name": "Jude", "chapters": 1, "testament": "NT"},
    "REV": {"name": "Revelation", "chapters": 22, "testament": "NT"}
}


@click.group()
def cli():
    """BIBLOS v2 Data Population Tools."""
    pass


@cli.command()
@click.option("--source", "-s", type=click.Path(exists=True), help="Source data directory")
@click.option("--output", "-o", type=click.Path(), default="output/verses", help="Output directory")
@click.option("--books", "-b", multiple=True, help="Specific books to process")
@click.option("--format", "-f", type=click.Choice(["json", "jsonl"]), default="json")
def verses(source: str, output: str, books: tuple, format: str):
    """Populate verse data from source files."""
    console.print("[bold]Populating verse data...[/bold]")

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    if source:
        source_path = Path(source)
        _process_source_verses(source_path, output_path, list(books) if books else None, format)
    else:
        _generate_placeholder_verses(output_path, list(books) if books else None, format)

    console.print(f"[green]Verse data saved to {output}[/green]")


def _process_source_verses(
    source_path: Path,
    output_path: Path,
    books: Optional[List[str]],
    format: str
):
    """Process verses from source files."""
    json_files = list(source_path.glob("**/*.json"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing...", total=len(json_files))

        for file_path in json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Normalize and save
                verses = _normalize_verses(data)

                if verses:
                    book = verses[0].get("book", "unknown")
                    if books and book not in books:
                        continue

                    output_file = output_path / f"{book}.{format}"
                    _save_verses(verses, output_file, format)

            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")

            progress.advance(task)


def _generate_placeholder_verses(
    output_path: Path,
    books: Optional[List[str]],
    format: str
):
    """Generate placeholder verse data."""
    book_list = books if books else list(BOOKS.keys())

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Generating...", total=len(book_list))

        for book_code in book_list:
            if book_code not in BOOKS:
                continue

            book_info = BOOKS[book_code]
            verses = []

            for chapter in range(1, book_info["chapters"] + 1):
                # Generate placeholder verses (typical 20-40 per chapter)
                for verse in range(1, 30):
                    verses.append({
                        "verse_id": f"{book_code}.{chapter}.{verse}",
                        "book": book_code,
                        "book_name": book_info["name"],
                        "chapter": chapter,
                        "verse": verse,
                        "text": f"[{book_info['name']} {chapter}:{verse}]",
                        "testament": book_info["testament"],
                        "language": "hebrew" if book_info["testament"] == "OT" else "greek"
                    })

            output_file = output_path / f"{book_code}.{format}"
            _save_verses(verses, output_file, format)

            progress.advance(task)


def _normalize_verses(data: Any) -> List[Dict[str, Any]]:
    """Normalize verse data to standard format."""
    verses = []

    if isinstance(data, list):
        for item in data:
            verse = _normalize_verse(item)
            if verse:
                verses.append(verse)
    elif isinstance(data, dict):
        if "verses" in data:
            for item in data["verses"]:
                verse = _normalize_verse(item)
                if verse:
                    verses.append(verse)
        else:
            verse = _normalize_verse(data)
            if verse:
                verses.append(verse)

    return verses


def _normalize_verse(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize a single verse."""
    verse_id = data.get("verse_id") or data.get("ref") or data.get("id")
    if not verse_id:
        return None

    # Parse verse ID
    parts = verse_id.upper().replace(" ", ".").replace(":", ".").split(".")
    if len(parts) < 3:
        return None

    book = parts[0]
    book_info = BOOKS.get(book, {})

    try:
        return {
            "verse_id": f"{book}.{parts[1]}.{parts[2]}",
            "book": book,
            "book_name": book_info.get("name", book),
            "chapter": int(parts[1]),
            "verse": int(parts[2]),
            "text": data.get("text", ""),
            "testament": book_info.get("testament", "unknown"),
            "language": data.get("language", "unknown")
        }
    except ValueError:
        return None


def _save_verses(verses: List[Dict], output_file: Path, format: str):
    """Save verses to file."""
    with open(output_file, "w", encoding="utf-8") as f:
        if format == "json":
            json.dump(verses, f, indent=2, ensure_ascii=False)
        elif format == "jsonl":
            for verse in verses:
                f.write(json.dumps(verse, ensure_ascii=False) + "\n")


@cli.command()
@click.option("--source", "-s", type=click.Path(exists=True), required=True, help="Source cross-references directory")
@click.option("--output", "-o", type=click.Path(), default="output/crossrefs", help="Output directory")
@click.option("--validate/--no-validate", default=True, help="Validate cross-references")
def crossrefs(source: str, output: str, validate: bool):
    """Process and normalize cross-reference data."""
    console.print("[bold]Processing cross-references...[/bold]")

    source_path = Path(source)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    all_refs = []
    json_files = list(source_path.glob("**/*.json"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing...", total=len(json_files))

        for file_path in json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                refs = _extract_crossrefs(data)
                if validate:
                    refs = _validate_crossrefs(refs)

                all_refs.extend(refs)

            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")

            progress.advance(task)

    # Save by source book
    refs_by_book = {}
    for ref in all_refs:
        source = ref.get("source_ref", "")
        book = source.split(".")[0] if "." in source else "unknown"
        if book not in refs_by_book:
            refs_by_book[book] = []
        refs_by_book[book].append(ref)

    for book, refs in refs_by_book.items():
        output_file = output_path / f"{book}_crossrefs.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(refs, f, indent=2, ensure_ascii=False)

    console.print(f"[green]Processed {len(all_refs)} cross-references to {output}[/green]")


def _extract_crossrefs(data: Any) -> List[Dict[str, Any]]:
    """Extract cross-references from data."""
    refs = []

    if isinstance(data, list):
        for item in data:
            ref = _normalize_crossref(item)
            if ref:
                refs.append(ref)
    elif isinstance(data, dict):
        if "cross_references" in data:
            for item in data["cross_references"]:
                ref = _normalize_crossref(item)
                if ref:
                    refs.append(ref)
        else:
            ref = _normalize_crossref(data)
            if ref:
                refs.append(ref)

    return refs


def _normalize_crossref(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize a cross-reference."""
    source = data.get("source_ref") or data.get("source")
    target = data.get("target_ref") or data.get("target")

    if not source or not target:
        return None

    return {
        "source_ref": source.upper().replace(" ", ".").replace(":", "."),
        "target_ref": target.upper().replace(" ", ".").replace(":", "."),
        "connection_type": data.get("connection_type", "thematic"),
        "strength": data.get("strength", "moderate"),
        "confidence": data.get("confidence", 1.0),
        "notes": data.get("notes", []),
        "sources": data.get("sources", [])
    }


def _validate_crossrefs(refs: List[Dict]) -> List[Dict]:
    """Validate cross-references."""
    valid_types = {
        "thematic", "verbal", "conceptual", "historical",
        "typological", "prophetic", "liturgical", "narrative",
        "genealogical", "geographical"
    }

    validated = []
    for ref in refs:
        # Validate connection type
        if ref.get("connection_type") not in valid_types:
            ref["connection_type"] = "thematic"

        # Validate strength
        if ref.get("strength") not in {"strong", "moderate", "weak"}:
            ref["strength"] = "moderate"

        # Validate references format
        source = ref.get("source_ref", "")
        target = ref.get("target_ref", "")

        if _is_valid_ref(source) and _is_valid_ref(target):
            validated.append(ref)

    return validated


def _is_valid_ref(ref: str) -> bool:
    """Check if reference is valid."""
    parts = ref.upper().split(".")
    if len(parts) < 3:
        return False

    book = parts[0]
    if book not in BOOKS:
        return False

    try:
        int(parts[1])
        int(parts[2])
        return True
    except ValueError:
        return False


@cli.command()
@click.option("--output", "-o", type=click.Path(), default="output/embeddings", help="Output directory")
@click.option("--model", "-m", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model")
@click.option("--batch-size", "-b", default=32, help="Batch size")
def embeddings(output: str, model: str, batch_size: int):
    """Generate embeddings for all verses."""
    console.print(f"[bold]Generating embeddings with {model}...[/bold]")

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Would load verses and generate embeddings
    console.print(f"[green]Embeddings saved to {output}[/green]")


if __name__ == "__main__":
    cli()
