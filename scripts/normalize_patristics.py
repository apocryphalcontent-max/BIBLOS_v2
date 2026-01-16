"""
BIBLOS v2 - Patristic Text Normalizer

Converts raw patristic text files to normalized JSON format for ingestion
into the BIBLOS data pipeline.

Features:
- Intelligent parsing of book/chapter/section structure
- Garbage content filtering (URLs, metadata, empty sections)
- Scripture reference extraction
- Footnote separation
- Author and work metadata extraction from filenames

Usage:
    python scripts/normalize_patristics.py --input /path/to/out_text --output /path/to/output
"""

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.schemas import PatristicTextSchema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# GARBAGE DETECTION PATTERNS
# =============================================================================

GARBAGE_PATTERNS = [
    r"file:///",
    r"https?://",
    r"\.htm[l]?\d*$",
    r"\.pdf$",
    r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}",  # Timestamps
    r"^\s*$",  # Empty lines
    r"^\s*n\s*$",  # Just "n" (navigation)
    r"^Index\s*$",
    r"^Table of Contents",
    r"^General Index",
    r"^\s*\(\d+ of \d+\)",  # Page numbers
]

GARBAGE_COMPILED = [re.compile(p, re.IGNORECASE) for p in GARBAGE_PATTERNS]


# =============================================================================
# SCRIPTURE REFERENCE PATTERNS
# =============================================================================

# Book abbreviations
BOOK_ABBREVS = {
    "Gen": "GEN", "Genesis": "GEN",
    "Exod": "EXO", "Exodus": "EXO", "Ex": "EXO",
    "Lev": "LEV", "Leviticus": "LEV",
    "Num": "NUM", "Numbers": "NUM",
    "Deut": "DEU", "Deuteronomy": "DEU",
    "Josh": "JOS", "Joshua": "JOS",
    "Judg": "JDG", "Judges": "JDG",
    "Ruth": "RUT",
    "1 Sam": "1SA", "1 Samuel": "1SA", "I Sam": "1SA",
    "2 Sam": "2SA", "2 Samuel": "2SA", "II Sam": "2SA",
    "1 Kings": "1KI", "I Kings": "1KI", "1 Kgs": "1KI",
    "2 Kings": "2KI", "II Kings": "2KI", "2 Kgs": "2KI",
    "1 Chron": "1CH", "I Chronicles": "1CH", "1 Chronicles": "1CH",
    "2 Chron": "2CH", "II Chronicles": "2CH", "2 Chronicles": "2CH",
    "Ezra": "EZR",
    "Neh": "NEH", "Nehemiah": "NEH",
    "Esth": "EST", "Esther": "EST",
    "Job": "JOB",
    "Ps": "PSA", "Psalm": "PSA", "Psalms": "PSA",
    "Prov": "PRO", "Proverbs": "PRO",
    "Eccl": "ECC", "Ecclesiastes": "ECC",
    "Song": "SNG", "Songs": "SNG", "Canticles": "SNG", "Song of Solomon": "SNG",
    "Isa": "ISA", "Isaiah": "ISA",
    "Jer": "JER", "Jeremiah": "JER",
    "Lam": "LAM", "Lamentations": "LAM",
    "Ezek": "EZK", "Ezekiel": "EZK",
    "Dan": "DAN", "Daniel": "DAN",
    "Hos": "HOS", "Hosea": "HOS",
    "Joel": "JOL",
    "Amos": "AMO",
    "Obad": "OBA", "Obadiah": "OBA",
    "Jonah": "JON",
    "Mic": "MIC", "Micah": "MIC",
    "Nah": "NAM", "Nahum": "NAM",
    "Hab": "HAB", "Habakkuk": "HAB",
    "Zeph": "ZEP", "Zephaniah": "ZEP",
    "Hag": "HAG", "Haggai": "HAG",
    "Zech": "ZEC", "Zechariah": "ZEC",
    "Mal": "MAL", "Malachi": "MAL",
    # New Testament
    "Matt": "MAT", "Matthew": "MAT", "Mt": "MAT",
    "Mark": "MRK", "Mk": "MRK",
    "Luke": "LUK", "Lk": "LUK",
    "John": "JHN", "Jn": "JHN", "Joh": "JHN",
    "Acts": "ACT",
    "Rom": "ROM", "Romans": "ROM",
    "1 Cor": "1CO", "1 Corinthians": "1CO", "I Cor": "1CO",
    "2 Cor": "2CO", "2 Corinthians": "2CO", "II Cor": "2CO",
    "Gal": "GAL", "Galatians": "GAL",
    "Eph": "EPH", "Ephesians": "EPH",
    "Phil": "PHP", "Philippians": "PHP",
    "Col": "COL", "Colossians": "COL",
    "1 Thess": "1TH", "1 Thessalonians": "1TH", "I Thess": "1TH",
    "2 Thess": "2TH", "2 Thessalonians": "2TH", "II Thess": "2TH",
    "1 Tim": "1TI", "1 Timothy": "1TI", "I Tim": "1TI",
    "2 Tim": "2TI", "2 Timothy": "2TI", "II Tim": "2TI",
    "Titus": "TIT",
    "Philem": "PHM", "Philemon": "PHM",
    "Heb": "HEB", "Hebrews": "HEB",
    "James": "JAS", "Jas": "JAS",
    "1 Pet": "1PE", "1 Peter": "1PE", "I Pet": "1PE",
    "2 Pet": "2PE", "2 Peter": "2PE", "II Pet": "2PE",
    "1 John": "1JN", "I John": "1JN", "1 Jn": "1JN",
    "2 John": "2JN", "II John": "2JN", "2 Jn": "2JN",
    "3 John": "3JN", "III John": "3JN", "3 Jn": "3JN",
    "Jude": "JUD",
    "Rev": "REV", "Revelation": "REV", "Apoc": "REV", "Apocalypse": "REV",
}

# Pattern to match scripture references like "Romans ii, 11" or "Gen 1:1"
SCRIPTURE_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in sorted(BOOK_ABBREVS.keys(), key=len, reverse=True)) +
    r')[\s\.]*(\d+)[,\s:\.]+(\d+(?:-\d+)?)',
    re.IGNORECASE
)


# =============================================================================
# AUTHOR METADATA EXTRACTION
# =============================================================================

KNOWN_AUTHORS = {
    "Ioannes Damascenus": {"name": "John of Damascus", "dates": "675-749"},
    "John Damascene": {"name": "John of Damascus", "dates": "675-749"},
    "John of Damascus": {"name": "John of Damascus", "dates": "675-749"},
    "Augustine": {"name": "Augustine of Hippo", "dates": "354-430"},
    "Athanasius": {"name": "Athanasius of Alexandria", "dates": "296-373"},
    "John Climacus": {"name": "John Climacus", "dates": "525-606"},
    "St. John Climacus": {"name": "John Climacus", "dates": "525-606"},
    "Gregory": {"name": "Gregory the Theologian", "dates": "329-390"},
    "St.GregoryTheologion": {"name": "Gregory the Theologian", "dates": "329-390"},
    "Gregory Theologian": {"name": "Gregory the Theologian", "dates": "329-390"},
    "Basil": {"name": "Basil the Great", "dates": "330-379"},
    "Chrysostom": {"name": "John Chrysostom", "dates": "347-407"},
    "Cyril": {"name": "Cyril of Alexandria", "dates": "376-444"},
    "Origen": {"name": "Origen of Alexandria", "dates": "185-253"},
    "Evagrius": {"name": "Evagrius Ponticus", "dates": "345-399"},
    "Isaac the Syrian": {"name": "Isaac of Nineveh", "dates": "613-700"},
    "Isaac of Nineveh": {"name": "Isaac of Nineveh", "dates": "613-700"},
    "Symeon": {"name": "Symeon the New Theologian", "dates": "949-1022"},
    "Symeon the New Theologian": {"name": "Symeon the New Theologian", "dates": "949-1022"},
}


def extract_author_from_filename(filename: str) -> Tuple[str, str, str]:
    """Extract author, dates, and title from filename."""
    # Pattern like "0675-0749,_Ioannes_Damascenus,_De_Fide_Orthodoxa"
    dated_pattern = re.match(
        r'(\d{4})-(\d{4}),_([^,]+),_(.+?)(?:,_\w+)?\.clean\.txt$',
        filename
    )
    if dated_pattern:
        dates = f"{dated_pattern.group(1)}-{dated_pattern.group(2)}"
        author_raw = dated_pattern.group(3).replace("_", " ")
        title = dated_pattern.group(4).replace("_", " ")

        # Normalize author name
        author = author_raw
        for known, info in KNOWN_AUTHORS.items():
            if known.lower() in author_raw.lower():
                author = info["name"]
                break

        return author, dates, title

    # Pattern like "TheLadderofDivineAscent.clean.txt"
    simple_pattern = re.match(r'(.+?)\.clean\.txt$', filename)
    if simple_pattern:
        title = simple_pattern.group(1)
        # Convert CamelCase to spaces
        title = re.sub(r'([a-z])([A-Z])', r'\1 \2', title)
        title = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', title)

        # Try to extract author from title
        author = "Unknown"
        dates = ""
        for known, info in KNOWN_AUTHORS.items():
            if known.lower() in title.lower() or known.lower() in filename.lower():
                author = info["name"]
                dates = info["dates"]
                break

        return author, dates, title

    return "Unknown", "", filename


# =============================================================================
# TEXT PARSING
# =============================================================================

def is_garbage_line(line: str) -> bool:
    """Check if a line is garbage (URLs, metadata, etc.)."""
    line = line.strip()
    if not line:
        return True
    for pattern in GARBAGE_COMPILED:
        if pattern.search(line):
            return True
    return False


def clean_content(text: str) -> str:
    """Clean text content, removing garbage and normalizing whitespace."""
    lines = text.split('\n')
    cleaned = []

    for line in lines:
        if not is_garbage_line(line):
            # Clean up common OCR/encoding issues
            line = line.replace('\u2019', "'")  # Right single quote
            line = line.replace('\u2018', "'")  # Left single quote
            line = line.replace('\u201c', '"')  # Left double quote
            line = line.replace('\u201d', '"')  # Right double quote
            line = line.replace('\u2014', '--')  # Em dash
            line = line.replace('\u2013', '-')   # En dash
            cleaned.append(line.strip())

    # Join and normalize whitespace
    result = '\n'.join(cleaned)
    result = re.sub(r'\n{3,}', '\n\n', result)  # Max 2 consecutive newlines
    return result.strip()


def extract_footnotes(text: str) -> Tuple[str, List[str]]:
    """Extract footnotes from text and return (main_text, footnotes)."""
    footnotes = []

    # Pattern for numbered footnotes like "1 Lit. 'head', Gk. kephale"
    footnote_pattern = re.compile(r'^(\d+)\s+(.+)$', re.MULTILINE)

    lines = text.split('\n')
    main_lines = []
    in_footnotes = False

    for line in lines:
        stripped = line.strip()

        # Detect start of footnotes section (multiple short numbered lines)
        if re.match(r'^\d+\s+.{10,80}$', stripped) and len(stripped) < 100:
            in_footnotes = True

        if in_footnotes:
            match = footnote_pattern.match(stripped)
            if match:
                footnotes.append(f"[{match.group(1)}] {match.group(2)}")
            elif stripped:
                # Continue previous footnote
                if footnotes:
                    footnotes[-1] += " " + stripped
        else:
            main_lines.append(line)

    return '\n'.join(main_lines), footnotes


def extract_scripture_refs(text: str) -> List[str]:
    """Extract scripture references from text."""
    refs = []
    for match in SCRIPTURE_PATTERN.finditer(text):
        book = match.group(1)
        chapter = match.group(2)
        verse = match.group(3)

        # Normalize book code
        book_code = BOOK_ABBREVS.get(book, book.upper()[:3])
        ref = f"{book_code}.{chapter}.{verse}"
        if ref not in refs:
            refs.append(ref)

    return refs


def parse_structure(text: str) -> List[Dict[str, str]]:
    """Parse book/chapter/section structure from text."""
    sections = []

    # Patterns for structural markers
    book_pattern = re.compile(r'^(?:BOOK\s+)?([IVX]+|[0-9]+)\s*$', re.MULTILINE | re.IGNORECASE)
    chapter_pattern = re.compile(
        r'^CHAPTER\s+([IVXLCDM]+|[0-9]+)\.?\s*(.*)$',
        re.MULTILINE | re.IGNORECASE
    )
    step_pattern = re.compile(r'^Step\s+(\d+)\s*(.*)$', re.MULTILINE | re.IGNORECASE)

    # Split by chapter markers
    chapters = re.split(r'(?=^CHAPTER\s+|^Step\s+)', text, flags=re.MULTILINE | re.IGNORECASE)

    current_book = ""

    for chapter_text in chapters:
        if not chapter_text.strip():
            continue

        # Check for book marker at start
        book_match = book_pattern.match(chapter_text.strip())
        if book_match:
            current_book = book_match.group(1)
            continue

        # Extract chapter info
        chapter_match = chapter_pattern.match(chapter_text.strip())
        step_match = step_pattern.match(chapter_text.strip())

        if chapter_match:
            sections.append({
                "book": current_book,
                "chapter": chapter_match.group(1),
                "chapter_title": chapter_match.group(2).strip() if chapter_match.group(2) else "",
                "content": chapter_text.strip()
            })
        elif step_match:
            sections.append({
                "book": "",
                "chapter": step_match.group(1),
                "chapter_title": step_match.group(2).strip() if step_match.group(2) else "",
                "content": chapter_text.strip()
            })
        elif chapter_text.strip():
            # Content without chapter marker
            if sections:
                # Append to previous section
                sections[-1]["content"] += "\n\n" + chapter_text.strip()
            else:
                # First section (introduction or untitled)
                sections.append({
                    "book": current_book,
                    "chapter": "0",
                    "chapter_title": "Introduction",
                    "content": chapter_text.strip()
                })

    # If no structure found, treat as single section
    if not sections:
        sections.append({
            "book": "",
            "chapter": "1",
            "chapter_title": "",
            "content": text
        })

    return sections


# =============================================================================
# MAIN NORMALIZER
# =============================================================================

def normalize_file(
    input_path: Path,
    output_dir: Path,
    source_collection: str = ""
) -> List[PatristicTextSchema]:
    """Normalize a single patristic text file to JSON."""
    logger.info(f"Processing: {input_path.name}")

    # Read file
    try:
        with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
            raw_text = f.read()
    except Exception as e:
        logger.error(f"Failed to read {input_path}: {e}")
        return []

    # Extract metadata from filename
    author, dates, title = extract_author_from_filename(input_path.name)

    # Generate text ID
    text_id = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
    if author != "Unknown":
        author_slug = re.sub(r'[^a-z0-9]+', '-', author.split()[-1].lower())
        text_id = f"{author_slug}-{text_id}"

    # Clean content
    cleaned_text = clean_content(raw_text)

    if len(cleaned_text) < 100:
        logger.warning(f"Skipping {input_path.name}: too little content after cleaning")
        return []

    # Parse structure
    sections = parse_structure(cleaned_text)

    # Create schema instances for each section
    results = []

    for section in sections:
        content = section["content"]
        content_clean = clean_content(content)

        # Skip garbage sections
        if len(content_clean) < 50:
            continue

        # Extract footnotes
        main_text, footnotes = extract_footnotes(content_clean)

        # Extract scripture references
        verse_refs = extract_scripture_refs(main_text)

        # Create schema
        schema = PatristicTextSchema(
            text_id=f"{text_id}-{section['book']}-{section['chapter']}".strip('-'),
            author=author,
            author_dates=dates,
            title=title,
            title_english=title,  # Could be enhanced with translation lookup
            book=section.get("book", ""),
            chapter=section.get("chapter", ""),
            chapter_title=section.get("chapter_title", ""),
            content=content,
            content_clean=main_text,
            language="greek",  # Default, could be detected
            translation_language="english",
            source_collection=source_collection,
            source_file=input_path.name,
            verse_references=verse_refs,
            footnotes=footnotes,
            metadata={
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "raw_length": len(raw_text),
                "clean_length": len(main_text),
            }
        )

        # Skip if garbage
        if schema.is_garbage():
            continue

        results.append(schema)

    logger.info(f"  -> Extracted {len(results)} sections from {input_path.name}")
    return results


def normalize_directory(
    input_dir: Path,
    output_dir: Path,
    source_collection: str = ""
) -> int:
    """Normalize all patristic text files in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all .txt files
    txt_files = list(input_dir.glob("*.txt"))
    logger.info(f"Found {len(txt_files)} text files in {input_dir}")

    all_texts = []
    total_sections = 0

    for txt_file in txt_files:
        sections = normalize_file(txt_file, output_dir, source_collection)
        all_texts.extend(sections)
        total_sections += len(sections)

    if all_texts:
        # Write combined JSON
        combined_output = output_dir / "patristic_texts.json"
        with open(combined_output, 'w', encoding='utf-8') as f:
            json.dump(
                [asdict(t) for t in all_texts],
                f,
                indent=2,
                ensure_ascii=False
            )
        logger.info(f"Wrote {len(all_texts)} texts to {combined_output}")

        # Write index file for quick lookup
        index = {}
        for text in all_texts:
            author = text.author
            if author not in index:
                index[author] = []
            index[author].append({
                "text_id": text.text_id,
                "title": text.title,
                "book": text.book,
                "chapter": text.chapter,
                "chapter_title": text.chapter_title,
                "verse_refs_count": len(text.verse_references)
            })

        index_output = output_dir / "patristic_index.json"
        with open(index_output, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote index to {index_output}")

    return total_sections


def main():
    parser = argparse.ArgumentParser(
        description="Normalize patristic text files to JSON format"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input directory containing .txt files"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output directory for JSON files (default: ../data/patristics)"
    )
    parser.add_argument(
        "--collection", "-c",
        type=str,
        default="",
        help="Source collection name (e.g., 'NPNF-2', 'ANF')"
    )

    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input directory does not exist: {args.input}")
        sys.exit(1)

    output_dir = args.output
    if output_dir is None:
        # Default to BIBLOS_v2/data/patristics
        output_dir = Path(__file__).parent.parent / "data" / "patristics"

    total = normalize_directory(args.input, output_dir, args.collection)
    logger.info(f"Normalization complete. Total sections: {total}")


if __name__ == "__main__":
    main()
