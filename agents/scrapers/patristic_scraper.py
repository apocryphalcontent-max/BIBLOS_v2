"""
BIBLOS v2 - Patristic Scraper Agent

Specialized scraper for patristic texts with:
- Author detection and normalization
- Scripture reference extraction
- Theological theme identification
- Section/chapter parsing
- Footnote extraction
- Quality assessment for patristic content
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Pattern
import json
import re

from agents.scrapers.base_scraper import (
    BaseScraperAgent,
    ScraperConfig,
    ScraperResult,
    ContentQuality,
    GarbageFilter,
)
from data.schemas import ProcessingStatus, PatristicTextSchema
from observability import get_tracer, get_logger


# Church Fathers metadata
CHURCH_FATHERS: Dict[str, Dict[str, Any]] = {
    "chrysostom": {
        "canonical_name": "John Chrysostom",
        "era": "4th-5th century",
        "school": "antiochene",
        "aliases": ["chrysostomus", "john chrysostom", "st. chrysostom", "golden mouth"],
    },
    "origen": {
        "canonical_name": "Origen of Alexandria",
        "era": "3rd century",
        "school": "alexandrian",
        "aliases": ["origen", "origenes", "origen of alexandria"],
    },
    "augustine": {
        "canonical_name": "Augustine of Hippo",
        "era": "4th-5th century",
        "school": "latin",
        "aliases": ["augustine", "st. augustine", "augustinus", "austin"],
    },
    "basil": {
        "canonical_name": "Basil the Great",
        "era": "4th century",
        "school": "cappadocian",
        "aliases": ["basil", "basil the great", "basilius", "st. basil"],
    },
    "gregory_nyssa": {
        "canonical_name": "Gregory of Nyssa",
        "era": "4th century",
        "school": "cappadocian",
        "aliases": ["gregory of nyssa", "gregory nyssa", "nyssen"],
    },
    "gregory_nazianzen": {
        "canonical_name": "Gregory the Theologian",
        "era": "4th century",
        "school": "cappadocian",
        "aliases": ["gregory nazianzen", "gregory the theologian", "gregory of nazianzus"],
    },
    "athanasius": {
        "canonical_name": "Athanasius of Alexandria",
        "era": "4th century",
        "school": "alexandrian",
        "aliases": ["athanasius", "st. athanasius", "athanasius contra mundum"],
    },
    "cyril_alexandria": {
        "canonical_name": "Cyril of Alexandria",
        "era": "5th century",
        "school": "alexandrian",
        "aliases": ["cyril of alexandria", "cyril", "st. cyril"],
    },
    "jerome": {
        "canonical_name": "Jerome",
        "era": "4th-5th century",
        "school": "latin",
        "aliases": ["jerome", "st. jerome", "hieronymus", "eusebius hieronymus"],
    },
    "ephrem": {
        "canonical_name": "Ephrem the Syrian",
        "era": "4th century",
        "school": "syrian",
        "aliases": ["ephrem", "ephraim", "st. ephrem", "ephrem the syrian"],
    },
    "john_damascene": {
        "canonical_name": "John of Damascus",
        "era": "8th century",
        "school": "eastern",
        "aliases": ["john damascene", "john of damascus", "damascene"],
    },
    "maximus": {
        "canonical_name": "Maximus the Confessor",
        "era": "7th century",
        "school": "eastern",
        "aliases": ["maximus", "maximus the confessor", "st. maximus"],
    },
    "irenaeus": {
        "canonical_name": "Irenaeus of Lyons",
        "era": "2nd century",
        "school": "early",
        "aliases": ["irenaeus", "st. irenaeus", "irenaeus of lyons"],
    },
    "clement_alexandria": {
        "canonical_name": "Clement of Alexandria",
        "era": "2nd-3rd century",
        "school": "alexandrian",
        "aliases": ["clement of alexandria", "clement", "titus flavius clemens"],
    },
    "tertullian": {
        "canonical_name": "Tertullian",
        "era": "2nd-3rd century",
        "school": "latin",
        "aliases": ["tertullian", "quintus septimius florens tertullianus"],
    },
    "ambrose": {
        "canonical_name": "Ambrose of Milan",
        "era": "4th century",
        "school": "latin",
        "aliases": ["ambrose", "st. ambrose", "ambrose of milan", "ambrosius"],
    },
    "leo_great": {
        "canonical_name": "Leo the Great",
        "era": "5th century",
        "school": "latin",
        "aliases": ["leo the great", "pope leo", "leo i"],
    },
    "gregory_great": {
        "canonical_name": "Gregory the Great",
        "era": "6th century",
        "school": "latin",
        "aliases": ["gregory the great", "pope gregory", "gregorius magnus"],
    },
}

# Scripture reference patterns
SCRIPTURE_PATTERNS = [
    # Standard format: "Genesis 1:1" or "Gen 1:1"
    re.compile(r'(?P<book>\d?\s?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?P<chapter>\d+):(?P<verse>\d+(?:-\d+)?)', re.IGNORECASE),
    # Roman numerals: "Romans ii, 11"
    re.compile(r'(?P<book>\d?\s?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?P<chapter>[ivxlc]+)\s*[,.:]\s*(?P<verse>\d+(?:-\d+)?)', re.IGNORECASE),
    # Abbreviated: "Rom 1.1" or "Gen. 1.1"
    re.compile(r'(?P<book>\d?\s?[A-Z][a-z]+\.?)\s*(?P<chapter>\d+)\.(?P<verse>\d+(?:-\d+)?)', re.IGNORECASE),
    # Parenthetical: "(Gen. 1:1)"
    re.compile(r'\((?P<book>\d?\s?[A-Z][a-z]+\.?)\s*(?P<chapter>\d+)[:.]\s*(?P<verse>\d+(?:-\d+)?)\)', re.IGNORECASE),
]

# Book name normalization
BOOK_ALIASES = {
    "genesis": "GEN", "gen": "GEN", "gn": "GEN",
    "exodus": "EXO", "exod": "EXO", "exo": "EXO", "ex": "EXO",
    "leviticus": "LEV", "lev": "LEV", "lv": "LEV",
    "numbers": "NUM", "num": "NUM", "nm": "NUM",
    "deuteronomy": "DEU", "deut": "DEU", "dt": "DEU",
    "joshua": "JOS", "josh": "JOS",
    "judges": "JDG", "judg": "JDG", "jdg": "JDG",
    "ruth": "RUT", "rth": "RUT",
    "1 samuel": "1SA", "1samuel": "1SA", "1sam": "1SA", "1 sam": "1SA",
    "2 samuel": "2SA", "2samuel": "2SA", "2sam": "2SA", "2 sam": "2SA",
    "1 kings": "1KI", "1kings": "1KI", "1 kgs": "1KI", "1kgs": "1KI",
    "2 kings": "2KI", "2kings": "2KI", "2 kgs": "2KI", "2kgs": "2KI",
    "1 chronicles": "1CH", "1chronicles": "1CH", "1 chr": "1CH", "1chr": "1CH",
    "2 chronicles": "2CH", "2chronicles": "2CH", "2 chr": "2CH", "2chr": "2CH",
    "ezra": "EZR", "ezr": "EZR",
    "nehemiah": "NEH", "neh": "NEH",
    "esther": "EST", "esth": "EST",
    "job": "JOB", "jb": "JOB",
    "psalms": "PSA", "psalm": "PSA", "ps": "PSA", "pss": "PSA",
    "proverbs": "PRO", "prov": "PRO", "prv": "PRO",
    "ecclesiastes": "ECC", "eccl": "ECC", "eccles": "ECC", "qoh": "ECC",
    "song of solomon": "SNG", "song": "SNG", "cant": "SNG", "canticles": "SNG",
    "isaiah": "ISA", "isa": "ISA", "is": "ISA",
    "jeremiah": "JER", "jer": "JER",
    "lamentations": "LAM", "lam": "LAM",
    "ezekiel": "EZK", "ezek": "EZK", "ez": "EZK",
    "daniel": "DAN", "dan": "DAN", "dn": "DAN",
    "hosea": "HOS", "hos": "HOS",
    "joel": "JOL", "jl": "JOL",
    "amos": "AMO", "am": "AMO",
    "obadiah": "OBA", "obad": "OBA", "ob": "OBA",
    "jonah": "JON", "jon": "JON",
    "micah": "MIC", "mic": "MIC",
    "nahum": "NAH", "nah": "NAH", "na": "NAH",
    "habakkuk": "HAB", "hab": "HAB",
    "zephaniah": "ZEP", "zeph": "ZEP",
    "haggai": "HAG", "hag": "HAG",
    "zechariah": "ZEC", "zech": "ZEC",
    "malachi": "MAL", "mal": "MAL",
    "matthew": "MAT", "matt": "MAT", "mt": "MAT",
    "mark": "MRK", "mrk": "MRK", "mk": "MRK",
    "luke": "LUK", "lk": "LUK",
    "john": "JHN", "jn": "JHN", "jhn": "JHN",
    "acts": "ACT", "ac": "ACT",
    "romans": "ROM", "rom": "ROM", "rm": "ROM",
    "1 corinthians": "1CO", "1corinthians": "1CO", "1 cor": "1CO", "1cor": "1CO",
    "2 corinthians": "2CO", "2corinthians": "2CO", "2 cor": "2CO", "2cor": "2CO",
    "galatians": "GAL", "gal": "GAL",
    "ephesians": "EPH", "eph": "EPH",
    "philippians": "PHP", "phil": "PHP", "php": "PHP",
    "colossians": "COL", "col": "COL",
    "1 thessalonians": "1TH", "1thessalonians": "1TH", "1 thess": "1TH", "1thess": "1TH",
    "2 thessalonians": "2TH", "2thessalonians": "2TH", "2 thess": "2TH", "2thess": "2TH",
    "1 timothy": "1TI", "1timothy": "1TI", "1 tim": "1TI", "1tim": "1TI",
    "2 timothy": "2TI", "2timothy": "2TI", "2 tim": "2TI", "2tim": "2TI",
    "titus": "TIT", "tit": "TIT",
    "philemon": "PHM", "phlm": "PHM", "phm": "PHM",
    "hebrews": "HEB", "heb": "HEB",
    "james": "JAS", "jas": "JAS", "jm": "JAS",
    "1 peter": "1PE", "1peter": "1PE", "1 pet": "1PE", "1pet": "1PE",
    "2 peter": "2PE", "2peter": "2PE", "2 pet": "2PE", "2pet": "2PE",
    "1 john": "1JN", "1john": "1JN", "1 jn": "1JN", "1jn": "1JN",
    "2 john": "2JN", "2john": "2JN", "2 jn": "2JN", "2jn": "2JN",
    "3 john": "3JN", "3john": "3JN", "3 jn": "3JN", "3jn": "3JN",
    "jude": "JUD", "jd": "JUD",
    "revelation": "REV", "rev": "REV", "apocalypse": "REV",
}


@dataclass
class PatristicGarbageFilter(GarbageFilter):
    """
    Extended garbage filter for patristic texts.

    Additional filtering for:
    - OCR artifacts
    - Incomplete translations
    - Modern editorial notes
    - Copyright notices
    """

    PATRISTIC_GARBAGE_PATTERNS = [
        # OCR artifacts
        re.compile(r'[|l1I]{3,}'),  # Repeated vertical chars
        re.compile(r'[_]{5,}'),  # Underline separators
        re.compile(r'\[?\?\?\?+\]?'),  # Unknown character markers

        # Editorial markers
        re.compile(r'^\s*\[editor[^\]]*\]', re.IGNORECASE),
        re.compile(r'^\s*\[translator[^\]]*\]', re.IGNORECASE),
        re.compile(r'^\s*\[note:', re.IGNORECASE),

        # Copyright and metadata
        re.compile(r'copyright\s+\d{4}', re.IGNORECASE),
        re.compile(r'all rights reserved', re.IGNORECASE),
        re.compile(r'public domain', re.IGNORECASE),

        # Navigation from digitized texts
        re.compile(r'^\s*\[page\s+\d+\]', re.IGNORECASE),
        re.compile(r'^\s*—+\s*$'),  # Horizontal rules

        # Scan artifacts
        re.compile(r'^\s*[0-9]+\s*$'),  # Lone page numbers
        re.compile(r'^\s*[ivxlcdm]+\s*$', re.IGNORECASE),  # Roman numeral pages
    ]

    PATRISTIC_GARBAGE_KEYWORDS = {
        "scanned by",
        "digitized by",
        "proof-read",
        "this text is",
        "the following text",
        "transcribed from",
        "original source:",
        "source url:",
        "[blank page]",
        "[missing text]",
        "[illegible]",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.patterns.extend(self.PATRISTIC_GARBAGE_PATTERNS)
        self.keywords.update(self.PATRISTIC_GARBAGE_KEYWORDS)


class PatristicScraperAgent(BaseScraperAgent):
    """
    Specialized scraper for patristic texts.

    Handles:
    - Author identification from filenames and content
    - Scripture reference extraction and normalization
    - Section/chapter parsing
    - Footnote extraction
    - Theological theme detection
    - Quality assessment for patristic content
    """

    # Chapter/section patterns
    CHAPTER_PATTERNS = [
        re.compile(r'^Chapter\s+([IVXLCDM]+|\d+)[:\.]?\s*(.*)$', re.IGNORECASE),
        re.compile(r'^Book\s+([IVXLCDM]+|\d+)[:\.]?\s*(.*)$', re.IGNORECASE),
        re.compile(r'^Homily\s+([IVXLCDM]+|\d+)[:\.]?\s*(.*)$', re.IGNORECASE),
        re.compile(r'^Sermon\s+([IVXLCDM]+|\d+)[:\.]?\s*(.*)$', re.IGNORECASE),
        re.compile(r'^([IVXLCDM]+|\d+)\.\s+(.+)$'),  # "I. Title" format
    ]

    # Footnote patterns
    FOOTNOTE_PATTERNS = [
        re.compile(r'\[(\d+)\]\s*(.+?)(?=\[\d+\]|$)', re.DOTALL),
        re.compile(r'\*+\s*(.+?)(?=\*+|$)', re.DOTALL),
    ]

    def __init__(self, config: Optional[ScraperConfig] = None):
        if config is None:
            config = ScraperConfig(
                name="patristic_scraper",
                source_type="patristic",
                batch_size=50,
                min_content_length=100,
                max_garbage_ratio=0.4,
                min_quality_score=0.4,
            )
        super().__init__(config)

        # Use patristic-specific garbage filter
        self.garbage_filter = PatristicGarbageFilter()

        self._logger = get_logger("biblos.scrapers.patristic")

    def get_source_files(self, source_dir: Path) -> List[Path]:
        """Get patristic text files from directory."""
        source_dir = Path(source_dir)
        extensions = {'.txt', '.md', '.text', '.html', '.htm'}

        files = []
        for ext in extensions:
            files.extend(source_dir.glob(f'**/*{ext}'))

        # Sort by name for consistent ordering
        return sorted(files)

    async def scrape(
        self,
        source: str,
        context: Dict[str, Any]
    ) -> ScraperResult:
        """Scrape patristic text from source file."""
        source_path = Path(source)

        if not source_path.exists():
            return ScraperResult(
                scraper_name=self.config.name,
                source_type=self.config.source_type,
                source_path=source,
                status=ProcessingStatus.FAILED,
                quality=ContentQuality.GARBAGE,
                errors=[f"File not found: {source}"],
            )

        try:
            # Read content with encoding detection
            content = self._read_file(source_path)

            # Extract author from filename
            author_info = self._extract_author_from_filename(source_path.stem)

            return ScraperResult(
                scraper_name=self.config.name,
                source_type=self.config.source_type,
                source_path=source,
                status=ProcessingStatus.PENDING,
                quality=ContentQuality.ACCEPTABLE,  # Will be recalculated
                content=content,
                metadata={
                    "filename": source_path.name,
                    "file_size": source_path.stat().st_size,
                    "author_from_filename": author_info,
                },
            )

        except Exception as e:
            return ScraperResult(
                scraper_name=self.config.name,
                source_type=self.config.source_type,
                source_path=source,
                status=ProcessingStatus.FAILED,
                quality=ContentQuality.GARBAGE,
                errors=[f"Failed to read file: {e}"],
            )

    def _read_file(self, file_path: Path) -> str:
        """Read file with encoding detection."""
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        # Fallback: read as bytes and decode with error handling
        with open(file_path, 'rb') as f:
            content = f.read()
            return content.decode('utf-8', errors='replace')

    def _extract_author_from_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Extract author information from filename."""
        filename_lower = filename.lower().replace('_', ' ').replace('-', ' ')

        for key, info in CHURCH_FATHERS.items():
            # Check canonical name
            if info["canonical_name"].lower() in filename_lower:
                return {
                    "key": key,
                    "canonical_name": info["canonical_name"],
                    "era": info["era"],
                    "school": info["school"],
                }

            # Check aliases
            for alias in info["aliases"]:
                if alias.lower() in filename_lower:
                    return {
                        "key": key,
                        "canonical_name": info["canonical_name"],
                        "era": info["era"],
                        "school": info["school"],
                    }

        return None

    async def extract_metadata(
        self,
        content: str,
        source: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract metadata from patristic content."""
        metadata = {}

        # Extract scripture references
        references = self._extract_scripture_references(content)
        metadata["verse_references"] = references

        # Parse sections
        sections = self._parse_sections(content)
        metadata["sections"] = sections

        # Extract footnotes
        footnotes = self._extract_footnotes(content)
        metadata["footnotes"] = footnotes

        # Detect author from content
        author_from_content = self._detect_author_from_content(content)
        if author_from_content:
            metadata["author_from_content"] = author_from_content

        # Detect theological themes
        themes = self._detect_themes(content)
        metadata["themes"] = themes

        # Detect work title
        title = self._detect_title(content)
        if title:
            metadata["title"] = title

        return metadata

    def _extract_scripture_references(self, content: str) -> List[str]:
        """Extract and normalize scripture references."""
        references = set()

        for pattern in SCRIPTURE_PATTERNS:
            for match in pattern.finditer(content):
                try:
                    book = match.group('book').strip().lower().rstrip('.')
                    chapter = match.group('chapter')
                    verse = match.group('verse')

                    # Convert Roman numerals if needed
                    if chapter.lower() in 'ivxlcdm' * 3:
                        chapter = str(self._roman_to_int(chapter))

                    # Normalize book name
                    book_code = BOOK_ALIASES.get(book)
                    if book_code:
                        ref = f"{book_code}.{chapter}.{verse}"
                        references.add(ref)
                except (KeyError, ValueError):
                    continue

        return sorted(list(references))

    def _roman_to_int(self, roman: str) -> int:
        """Convert Roman numeral to integer."""
        roman = roman.upper()
        roman_values = {
            'I': 1, 'V': 5, 'X': 10, 'L': 50,
            'C': 100, 'D': 500, 'M': 1000
        }

        result = 0
        prev_value = 0

        for char in reversed(roman):
            value = roman_values.get(char, 0)
            if value < prev_value:
                result -= value
            else:
                result += value
            prev_value = value

        return result

    def _parse_sections(self, content: str) -> List[Dict[str, Any]]:
        """Parse content into sections/chapters."""
        sections = []
        lines = content.split('\n')
        current_section = None
        current_content = []

        for line in lines:
            section_match = None
            for pattern in self.CHAPTER_PATTERNS:
                section_match = pattern.match(line.strip())
                if section_match:
                    break

            if section_match:
                # Save previous section
                if current_section:
                    current_section["content"] = '\n'.join(current_content).strip()
                    sections.append(current_section)

                # Start new section
                current_section = {
                    "number": section_match.group(1),
                    "title": section_match.group(2).strip() if section_match.lastindex >= 2 else "",
                    "content": "",
                }
                current_content = []
            else:
                if current_section:
                    current_content.append(line)

        # Save last section
        if current_section:
            current_section["content"] = '\n'.join(current_content).strip()
            sections.append(current_section)

        return sections

    def _extract_footnotes(self, content: str) -> List[str]:
        """Extract footnotes from content."""
        footnotes = []

        for pattern in self.FOOTNOTE_PATTERNS:
            for match in pattern.finditer(content):
                footnote_text = match.group(2 if match.lastindex >= 2 else 1).strip()
                if len(footnote_text) > 10:  # Filter tiny footnotes
                    footnotes.append(footnote_text)

        return footnotes

    def _detect_author_from_content(self, content: str) -> Optional[Dict[str, Any]]:
        """Detect author from content headers/metadata."""
        content_lower = content.lower()

        # Check first 500 characters for author mentions
        header = content_lower[:500]

        for key, info in CHURCH_FATHERS.items():
            canonical = info["canonical_name"].lower()
            if canonical in header:
                return {
                    "key": key,
                    "canonical_name": info["canonical_name"],
                    "era": info["era"],
                    "school": info["school"],
                    "detection_method": "header_match",
                }

            for alias in info["aliases"]:
                if alias.lower() in header:
                    return {
                        "key": key,
                        "canonical_name": info["canonical_name"],
                        "era": info["era"],
                        "school": info["school"],
                        "detection_method": "alias_match",
                    }

        return None

    def _detect_themes(self, content: str) -> List[str]:
        """Detect theological themes in content."""
        content_lower = content.lower()

        themes = []
        theme_keywords = {
            "christology": ["christ", "logos", "incarnation", "son of god", "messiah"],
            "trinity": ["trinity", "father, son", "three persons", "consubstantial"],
            "soteriology": ["salvation", "redemption", "atonement", "justification"],
            "ecclesiology": ["church", "body of christ", "bride", "eucharist"],
            "eschatology": ["kingdom", "judgment", "resurrection", "eternal life"],
            "mariology": ["mary", "theotokos", "virgin", "mother of god"],
            "asceticism": ["fasting", "prayer", "vigil", "monk", "desert"],
            "typology": ["type", "antitype", "shadow", "fulfillment", "figure"],
            "creation": ["creation", "genesis", "image of god", "likeness"],
            "baptism": ["baptism", "regeneration", "illumination", "catechumen"],
        }

        for theme, keywords in theme_keywords.items():
            if any(kw in content_lower for kw in keywords):
                themes.append(theme)

        return themes

    def _detect_title(self, content: str) -> Optional[str]:
        """Detect work title from content."""
        lines = content.split('\n')[:10]  # Check first 10 lines

        for line in lines:
            line = line.strip()
            # Skip empty lines and garbage
            if not line or self.garbage_filter.is_garbage(line):
                continue

            # Look for title patterns
            if len(line) < 200 and not line.endswith('.'):
                # Likely a title
                return line

        return None

    async def convert_to_schema(
        self,
        result: ScraperResult,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[PatristicTextSchema]:
        """Convert scraper result to PatristicTextSchema."""
        if not result.is_usable():
            return None

        context = context or {}

        # Determine author
        author_info = result.metadata.get("author_from_content") or \
                     result.metadata.get("author_from_filename")

        author = author_info.get("canonical_name", "Unknown") if author_info else "Unknown"
        author_dates = author_info.get("era", "") if author_info else ""

        # Get extracted data
        extracted = result.extracted_data

        # Generate text ID
        source_path = Path(result.source_path)
        text_id = f"patristic_{source_path.stem}_{hash(result.content_clean) % 10000:04d}"

        return PatristicTextSchema(
            text_id=text_id,
            author=author,
            author_dates=author_dates,
            title=extracted.get("title", source_path.stem),
            title_english=extracted.get("title", source_path.stem),
            content=result.content,
            content_clean=result.content_clean,
            language="greek",  # Assumed; could be detected
            translation_language="english",
            source_file=source_path.name,
            verse_references=extracted.get("verse_references", []),
            themes=extracted.get("themes", []),
            footnotes=extracted.get("footnotes", []),
            metadata={
                "quality": result.quality.value,
                "quality_score": result.quality_score,
                "sections": extracted.get("sections", []),
                "scraper": self.config.name,
            },
        )

    async def process_to_json(
        self,
        source_dir: Path,
        output_path: Path,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Process all files in directory and output to JSON.

        Args:
            source_dir: Source directory containing patristic texts
            output_path: Output JSON file path

        Returns:
            Statistics about the processing
        """
        results = await self.scan_directory(source_dir, progress_callback=progress_callback)

        # Convert to schemas
        schemas = []
        for result in results:
            schema = await self.convert_to_schema(result)
            if schema:
                schemas.append(schema)

        # Convert to dicts for JSON serialization
        data = [s.to_dict() for s in schemas]

        # Write output
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Generate statistics
        stats = {
            "total_files": len(results),
            "successful": sum(1 for r in results if r.status == ProcessingStatus.COMPLETED),
            "failed": sum(1 for r in results if r.status == ProcessingStatus.FAILED),
            "skipped": sum(1 for r in results if r.status == ProcessingStatus.SKIPPED),
            "needs_review": sum(1 for r in results if r.status == ProcessingStatus.NEEDS_REVIEW),
            "output_file": str(output_path),
            "total_texts": len(schemas),
        }

        self._logger.info(
            f"Processed {stats['total_files']} files -> {stats['total_texts']} texts",
            **stats
        )

        return stats
