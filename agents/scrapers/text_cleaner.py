"""
BIBLOS v2 - Text Cleaner Agent

Post-processing agent for cleaning and normalizing scraped text content.
Works in conjunction with scrapers to improve content quality.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import re
import unicodedata

from agents.scrapers.base_scraper import (
    BaseScraperAgent,
    ScraperConfig,
    ScraperResult,
    ContentQuality,
    GarbageFilter,
)
from data.schemas import ProcessingStatus
from observability import get_tracer, get_logger


class TextCleanerAgent(BaseScraperAgent):
    """
    Text cleaning and normalization agent.

    Performs:
    - Unicode normalization
    - Whitespace standardization
    - OCR error correction
    - Encoding fix
    - Diacritic handling for Greek/Hebrew
    - Quote normalization
    - Hyphenation correction
    """

    # Common OCR errors and corrections
    OCR_CORRECTIONS = {
        # Letter substitutions
        "rn": {"pattern": r'\brn\b', "check_context": True},  # m confusion
        "vv": "w",
        "1": {"in_word": "l", "standalone": "1"},
        "0": {"in_word": "O", "standalone": "0"},
        "|": {"in_word": "I", "standalone": "|"},

        # Common Greek OCR errors
        "ά": "ά",  # Combining vs precomposed
        "έ": "έ",
        "ή": "ή",
        "ί": "ί",
        "ό": "ό",
        "ύ": "ύ",
        "ώ": "ώ",

        # Common ligature issues
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
    }

    # Quote normalization
    QUOTE_MAP = {
        '"': '"',  # Straight double
        '"': '"',  # Left double
        '"': '"',  # Right double
        '„': '"',  # German low
        '«': '"',  # Left guillemet
        '»': '"',  # Right guillemet
        "'": "'",  # Straight single
        ''': "'",  # Left single
        ''': "'",  # Right single
        '‹': "'",  # Left single guillemet
        '›': "'",  # Right single guillemet
    }

    # Hyphenation patterns (line-break artifacts)
    HYPHEN_PATTERN = re.compile(r'(\w+)-\s*\n\s*(\w+)')

    # Multiple space pattern
    MULTI_SPACE = re.compile(r' {2,}')

    # Orphaned punctuation
    ORPHAN_PUNCT = re.compile(r'^\s*[,.;:!?]\s*$', re.MULTILINE)

    def __init__(self, config: Optional[ScraperConfig] = None):
        if config is None:
            config = ScraperConfig(
                name="text_cleaner",
                source_type="processed",
                batch_size=100,
            )
        super().__init__(config)
        self._logger = get_logger("biblos.scrapers.text_cleaner")

    def get_source_files(self, source_dir: Path) -> List[Path]:
        """Get text files for cleaning."""
        extensions = {'.txt', '.md', '.json'}
        files = []
        for ext in extensions:
            files.extend(source_dir.glob(f'**/*{ext}'))
        return sorted(files)

    async def scrape(
        self,
        source: str,
        context: Dict[str, Any]
    ) -> ScraperResult:
        """Read content for cleaning."""
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
            with open(source_path, 'r', encoding='utf-8') as f:
                content = f.read()

            return ScraperResult(
                scraper_name=self.config.name,
                source_type=self.config.source_type,
                source_path=source,
                status=ProcessingStatus.PENDING,
                quality=ContentQuality.ACCEPTABLE,
                content=content,
            )

        except Exception as e:
            return ScraperResult(
                scraper_name=self.config.name,
                source_type=self.config.source_type,
                source_path=source,
                status=ProcessingStatus.FAILED,
                quality=ContentQuality.GARBAGE,
                errors=[f"Failed to read: {e}"],
            )

    async def extract_metadata(
        self,
        content: str,
        source: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract cleaning statistics as metadata."""
        return {
            "original_length": len(content),
            "word_count": len(content.split()),
            "line_count": content.count('\n') + 1,
        }

    def clean_content(self, content: str) -> str:
        """Apply full cleaning pipeline."""
        if not content:
            return ""

        # 1. Unicode normalization
        content = self._normalize_unicode(content)

        # 2. Fix encoding issues
        content = self._fix_encoding(content)

        # 3. Normalize quotes
        content = self._normalize_quotes(content)

        # 4. Fix hyphenation
        content = self._fix_hyphenation(content)

        # 5. Normalize whitespace
        content = self._normalize_whitespace(content)

        # 6. Apply OCR corrections (careful, context-sensitive)
        content = self._apply_ocr_corrections(content)

        # 7. Clean up orphaned punctuation
        content = self._clean_orphan_punctuation(content)

        # 8. Final cleanup
        content = self._final_cleanup(content)

        return content

    def _normalize_unicode(self, content: str) -> str:
        """Apply Unicode NFC normalization."""
        return unicodedata.normalize('NFC', content)

    def _fix_encoding(self, content: str) -> str:
        """Fix common encoding issues."""
        # Fix mojibake from Latin-1/UTF-8 confusion
        replacements = {
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€"': '—',
            'â€"': '–',
            'Ã©': 'é',
            'Ã¨': 'è',
            'Ã ': 'à',
            'Ã¢': 'â',
            'Ã´': 'ô',
            'Ã¶': 'ö',
            'Ã¼': 'ü',
        }

        for bad, good in replacements.items():
            content = content.replace(bad, good)

        return content

    def _normalize_quotes(self, content: str) -> str:
        """Normalize all quote characters."""
        for fancy, simple in self.QUOTE_MAP.items():
            content = content.replace(fancy, simple)
        return content

    def _fix_hyphenation(self, content: str) -> str:
        """Fix word breaks from line-end hyphenation."""
        # Join hyphenated words split across lines
        content = self.HYPHEN_PATTERN.sub(r'\1\2', content)
        return content

    def _normalize_whitespace(self, content: str) -> str:
        """Normalize whitespace throughout."""
        # Normalize line endings
        content = content.replace('\r\n', '\n').replace('\r', '\n')

        # Collapse multiple spaces
        content = self.MULTI_SPACE.sub(' ', content)

        # Remove trailing whitespace per line
        lines = [line.rstrip() for line in content.split('\n')]
        content = '\n'.join(lines)

        # Collapse multiple blank lines
        content = re.sub(r'\n{3,}', '\n\n', content)

        return content

    def _apply_ocr_corrections(self, content: str) -> str:
        """Apply careful OCR corrections."""
        # Fix ligatures unconditionally
        for lig, expanded in [('ﬁ', 'fi'), ('ﬂ', 'fl'), ('ﬀ', 'ff'),
                             ('ﬃ', 'ffi'), ('ﬄ', 'ffl')]:
            content = content.replace(lig, expanded)

        return content

    def _clean_orphan_punctuation(self, content: str) -> str:
        """Remove orphaned punctuation on its own lines."""
        return self.ORPHAN_PUNCT.sub('', content)

    def _final_cleanup(self, content: str) -> str:
        """Final cleanup pass."""
        # Strip leading/trailing whitespace
        content = content.strip()

        # Remove null bytes
        content = content.replace('\x00', '')

        # Ensure single newline at end
        if content and not content.endswith('\n'):
            content += '\n'

        return content

    def calculate_quality_score(self, content: str, clean_content: str) -> float:
        """Calculate quality based on cleaning effectiveness."""
        if not content or not clean_content:
            return 0.0

        base_score = super().calculate_quality_score(content, clean_content)

        # Bonus for minimal cleaning needed (already clean)
        if len(content) > 0:
            change_ratio = abs(len(content) - len(clean_content)) / len(content)
            if change_ratio < 0.05:
                base_score = min(1.0, base_score + 0.1)

        return base_score
