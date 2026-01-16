"""
BIBLOS v2 - Base Scraper Agent

Abstract base class for content scraping agents with garbage filtering,
quality assessment, and content normalization capabilities.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Pattern
import asyncio
import hashlib
import json
import re
import unicodedata

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from core.errors import BiblosError, BiblosValidationError, ErrorContext
from data.schemas import ProcessingStatus
from observability import get_tracer, get_logger


class ContentQuality(Enum):
    """Quality assessment for scraped content."""
    EXCELLENT = "excellent"    # High-quality, no cleanup needed
    GOOD = "good"              # Minor cleanup needed
    ACCEPTABLE = "acceptable"  # Moderate cleanup needed
    POOR = "poor"              # Significant cleanup needed
    GARBAGE = "garbage"        # Should be filtered out


class ScraperMode(Enum):
    """Scraper operation modes."""
    DISCOVER = auto()      # Find new sources
    INGEST = auto()        # Ingest content from known sources
    VALIDATE = auto()      # Validate existing content
    NORMALIZE = auto()     # Normalize content format
    DEDUPLICATE = auto()   # Remove duplicates


@dataclass
class ScraperConfig:
    """Configuration for scraper agents."""
    name: str
    source_type: str  # patristic, manuscript, lexicon, etc.

    # Processing settings
    batch_size: int = 100
    max_retries: int = 3
    timeout_seconds: int = 300

    # Quality thresholds
    min_content_length: int = 50
    max_garbage_ratio: float = 0.3
    min_quality_score: float = 0.5

    # Filtering settings
    filter_garbage: bool = True
    normalize_unicode: bool = True
    extract_metadata: bool = True
    detect_language: bool = True

    # Deduplication
    deduplicate: bool = True
    similarity_threshold: float = 0.95

    # Output settings
    output_format: str = "json"  # json, jsonl, parquet

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScraperResult:
    """Result from a scraping operation."""
    scraper_name: str
    source_type: str
    source_path: str
    status: ProcessingStatus
    quality: ContentQuality

    # Content
    content: str = ""
    content_clean: str = ""
    content_length: int = 0

    # Extracted data
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Statistics
    garbage_ratio: float = 0.0
    quality_score: float = 0.0
    processing_time_ms: float = 0.0

    # Issues
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Tracing
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trace_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "scraper_name": self.scraper_name,
            "source_type": self.source_type,
            "source_path": self.source_path,
            "status": self.status.value,
            "quality": self.quality.value,
            "content": self.content,
            "content_clean": self.content_clean,
            "content_length": self.content_length,
            "extracted_data": self.extracted_data,
            "metadata": self.metadata,
            "garbage_ratio": self.garbage_ratio,
            "quality_score": self.quality_score,
            "processing_time_ms": self.processing_time_ms,
            "errors": self.errors,
            "warnings": self.warnings,
            "timestamp": self.timestamp.isoformat(),
            "trace_id": self.trace_id,
        }

    def is_usable(self) -> bool:
        """Check if content is usable."""
        return (
            self.status == ProcessingStatus.COMPLETED
            and self.quality != ContentQuality.GARBAGE
            and self.quality_score >= 0.3
        )


class GarbageFilter:
    """
    Intelligent garbage content filter.

    Detects and filters out meaningless content like:
    - File paths and URLs
    - Timestamps and dates without context
    - Navigation elements
    - Table of contents markers
    - Metadata fragments
    - Encoding artifacts
    """

    # Regex patterns for garbage detection
    GARBAGE_PATTERNS: List[Pattern] = [
        # URLs and file paths
        re.compile(r'(https?://|ftp://|file:///)[^\s]+', re.IGNORECASE),
        re.compile(r'[A-Z]:\\[^\s]+'),  # Windows paths
        re.compile(r'/[a-z]+(/[a-z_]+)+\.[\w]+', re.IGNORECASE),  # Unix paths

        # File extensions
        re.compile(r'\.(htm|html|pdf|doc|txt|xml|json)(\s|$)', re.IGNORECASE),

        # Timestamps and dates (standalone)
        re.compile(r'^\s*\d{4}[-/]\d{2}[-/]\d{2}\s*$'),
        re.compile(r'^\s*\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?\s*$', re.IGNORECASE),

        # Navigation and TOC markers
        re.compile(r'^(index|table of contents|contents|toc)$', re.IGNORECASE),
        re.compile(r'^(next|previous|back|forward|home|up|down)$', re.IGNORECASE),
        re.compile(r'^\s*page\s+\d+\s*$', re.IGNORECASE),

        # Encoding artifacts
        re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]'),  # Control characters
        re.compile(r'\\u[0-9a-fA-F]{4}'),  # Unicode escapes
        re.compile(r'&(nbsp|amp|lt|gt|quot|apos);'),  # HTML entities in raw text

        # Metadata fragments
        re.compile(r'^(created|modified|author|copyright|version)\s*:', re.IGNORECASE),
        re.compile(r'^[\[\{].*[\]\}]$'),  # Bracketed single-line content

        # Empty or whitespace-only
        re.compile(r'^\s*$'),
    ]

    # Keywords that indicate garbage content
    GARBAGE_KEYWORDS: Set[str] = {
        "file:///",
        "http://",
        "https://",
        ".htm",
        ".html",
        ".pdf",
        "index",
        "table of contents",
        "loading...",
        "please wait",
        "javascript",
        "stylesheet",
        "<!DOCTYPE",
        "<html",
        "<head",
        "<body",
        "<?xml",
        "[object Object]",
    }

    # Minimum meaningful content ratio
    MIN_ALPHA_RATIO = 0.4
    MIN_WORD_COUNT = 5

    def __init__(
        self,
        custom_patterns: Optional[List[Pattern]] = None,
        custom_keywords: Optional[Set[str]] = None,
        min_alpha_ratio: float = 0.4,
        min_word_count: int = 5,
    ):
        self.patterns = list(self.GARBAGE_PATTERNS)
        if custom_patterns:
            self.patterns.extend(custom_patterns)

        self.keywords = set(self.GARBAGE_KEYWORDS)
        if custom_keywords:
            self.keywords.update(custom_keywords)

        self.min_alpha_ratio = min_alpha_ratio
        self.min_word_count = min_word_count

    def is_garbage(self, content: str) -> bool:
        """Check if content is garbage."""
        if not content:
            return True

        content_stripped = content.strip()

        # Check length
        if len(content_stripped) < 20:
            return True

        # Check against patterns
        for pattern in self.patterns:
            if pattern.search(content_stripped):
                return True

        # Check against keywords
        content_lower = content_stripped.lower()
        for keyword in self.keywords:
            if keyword.lower() in content_lower:
                return True

        # Check alphabetic ratio
        alpha_chars = sum(1 for c in content_stripped if c.isalpha())
        if len(content_stripped) > 0:
            alpha_ratio = alpha_chars / len(content_stripped)
            if alpha_ratio < self.min_alpha_ratio:
                return True

        # Check word count
        words = content_stripped.split()
        if len(words) < self.min_word_count:
            return True

        return False

    def calculate_garbage_ratio(self, content: str) -> float:
        """Calculate ratio of garbage content."""
        if not content:
            return 1.0

        lines = content.split('\n')
        if not lines:
            return 1.0

        garbage_lines = sum(1 for line in lines if self.is_garbage(line))
        return garbage_lines / len(lines)

    def filter_content(self, content: str) -> str:
        """Remove garbage lines from content."""
        if not content:
            return ""

        lines = content.split('\n')
        clean_lines = [line for line in lines if not self.is_garbage(line)]
        return '\n'.join(clean_lines)

    def assess_quality(self, content: str) -> ContentQuality:
        """Assess content quality."""
        if not content or len(content.strip()) < 20:
            return ContentQuality.GARBAGE

        garbage_ratio = self.calculate_garbage_ratio(content)

        if garbage_ratio >= 0.8:
            return ContentQuality.GARBAGE
        elif garbage_ratio >= 0.5:
            return ContentQuality.POOR
        elif garbage_ratio >= 0.3:
            return ContentQuality.ACCEPTABLE
        elif garbage_ratio >= 0.1:
            return ContentQuality.GOOD
        else:
            return ContentQuality.EXCELLENT


class BaseScraperAgent(ABC):
    """
    Abstract base class for scraper agents.

    Provides common functionality for:
    - Content scraping and ingestion
    - Garbage filtering
    - Quality assessment
    - Content normalization
    - Metadata extraction
    - Deduplication
    """

    def __init__(self, config: ScraperConfig):
        self.config = config
        self.garbage_filter = GarbageFilter()
        self._initialized = False
        self._seen_hashes: Set[str] = set()

        # Observability
        self._tracer = get_tracer(f"biblos.scrapers.{config.name}")
        self._logger = get_logger(f"biblos.scrapers.{config.name}")

    async def initialize(self) -> None:
        """Initialize scraper resources."""
        if self._initialized:
            return

        with self._tracer.start_as_current_span(
            f"scraper.{self.config.name}.initialize",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("scraper.name", self.config.name)
            span.set_attribute("scraper.source_type", self.config.source_type)

            self._logger.info(
                f"Initializing scraper: {self.config.name}",
                scraper=self.config.name,
            )

            try:
                await self._setup_resources()
                self._initialized = True
                span.set_attribute("status", "success")
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    async def _setup_resources(self) -> None:
        """Override to set up scraper-specific resources."""
        pass

    async def shutdown(self) -> None:
        """Clean up scraper resources."""
        with self._tracer.start_as_current_span(
            f"scraper.{self.config.name}.shutdown",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("scraper.name", self.config.name)
            await self._cleanup_resources()
            self._initialized = False

    async def _cleanup_resources(self) -> None:
        """Override to clean up scraper-specific resources."""
        pass

    @abstractmethod
    async def scrape(
        self,
        source: str,
        context: Dict[str, Any]
    ) -> ScraperResult:
        """
        Scrape content from a source.

        Args:
            source: Source identifier (path, URL, etc.)
            context: Additional context for scraping

        Returns:
            ScraperResult with scraped content
        """
        pass

    @abstractmethod
    async def extract_metadata(
        self,
        content: str,
        source: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract metadata from content.

        Args:
            content: Content to extract from
            source: Source identifier
            context: Additional context

        Returns:
            Extracted metadata dictionary
        """
        pass

    @abstractmethod
    def get_source_files(self, source_dir: Path) -> List[Path]:
        """
        Get list of source files to process.

        Args:
            source_dir: Directory to scan

        Returns:
            List of file paths
        """
        pass

    def clean_content(self, content: str) -> str:
        """Clean and normalize content."""
        if not content:
            return ""

        # Normalize unicode
        if self.config.normalize_unicode:
            content = unicodedata.normalize('NFC', content)

        # Remove control characters
        content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', content)

        # Normalize whitespace
        content = re.sub(r'\r\n', '\n', content)
        content = re.sub(r'\r', '\n', content)
        content = re.sub(r'[ \t]+', ' ', content)
        content = re.sub(r'\n{3,}', '\n\n', content)

        # Strip leading/trailing whitespace per line
        lines = [line.strip() for line in content.split('\n')]
        content = '\n'.join(lines)

        # Filter garbage if enabled
        if self.config.filter_garbage:
            content = self.garbage_filter.filter_content(content)

        return content.strip()

    def calculate_quality_score(self, content: str, clean_content: str) -> float:
        """Calculate overall quality score."""
        if not content:
            return 0.0

        scores = []

        # Length score
        if len(clean_content) >= 200:
            scores.append(1.0)
        elif len(clean_content) >= 100:
            scores.append(0.7)
        elif len(clean_content) >= 50:
            scores.append(0.4)
        else:
            scores.append(0.1)

        # Cleanup ratio score (how much was kept after cleaning)
        if len(content) > 0:
            retention = len(clean_content) / len(content)
            scores.append(min(1.0, retention * 1.2))

        # Garbage ratio score
        garbage_ratio = self.garbage_filter.calculate_garbage_ratio(clean_content)
        scores.append(1.0 - garbage_ratio)

        # Word count score
        word_count = len(clean_content.split())
        if word_count >= 100:
            scores.append(1.0)
        elif word_count >= 50:
            scores.append(0.7)
        elif word_count >= 20:
            scores.append(0.4)
        else:
            scores.append(0.1)

        return sum(scores) / len(scores) if scores else 0.0

    def is_duplicate(self, content: str) -> bool:
        """Check if content is a duplicate."""
        if not self.config.deduplicate:
            return False

        content_hash = hashlib.sha256(content.encode()).hexdigest()

        if content_hash in self._seen_hashes:
            return True

        self._seen_hashes.add(content_hash)
        return False

    async def process(
        self,
        source: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ScraperResult:
        """
        Process a source with full pipeline.

        Args:
            source: Source identifier
            context: Additional context

        Returns:
            ScraperResult with processed content
        """
        import time

        context = context or {}
        start_time = time.perf_counter()

        with self._tracer.start_as_current_span(
            f"scraper.{self.config.name}.process",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("scraper.name", self.config.name)
            span.set_attribute("source", source)

            try:
                # Scrape content
                result = await self.scrape(source, context)

                # Clean content
                result.content_clean = self.clean_content(result.content)
                result.content_length = len(result.content_clean)

                # Check for duplicate
                if self.is_duplicate(result.content_clean):
                    result.status = ProcessingStatus.SKIPPED
                    result.warnings.append("Duplicate content detected")
                    return result

                # Calculate quality
                result.garbage_ratio = self.garbage_filter.calculate_garbage_ratio(
                    result.content
                )
                result.quality = self.garbage_filter.assess_quality(result.content_clean)
                result.quality_score = self.calculate_quality_score(
                    result.content, result.content_clean
                )

                # Extract metadata if enabled
                if self.config.extract_metadata and result.quality != ContentQuality.GARBAGE:
                    result.extracted_data = await self.extract_metadata(
                        result.content_clean, source, context
                    )

                # Final status based on quality
                if result.quality == ContentQuality.GARBAGE:
                    result.status = ProcessingStatus.SKIPPED
                    result.warnings.append("Content identified as garbage")
                elif result.quality == ContentQuality.POOR:
                    result.status = ProcessingStatus.NEEDS_REVIEW
                    result.warnings.append("Low quality content - needs review")
                else:
                    result.status = ProcessingStatus.COMPLETED

                result.processing_time_ms = (time.perf_counter() - start_time) * 1000

                # Set span attributes
                span.set_attribute("result.status", result.status.value)
                span.set_attribute("result.quality", result.quality.value)
                span.set_attribute("result.quality_score", result.quality_score)
                span.set_attribute("result.content_length", result.content_length)

                return result

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)

                return ScraperResult(
                    scraper_name=self.config.name,
                    source_type=self.config.source_type,
                    source_path=source,
                    status=ProcessingStatus.FAILED,
                    quality=ContentQuality.GARBAGE,
                    errors=[str(e)],
                    processing_time_ms=(time.perf_counter() - start_time) * 1000,
                )

    async def process_batch(
        self,
        sources: List[str],
        context: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[callable] = None
    ) -> List[ScraperResult]:
        """
        Process multiple sources.

        Args:
            sources: List of source identifiers
            context: Additional context
            progress_callback: Optional callback(current, total)

        Returns:
            List of ScraperResult
        """
        context = context or {}
        results = []
        total = len(sources)

        with self._tracer.start_as_current_span(
            f"scraper.{self.config.name}.process_batch",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("batch.size", total)

            for i, source in enumerate(sources):
                result = await self.process(source, context)
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, total)

            # Record batch statistics
            successful = sum(1 for r in results if r.status == ProcessingStatus.COMPLETED)
            span.set_attribute("batch.successful", successful)
            span.set_attribute("batch.failed", total - successful)

            return results

    async def scan_directory(
        self,
        source_dir: Path,
        context: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[callable] = None
    ) -> List[ScraperResult]:
        """
        Scan and process all files in a directory.

        Args:
            source_dir: Directory to scan
            context: Additional context
            progress_callback: Optional callback(current, total)

        Returns:
            List of ScraperResult
        """
        source_dir = Path(source_dir)

        if not source_dir.exists():
            raise BiblosError(
                f"Source directory not found: {source_dir}",
                context=ErrorContext(
                    operation="scan_directory",
                    component=self.config.name,
                )
            )

        files = self.get_source_files(source_dir)
        sources = [str(f) for f in files]

        self._logger.info(
            f"Found {len(sources)} files to process",
            scraper=self.config.name,
            directory=str(source_dir),
        )

        return await self.process_batch(sources, context, progress_callback)
