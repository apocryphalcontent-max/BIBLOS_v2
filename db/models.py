"""
BIBLOS v2 - SQLAlchemy ORM Models

Database models for biblical text storage, cross-references,
patristic citations, and extraction results.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean, DateTime,
    ForeignKey, Index, UniqueConstraint, Enum as SQLEnum
)
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func
import enum


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class Testament(enum.Enum):
    """Testament enumeration."""
    OT = "OT"
    NT = "NT"


class ConnectionType(enum.Enum):
    """Cross-reference connection types."""
    THEMATIC = "thematic"
    VERBAL = "verbal"
    CONCEPTUAL = "conceptual"
    HISTORICAL = "historical"
    TYPOLOGICAL = "typological"
    PROPHETIC = "prophetic"
    LITURGICAL = "liturgical"
    NARRATIVE = "narrative"
    GENEALOGICAL = "genealogical"
    GEOGRAPHICAL = "geographical"


class Strength(enum.Enum):
    """Connection strength levels."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


class ProcessingStatus(enum.Enum):
    """Extraction processing status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


class Book(Base):
    """Biblical book model with optimized indexes and relationships."""
    __tablename__ = "books"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(3), unique=True, index=True)  # GEN, EXO, MAT
    name: Mapped[str] = mapped_column(String(50))
    testament: Mapped[str] = mapped_column(String(3), index=True)  # OT, NT - indexed for filtering
    order_num: Mapped[int] = mapped_column(index=True)  # Indexed for ordering
    chapter_count: Mapped[int] = mapped_column()
    verse_count: Mapped[int] = mapped_column()
    metadata_: Mapped[Optional[Dict]] = mapped_column("metadata", JSONB, nullable=True)

    # Relationships - use selectin for eager loading to avoid N+1 queries
    verses: Mapped[List["Verse"]] = relationship(
        back_populates="book",
        lazy="selectin",  # Changed from dynamic for better N+1 prevention
        passive_deletes=True,
    )

    __table_args__ = (
        Index("ix_books_testament_order", "testament", "order_num"),
    )

    def __repr__(self) -> str:
        return f"<Book {self.code}: {self.name}>"


class Verse(Base):
    """Biblical verse model with multilingual text, embeddings, and optimized indexes."""
    __tablename__ = "verses"

    id: Mapped[int] = mapped_column(primary_key=True)
    reference: Mapped[str] = mapped_column(String(20), unique=True, index=True)  # GEN.1.1
    book_id: Mapped[int] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"), index=True)
    chapter: Mapped[int] = mapped_column(index=True)  # Indexed for chapter queries
    verse_num: Mapped[int] = mapped_column()

    # Multilingual text
    text_greek: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    text_hebrew: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    text_english: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    text_latin: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    text_syriac: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Embeddings (stored as arrays for pgvector)
    # Note: For production, use pgvector's VECTOR type with HNSW index
    embedding: Mapped[Optional[List[float]]] = mapped_column(ARRAY(Float), nullable=True)
    embedding_greek: Mapped[Optional[List[float]]] = mapped_column(ARRAY(Float), nullable=True)
    embedding_hebrew: Mapped[Optional[List[float]]] = mapped_column(ARRAY(Float), nullable=True)

    # Linguistic analysis (JSONB for flexible schema)
    morphology: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    syntax: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    semantics: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    discourse: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)

    # Metadata
    metadata_: Mapped[Optional[Dict]] = mapped_column("metadata", JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships with optimized lazy loading
    book: Mapped["Book"] = relationship(back_populates="verses", lazy="joined")
    source_refs: Mapped[List["CrossReference"]] = relationship(
        back_populates="source_verse",
        foreign_keys="CrossReference.source_id",
        lazy="selectin",
        passive_deletes=True,
    )
    target_refs: Mapped[List["CrossReference"]] = relationship(
        back_populates="target_verse",
        foreign_keys="CrossReference.target_id",
        lazy="selectin",
        passive_deletes=True,
    )
    patristic_citations: Mapped[List["PatristicCitation"]] = relationship(
        back_populates="verse",
        lazy="selectin",
        passive_deletes=True,
    )
    extraction_results: Mapped[List["ExtractionResult"]] = relationship(
        back_populates="verse",
        lazy="selectin",
        passive_deletes=True,
    )

    __table_args__ = (
        # Composite index for common book+chapter queries
        Index("ix_verses_book_chapter", "book_id", "chapter"),
        # Composite index for book+chapter+verse navigation
        Index("ix_verses_book_chapter_verse", "book_id", "chapter", "verse_num"),
        # GIN index on JSONB fields for efficient JSON queries
        Index("ix_verses_morphology_gin", "morphology", postgresql_using="gin"),
        Index("ix_verses_semantics_gin", "semantics", postgresql_using="gin"),
    )

    def __repr__(self) -> str:
        return f"<Verse {self.reference}>"


class CrossReference(Base):
    """Cross-reference relationship between verses with optimized queries."""
    __tablename__ = "cross_references"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(
        ForeignKey("verses.id", ondelete="CASCADE"),
        index=True
    )
    target_id: Mapped[int] = mapped_column(
        ForeignKey("verses.id", ondelete="CASCADE"),
        index=True
    )

    # Classification with indexed fields for filtering
    connection_type: Mapped[str] = mapped_column(String(20), index=True)
    strength: Mapped[str] = mapped_column(String(10), default="moderate", index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, index=True)  # For threshold queries
    bidirectional: Mapped[bool] = mapped_column(Boolean, default=False)

    # Evidence (array fields for pattern matching)
    patristic_support: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String), nullable=True)
    keywords: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String), nullable=True)
    verbal_parallels: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String), nullable=True)

    # Analysis
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    exegesis: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    metadata_: Mapped[Optional[Dict]] = mapped_column("metadata", JSONB, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships with joined loading for common access patterns
    source_verse: Mapped["Verse"] = relationship(
        back_populates="source_refs",
        foreign_keys=[source_id],
        lazy="joined",
    )
    target_verse: Mapped["Verse"] = relationship(
        back_populates="target_refs",
        foreign_keys=[target_id],
        lazy="joined",
    )

    __table_args__ = (
        # Unique constraint for preventing duplicates
        UniqueConstraint("source_id", "target_id", "connection_type", name="uq_crossref_pair"),
        # Composite index for type + confidence filtering
        Index("ix_crossref_type_confidence", "connection_type", "confidence"),
        # Composite index for source + type queries
        Index("ix_crossref_source_type", "source_id", "connection_type"),
        # Composite index for target + type queries
        Index("ix_crossref_target_type", "target_id", "connection_type"),
        # Partial index for high confidence cross-references
        Index(
            "ix_crossref_high_confidence",
            "source_id", "target_id",
            postgresql_where="confidence >= 0.7"
        ),
        # GIN index on keywords array for containment queries
        Index("ix_crossref_keywords_gin", "keywords", postgresql_using="gin"),
    )

    def __repr__(self) -> str:
        return f"<CrossRef {self.source_id} -> {self.target_id} ({self.connection_type})>"


class PatristicCitation(Base):
    """Patristic commentary and citations."""
    __tablename__ = "patristic_citations"

    id: Mapped[int] = mapped_column(primary_key=True)
    verse_id: Mapped[int] = mapped_column(ForeignKey("verses.id"), index=True)

    # Source information
    father_name: Mapped[str] = mapped_column(String(100), index=True)
    work_title: Mapped[str] = mapped_column(String(200))
    citation_ref: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Text content
    text_greek: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    text_latin: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    text_english: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Classification
    category: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # exegetical, homiletic, doctrinal
    century: Mapped[Optional[int]] = mapped_column(nullable=True)
    tradition: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # Greek, Latin, Syriac

    # Analysis
    themes: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String), nullable=True)
    metadata_: Mapped[Optional[Dict]] = mapped_column("metadata", JSONB, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    verse: Mapped["Verse"] = relationship(back_populates="patristic_citations")

    def __repr__(self) -> str:
        return f"<PatristicCitation {self.father_name}: {self.work_title}>"


class ExtractionResult(Base):
    """Results from SDES agent extractions."""
    __tablename__ = "extraction_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    verse_id: Mapped[int] = mapped_column(ForeignKey("verses.id"), index=True)

    # Agent info
    agent_name: Mapped[str] = mapped_column(String(50), index=True)
    extraction_type: Mapped[str] = mapped_column(String(30))
    status: Mapped[str] = mapped_column(String(20))

    # Results
    data: Mapped[Dict] = mapped_column(JSONB)
    confidence: Mapped[float] = mapped_column(Float)
    processing_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Validation
    validated: Mapped[bool] = mapped_column(Boolean, default=False)
    validation_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    verse: Mapped["Verse"] = relationship(back_populates="extraction_results")

    __table_args__ = (
        Index("ix_extraction_verse_agent", "verse_id", "agent_name"),
    )

    def __repr__(self) -> str:
        return f"<ExtractionResult {self.agent_name} for verse {self.verse_id}>"


# Canonical book data for population
CANONICAL_BOOKS = [
    # Old Testament
    {"code": "GEN", "name": "Genesis", "testament": "OT", "order_num": 1, "chapter_count": 50, "verse_count": 1533},
    {"code": "EXO", "name": "Exodus", "testament": "OT", "order_num": 2, "chapter_count": 40, "verse_count": 1213},
    {"code": "LEV", "name": "Leviticus", "testament": "OT", "order_num": 3, "chapter_count": 27, "verse_count": 859},
    {"code": "NUM", "name": "Numbers", "testament": "OT", "order_num": 4, "chapter_count": 36, "verse_count": 1288},
    {"code": "DEU", "name": "Deuteronomy", "testament": "OT", "order_num": 5, "chapter_count": 34, "verse_count": 959},
    {"code": "JOS", "name": "Joshua", "testament": "OT", "order_num": 6, "chapter_count": 24, "verse_count": 658},
    {"code": "JDG", "name": "Judges", "testament": "OT", "order_num": 7, "chapter_count": 21, "verse_count": 618},
    {"code": "RUT", "name": "Ruth", "testament": "OT", "order_num": 8, "chapter_count": 4, "verse_count": 85},
    {"code": "1SA", "name": "1 Samuel", "testament": "OT", "order_num": 9, "chapter_count": 31, "verse_count": 810},
    {"code": "2SA", "name": "2 Samuel", "testament": "OT", "order_num": 10, "chapter_count": 24, "verse_count": 695},
    {"code": "1KI", "name": "1 Kings", "testament": "OT", "order_num": 11, "chapter_count": 22, "verse_count": 816},
    {"code": "2KI", "name": "2 Kings", "testament": "OT", "order_num": 12, "chapter_count": 25, "verse_count": 719},
    {"code": "1CH", "name": "1 Chronicles", "testament": "OT", "order_num": 13, "chapter_count": 29, "verse_count": 942},
    {"code": "2CH", "name": "2 Chronicles", "testament": "OT", "order_num": 14, "chapter_count": 36, "verse_count": 822},
    {"code": "EZR", "name": "Ezra", "testament": "OT", "order_num": 15, "chapter_count": 10, "verse_count": 280},
    {"code": "NEH", "name": "Nehemiah", "testament": "OT", "order_num": 16, "chapter_count": 13, "verse_count": 406},
    {"code": "EST", "name": "Esther", "testament": "OT", "order_num": 17, "chapter_count": 10, "verse_count": 167},
    {"code": "JOB", "name": "Job", "testament": "OT", "order_num": 18, "chapter_count": 42, "verse_count": 1070},
    {"code": "PSA", "name": "Psalms", "testament": "OT", "order_num": 19, "chapter_count": 150, "verse_count": 2461},
    {"code": "PRO", "name": "Proverbs", "testament": "OT", "order_num": 20, "chapter_count": 31, "verse_count": 915},
    {"code": "ECC", "name": "Ecclesiastes", "testament": "OT", "order_num": 21, "chapter_count": 12, "verse_count": 222},
    {"code": "SNG", "name": "Song of Solomon", "testament": "OT", "order_num": 22, "chapter_count": 8, "verse_count": 117},
    {"code": "ISA", "name": "Isaiah", "testament": "OT", "order_num": 23, "chapter_count": 66, "verse_count": 1292},
    {"code": "JER", "name": "Jeremiah", "testament": "OT", "order_num": 24, "chapter_count": 52, "verse_count": 1364},
    {"code": "LAM", "name": "Lamentations", "testament": "OT", "order_num": 25, "chapter_count": 5, "verse_count": 154},
    {"code": "EZK", "name": "Ezekiel", "testament": "OT", "order_num": 26, "chapter_count": 48, "verse_count": 1273},
    {"code": "DAN", "name": "Daniel", "testament": "OT", "order_num": 27, "chapter_count": 12, "verse_count": 357},
    {"code": "HOS", "name": "Hosea", "testament": "OT", "order_num": 28, "chapter_count": 14, "verse_count": 197},
    {"code": "JOL", "name": "Joel", "testament": "OT", "order_num": 29, "chapter_count": 3, "verse_count": 73},
    {"code": "AMO", "name": "Amos", "testament": "OT", "order_num": 30, "chapter_count": 9, "verse_count": 146},
    {"code": "OBA", "name": "Obadiah", "testament": "OT", "order_num": 31, "chapter_count": 1, "verse_count": 21},
    {"code": "JON", "name": "Jonah", "testament": "OT", "order_num": 32, "chapter_count": 4, "verse_count": 48},
    {"code": "MIC", "name": "Micah", "testament": "OT", "order_num": 33, "chapter_count": 7, "verse_count": 105},
    {"code": "NAM", "name": "Nahum", "testament": "OT", "order_num": 34, "chapter_count": 3, "verse_count": 47},
    {"code": "HAB", "name": "Habakkuk", "testament": "OT", "order_num": 35, "chapter_count": 3, "verse_count": 56},
    {"code": "ZEP", "name": "Zephaniah", "testament": "OT", "order_num": 36, "chapter_count": 3, "verse_count": 53},
    {"code": "HAG", "name": "Haggai", "testament": "OT", "order_num": 37, "chapter_count": 2, "verse_count": 38},
    {"code": "ZEC", "name": "Zechariah", "testament": "OT", "order_num": 38, "chapter_count": 14, "verse_count": 211},
    {"code": "MAL", "name": "Malachi", "testament": "OT", "order_num": 39, "chapter_count": 4, "verse_count": 55},
    # New Testament
    {"code": "MAT", "name": "Matthew", "testament": "NT", "order_num": 40, "chapter_count": 28, "verse_count": 1071},
    {"code": "MRK", "name": "Mark", "testament": "NT", "order_num": 41, "chapter_count": 16, "verse_count": 678},
    {"code": "LUK", "name": "Luke", "testament": "NT", "order_num": 42, "chapter_count": 24, "verse_count": 1151},
    {"code": "JHN", "name": "John", "testament": "NT", "order_num": 43, "chapter_count": 21, "verse_count": 879},
    {"code": "ACT", "name": "Acts", "testament": "NT", "order_num": 44, "chapter_count": 28, "verse_count": 1007},
    {"code": "ROM", "name": "Romans", "testament": "NT", "order_num": 45, "chapter_count": 16, "verse_count": 433},
    {"code": "1CO", "name": "1 Corinthians", "testament": "NT", "order_num": 46, "chapter_count": 16, "verse_count": 437},
    {"code": "2CO", "name": "2 Corinthians", "testament": "NT", "order_num": 47, "chapter_count": 13, "verse_count": 257},
    {"code": "GAL", "name": "Galatians", "testament": "NT", "order_num": 48, "chapter_count": 6, "verse_count": 149},
    {"code": "EPH", "name": "Ephesians", "testament": "NT", "order_num": 49, "chapter_count": 6, "verse_count": 155},
    {"code": "PHP", "name": "Philippians", "testament": "NT", "order_num": 50, "chapter_count": 4, "verse_count": 104},
    {"code": "COL", "name": "Colossians", "testament": "NT", "order_num": 51, "chapter_count": 4, "verse_count": 95},
    {"code": "1TH", "name": "1 Thessalonians", "testament": "NT", "order_num": 52, "chapter_count": 5, "verse_count": 89},
    {"code": "2TH", "name": "2 Thessalonians", "testament": "NT", "order_num": 53, "chapter_count": 3, "verse_count": 47},
    {"code": "1TI", "name": "1 Timothy", "testament": "NT", "order_num": 54, "chapter_count": 6, "verse_count": 113},
    {"code": "2TI", "name": "2 Timothy", "testament": "NT", "order_num": 55, "chapter_count": 4, "verse_count": 83},
    {"code": "TIT", "name": "Titus", "testament": "NT", "order_num": 56, "chapter_count": 3, "verse_count": 46},
    {"code": "PHM", "name": "Philemon", "testament": "NT", "order_num": 57, "chapter_count": 1, "verse_count": 25},
    {"code": "HEB", "name": "Hebrews", "testament": "NT", "order_num": 58, "chapter_count": 13, "verse_count": 303},
    {"code": "JAS", "name": "James", "testament": "NT", "order_num": 59, "chapter_count": 5, "verse_count": 108},
    {"code": "1PE", "name": "1 Peter", "testament": "NT", "order_num": 60, "chapter_count": 5, "verse_count": 105},
    {"code": "2PE", "name": "2 Peter", "testament": "NT", "order_num": 61, "chapter_count": 3, "verse_count": 61},
    {"code": "1JN", "name": "1 John", "testament": "NT", "order_num": 62, "chapter_count": 5, "verse_count": 105},
    {"code": "2JN", "name": "2 John", "testament": "NT", "order_num": 63, "chapter_count": 1, "verse_count": 13},
    {"code": "3JN", "name": "3 John", "testament": "NT", "order_num": 64, "chapter_count": 1, "verse_count": 15},
    {"code": "JUD", "name": "Jude", "testament": "NT", "order_num": 65, "chapter_count": 1, "verse_count": 25},
    {"code": "REV", "name": "Revelation", "testament": "NT", "order_num": 66, "chapter_count": 22, "verse_count": 404},
]
