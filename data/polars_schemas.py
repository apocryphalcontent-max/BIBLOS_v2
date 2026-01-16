"""
BIBLOS v2 - Polars Schema Definitions

High-performance data schemas using Polars DataFrames with Apache Arrow backend.
Provides zero-copy data transfer between components and memory-mapped file support.

This module defines:
1. Polars schema definitions for all data types
2. Conversion utilities between dataclass <-> Polars <-> Arrow
3. Schema validation and type coercion
4. Efficient batch operations with columnar storage
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
from datetime import timezone

import polars as pl
import pyarrow as pa
try:
    import numpy as np
except ImportError:
    np = None

try:
    from data.schemas import (
    BaseSchema,
    VerseSchema,
    WordSchema,
    MorphologySchema,
    CrossReferenceSchema,
    ExtractionResultSchema,
    GoldenRecordSchema,
    PatristicTextSchema,
    PatristicCitationSchema,
    TypologicalConnectionSchema,
    PhaseResultSchema,
    PipelineResultSchema,
    InferenceCandidateSchema,
    InferenceResultSchema,
    ConnectionType,
    ConnectionStrength,
    ProcessingStatus,
    Testament,
        Language,
    )
except ImportError:
    # Provide minimal stubs if schemas not available
    from dataclasses import dataclass
    from enum import Enum

    @dataclass
    class BaseSchema:
        pass

    class ConnectionType(str, Enum):
        THEMATIC = "thematic"

    class ConnectionStrength(str, Enum):
        STRONG = "strong"

    class ProcessingStatus(str, Enum):
        PENDING = "pending"

    class Testament(str, Enum):
        OLD_TESTAMENT = "OT"

    class Language(str, Enum):
        HEBREW = "hebrew"

    VerseSchema = WordSchema = MorphologySchema = CrossReferenceSchema = BaseSchema
    ExtractionResultSchema = GoldenRecordSchema = PatristicCitationSchema = BaseSchema
    TypologicalConnectionSchema = PhaseResultSchema = PipelineResultSchema = BaseSchema
    InferenceCandidateSchema = InferenceResultSchema = BaseSchema

logger = logging.getLogger("biblos.data.polars_schemas")

T = TypeVar("T", bound=BaseSchema)


# =============================================================================
# POLARS SCHEMA DEFINITIONS
# =============================================================================

class PolarsSchemas:
    """
    Central registry of Polars schema definitions.

    Each schema maps to the corresponding dataclass schema with optimized
    Arrow/Polars data types for high-performance operations.
    """

    # Verse schema - primary data type for biblical text
    VERSE = {
        "verse_id": pl.Utf8,           # Canonical ID: BOOK.CHAPTER.VERSE
        "book": pl.Utf8,               # 3-letter book code
        "book_name": pl.Utf8,          # Full book name
        "chapter": pl.UInt16,          # Chapter number (max ~150)
        "verse": pl.UInt16,            # Verse number (max ~176)
        "text": pl.Utf8,               # English/translated text
        "original_text": pl.Utf8,      # Hebrew/Greek/Aramaic
        "testament": pl.Categorical,   # OT or NT
        "language": pl.Categorical,    # hebrew, aramaic, greek
        "created_at": pl.Datetime,     # ISO timestamp
        "updated_at": pl.Datetime,     # ISO timestamp
        "version": pl.Utf8,            # Schema version
    }

    # Word schema - word-level linguistic analysis
    WORD = {
        "word_id": pl.Utf8,            # BOOK.CHAPTER.VERSE.POSITION
        "verse_id": pl.Utf8,           # Parent verse reference
        "surface_form": pl.Utf8,       # Word as it appears
        "lemma": pl.Utf8,              # Dictionary form
        "position": pl.UInt16,         # Position in verse (0-indexed)
        "language": pl.Categorical,    # Word's language
        "transliteration": pl.Utf8,    # Romanized form
        "gloss": pl.Utf8,              # Short translation
        "strongs": pl.Utf8,            # Strong's number
        "syntax_role": pl.Utf8,        # Syntactic function
        "clause_id": pl.Utf8,          # Clause reference
        "phrase_id": pl.Utf8,          # Phrase reference
        "created_at": pl.Datetime,
        "updated_at": pl.Datetime,
        "version": pl.Utf8,
    }

    # Morphology schema - grammatical analysis
    MORPHOLOGY = {
        "word_id": pl.Utf8,            # Foreign key to word
        "part_of_speech": pl.Categorical,
        "person": pl.Utf8,             # 1, 2, 3 or null
        "number": pl.Categorical,      # singular, plural, dual
        "gender": pl.Categorical,      # masculine, feminine, neuter
        "case_": pl.Categorical,       # nominative, accusative, etc.
        "tense": pl.Categorical,       # perfect, imperfect, etc.
        "voice": pl.Categorical,       # active, passive, middle
        "mood": pl.Categorical,        # indicative, subjunctive, etc.
        "stem": pl.Utf8,               # Verb stem (qal, niphal, etc.)
        "state": pl.Categorical,       # construct, absolute
        "pattern": pl.Utf8,            # Morphological pattern
        "raw_code": pl.Utf8,           # Original parser code
        "created_at": pl.Datetime,
        "updated_at": pl.Datetime,
        "version": pl.Utf8,
    }

    # Cross-reference schema - verse connections
    CROSS_REFERENCE = {
        "source_ref": pl.Utf8,         # Source verse ID
        "target_ref": pl.Utf8,         # Target verse ID
        "connection_type": pl.Categorical,  # ConnectionType enum
        "strength": pl.Categorical,    # ConnectionStrength enum
        "confidence": pl.Float32,      # 0.0 to 1.0
        "bidirectional": pl.Boolean,   # Is relationship bidirectional
        "verified": pl.Boolean,        # Human verified
        "patristic_support": pl.Boolean,
        "created_at": pl.Datetime,
        "updated_at": pl.Datetime,
        "version": pl.Utf8,
    }

    # Extraction result schema - agent outputs
    EXTRACTION_RESULT = {
        "agent_name": pl.Utf8,         # Agent identifier
        "extraction_type": pl.Categorical,
        "verse_id": pl.Utf8,           # Processed verse
        "status": pl.Categorical,      # ProcessingStatus enum
        "confidence": pl.Float32,
        "processing_time": pl.Float32, # Seconds
        "error": pl.Utf8,              # Error message if failed
        "created_at": pl.Datetime,
        "updated_at": pl.Datetime,
        "version": pl.Utf8,
    }

    # Golden record schema - final pipeline output
    GOLDEN_RECORD = {
        "verse_id": pl.Utf8,
        "text": pl.Utf8,
        "certification_level": pl.Categorical,  # gold, silver, bronze, provisional
        "certification_score": pl.Float32,
        "validation_passed": pl.Boolean,
        "quality_passed": pl.Boolean,
        "phases_executed": pl.List(pl.Utf8),
        "agent_count": pl.UInt8,
        "total_processing_time": pl.Float32,
        "created_at": pl.Datetime,
        "updated_at": pl.Datetime,
        "version": pl.Utf8,
    }

    # Patristic text schema - full patristic works
    PATRISTIC_TEXT = {
        "text_id": pl.Utf8,            # Unique identifier
        "author": pl.Utf8,             # Church Father name
        "author_dates": pl.Utf8,       # Birth-death years
        "title": pl.Utf8,              # Original title
        "title_english": pl.Utf8,      # English title
        "book": pl.Utf8,               # Book within work
        "chapter": pl.Utf8,            # Chapter number
        "chapter_title": pl.Utf8,      # Chapter title
        "section": pl.Utf8,            # Section within chapter
        "content": pl.Utf8,            # Raw content
        "content_clean": pl.Utf8,      # Cleaned content
        "language": pl.Categorical,    # Original language
        "translation_language": pl.Categorical,  # Translation language
        "source_collection": pl.Utf8,  # Source collection (NPNF, ANF, etc.)
        "source_file": pl.Utf8,        # Source filename
        "created_at": pl.Datetime,
        "updated_at": pl.Datetime,
        "version": pl.Utf8,
    }

    # Patristic citation schema
    PATRISTIC_CITATION = {
        "father": pl.Utf8,             # Church Father name
        "work": pl.Utf8,               # Work title
        "book": pl.Utf8,               # Book within work
        "chapter": pl.Utf8,
        "section": pl.Utf8,
        "interpretation_type": pl.Categorical,  # literal, allegorical, etc.
        "quote": pl.Utf8,              # Quoted text
        "summary": pl.Utf8,            # Summary of interpretation
        "language": pl.Categorical,    # Original language
        "created_at": pl.Datetime,
        "updated_at": pl.Datetime,
        "version": pl.Utf8,
    }

    # Typological connection schema
    TYPOLOGICAL_CONNECTION = {
        "type_ref": pl.Utf8,           # OT type verse
        "antitype_ref": pl.Utf8,       # NT antitype verse
        "type_category": pl.Categorical,  # sacrifice, exodus, etc.
        "description": pl.Utf8,
        "confidence": pl.Float32,
        "created_at": pl.Datetime,
        "updated_at": pl.Datetime,
        "version": pl.Utf8,
    }

    # Inference candidate schema - ML predictions
    INFERENCE_CANDIDATE = {
        "source_verse": pl.Utf8,
        "target_verse": pl.Utf8,
        "connection_type": pl.Categorical,
        "confidence": pl.Float32,
        "embedding_similarity": pl.Float32,
        "semantic_similarity": pl.Float32,
        "created_at": pl.Datetime,
        "updated_at": pl.Datetime,
        "version": pl.Utf8,
    }

    # Pipeline result schema - batch processing results
    PIPELINE_RESULT = {
        "verse_id": pl.Utf8,
        "status": pl.Categorical,      # completed, failed, partial
        "duration": pl.Float32,        # Seconds
        "error_count": pl.UInt16,
        "start_time": pl.Float64,      # Unix timestamp
        "end_time": pl.Float64,
        "created_at": pl.Datetime,
        "updated_at": pl.Datetime,
        "version": pl.Utf8,
    }


# =============================================================================
# ARROW SCHEMA DEFINITIONS
# =============================================================================

class ArrowSchemas:
    """
    PyArrow schema definitions for IPC and memory-mapped operations.

    These schemas enable zero-copy data sharing between processes
    and efficient serialization to Arrow IPC format.
    """

    @staticmethod
    def verse_schema() -> pa.Schema:
        """Get Arrow schema for verses."""
        return pa.schema([
            ("verse_id", pa.string()),
            ("book", pa.string()),
            ("book_name", pa.string()),
            ("chapter", pa.uint16()),
            ("verse", pa.uint16()),
            ("text", pa.string()),
            ("original_text", pa.string()),
            ("testament", pa.dictionary(pa.int8(), pa.string())),
            ("language", pa.dictionary(pa.int8(), pa.string())),
            ("created_at", pa.timestamp("us")),
            ("updated_at", pa.timestamp("us")),
            ("version", pa.string()),
        ])

    @staticmethod
    def word_schema() -> pa.Schema:
        """Get Arrow schema for words."""
        return pa.schema([
            ("word_id", pa.string()),
            ("verse_id", pa.string()),
            ("surface_form", pa.string()),
            ("lemma", pa.string()),
            ("position", pa.uint16()),
            ("language", pa.dictionary(pa.int8(), pa.string())),
            ("transliteration", pa.string()),
            ("gloss", pa.string()),
            ("strongs", pa.string()),
            ("syntax_role", pa.string()),
            ("clause_id", pa.string()),
            ("phrase_id", pa.string()),
            ("created_at", pa.timestamp("us")),
            ("updated_at", pa.timestamp("us")),
            ("version", pa.string()),
        ])

    @staticmethod
    def cross_reference_schema() -> pa.Schema:
        """Get Arrow schema for cross-references."""
        return pa.schema([
            ("source_ref", pa.string()),
            ("target_ref", pa.string()),
            ("connection_type", pa.dictionary(pa.int8(), pa.string())),
            ("strength", pa.dictionary(pa.int8(), pa.string())),
            ("confidence", pa.float32()),
            ("bidirectional", pa.bool_()),
            ("verified", pa.bool_()),
            ("patristic_support", pa.bool_()),
            ("notes", pa.list_(pa.string())),
            ("sources", pa.list_(pa.string())),
            ("created_at", pa.timestamp("us")),
            ("updated_at", pa.timestamp("us")),
            ("version", pa.string()),
        ])

    @staticmethod
    def golden_record_schema() -> pa.Schema:
        """Get Arrow schema for golden records."""
        return pa.schema([
            ("verse_id", pa.string()),
            ("text", pa.string()),
            ("certification_level", pa.dictionary(pa.int8(), pa.string())),
            ("certification_score", pa.float32()),
            ("validation_passed", pa.bool_()),
            ("quality_passed", pa.bool_()),
            ("phases_executed", pa.list_(pa.string())),
            ("agent_count", pa.uint8()),
            ("total_processing_time", pa.float32()),
            ("created_at", pa.timestamp("us")),
            ("updated_at", pa.timestamp("us")),
            ("version", pa.string()),
        ])

    @staticmethod
    def inference_candidate_schema() -> pa.Schema:
        """Get Arrow schema for inference candidates."""
        return pa.schema([
            ("source_verse", pa.string()),
            ("target_verse", pa.string()),
            ("connection_type", pa.dictionary(pa.int8(), pa.string())),
            ("confidence", pa.float32()),
            ("embedding_similarity", pa.float32()),
            ("semantic_similarity", pa.float32()),
            ("evidence", pa.list_(pa.string())),
            ("created_at", pa.timestamp("us")),
            ("updated_at", pa.timestamp("us")),
            ("version", pa.string()),
        ])


# =============================================================================
# SCHEMA CONVERSION UTILITIES
# =============================================================================

class SchemaConverter:
    """
    Bidirectional converter between dataclass schemas and Polars DataFrames.

    Provides efficient conversion with proper type coercion and validation.
    """

    # Map dataclass types to Polars schemas
    SCHEMA_MAP: Dict[Type[BaseSchema], Dict[str, pl.DataType]] = {
        VerseSchema: PolarsSchemas.VERSE,
        WordSchema: PolarsSchemas.WORD,
        MorphologySchema: PolarsSchemas.MORPHOLOGY,
        CrossReferenceSchema: PolarsSchemas.CROSS_REFERENCE,
        ExtractionResultSchema: PolarsSchemas.EXTRACTION_RESULT,
        GoldenRecordSchema: PolarsSchemas.GOLDEN_RECORD,
        PatristicCitationSchema: PolarsSchemas.PATRISTIC_CITATION,
        TypologicalConnectionSchema: PolarsSchemas.TYPOLOGICAL_CONNECTION,
        InferenceCandidateSchema: PolarsSchemas.INFERENCE_CANDIDATE,
        PipelineResultSchema: PolarsSchemas.PIPELINE_RESULT,
    }

    # Map dataclass types to Arrow schemas
    ARROW_SCHEMA_MAP: Dict[Type[BaseSchema], Callable[[], pa.Schema]] = {
        VerseSchema: ArrowSchemas.verse_schema,
        WordSchema: ArrowSchemas.word_schema,
        CrossReferenceSchema: ArrowSchemas.cross_reference_schema,
        GoldenRecordSchema: ArrowSchemas.golden_record_schema,
        InferenceCandidateSchema: ArrowSchemas.inference_candidate_schema,
    }

    @classmethod
    def dataclass_to_polars(
        cls,
        items: Union[T, List[T]],
        schema_type: Optional[Type[T]] = None
    ) -> pl.DataFrame:
        """
        Convert dataclass instance(s) to Polars DataFrame.

        Args:
            items: Single dataclass instance or list of instances
            schema_type: Optional schema type hint for empty lists

        Returns:
            Polars DataFrame with appropriate schema

        Example:
            >>> verses = [VerseSchema(verse_id="GEN.1.1", ...)]
            >>> df = SchemaConverter.dataclass_to_polars(verses)
        """
        if not isinstance(items, list):
            items = [items]

        if not items:
            if schema_type is None:
                return pl.DataFrame()
            polars_schema = cls.SCHEMA_MAP.get(schema_type, {})
            return pl.DataFrame(schema=polars_schema)

        # Infer schema from first item
        item_type = type(items[0])
        polars_schema = cls.SCHEMA_MAP.get(item_type)

        # Convert to list of dicts
        data = []
        for item in items:
            item_dict = cls._flatten_dataclass(item)
            data.append(item_dict)

        # Create DataFrame
        df = pl.DataFrame(data)

        # Apply schema casting if available
        if polars_schema:
            df = cls._cast_to_schema(df, polars_schema)

        return df

    @classmethod
    def polars_to_dataclass(
        cls,
        df: pl.DataFrame,
        schema_type: Type[T]
    ) -> List[T]:
        """
        Convert Polars DataFrame to list of dataclass instances.

        Args:
            df: Polars DataFrame
            schema_type: Target dataclass type

        Returns:
            List of dataclass instances

        Example:
            >>> df = pl.DataFrame({"verse_id": ["GEN.1.1"], ...})
            >>> verses = SchemaConverter.polars_to_dataclass(df, VerseSchema)
        """
        if df.is_empty():
            return []

        results = []
        for row in df.iter_rows(named=True):
            # Handle nested fields
            row_dict = cls._unflatten_dict(row, schema_type)
            # Convert datetime columns
            row_dict = cls._convert_timestamps(row_dict)
            try:
                instance = schema_type(**row_dict)
                results.append(instance)
            except Exception as e:
                logger.warning(f"Failed to create {schema_type.__name__}: {e}")
                continue

        return results

    @classmethod
    def polars_to_arrow(cls, df: pl.DataFrame) -> pa.Table:
        """
        Convert Polars DataFrame to Arrow Table (zero-copy when possible).

        Args:
            df: Polars DataFrame

        Returns:
            PyArrow Table
        """
        return df.to_arrow()

    @classmethod
    def arrow_to_polars(cls, table: pa.Table) -> pl.DataFrame:
        """
        Convert Arrow Table to Polars DataFrame (zero-copy).

        Args:
            table: PyArrow Table

        Returns:
            Polars DataFrame
        """
        return pl.from_arrow(table)

    @classmethod
    def dataclass_to_arrow(
        cls,
        items: Union[T, List[T]],
        schema_type: Optional[Type[T]] = None
    ) -> pa.Table:
        """
        Convert dataclass instance(s) directly to Arrow Table.

        Args:
            items: Single dataclass instance or list of instances
            schema_type: Optional schema type hint

        Returns:
            PyArrow Table
        """
        df = cls.dataclass_to_polars(items, schema_type)
        return df.to_arrow()

    @classmethod
    def arrow_to_dataclass(
        cls,
        table: pa.Table,
        schema_type: Type[T]
    ) -> List[T]:
        """
        Convert Arrow Table to list of dataclass instances.

        Args:
            table: PyArrow Table
            schema_type: Target dataclass type

        Returns:
            List of dataclass instances
        """
        df = pl.from_arrow(table)
        return cls.polars_to_dataclass(df, schema_type)

    @staticmethod
    def _flatten_dataclass(item: BaseSchema) -> Dict[str, Any]:
        """Flatten nested dataclass to dict with prefixed keys."""
        result = {}
        item_dict = asdict(item)

        for key, value in item_dict.items():
            if isinstance(value, dict):
                # Flatten nested dicts (like metadata, morphology)
                for nested_key, nested_value in value.items():
                    result[f"{key}_{nested_key}"] = nested_value
            elif isinstance(value, list):
                # Keep lists as-is for List columns
                result[key] = value
            elif isinstance(value, datetime):
                result[key] = value
            else:
                result[key] = value

        return result

    @staticmethod
    def _unflatten_dict(
        flat_dict: Dict[str, Any],
        schema_type: Type[BaseSchema]
    ) -> Dict[str, Any]:
        """Unflatten dict back to nested structure for dataclass."""
        result = {}
        nested_fields = {}

        # Get the dataclass fields to know what's nested
        import dataclasses
        if not dataclasses.is_dataclass(schema_type):
            return flat_dict

        field_types = {f.name: f.type for f in dataclasses.fields(schema_type)}

        for key, value in flat_dict.items():
            if "_" in key:
                # Check if this is a nested field
                prefix, suffix = key.split("_", 1)
                if prefix in field_types and "Dict" in str(field_types.get(prefix, "")):
                    if prefix not in nested_fields:
                        nested_fields[prefix] = {}
                    nested_fields[prefix][suffix] = value
                else:
                    result[key] = value
            else:
                result[key] = value

        # Merge nested fields back
        result.update(nested_fields)
        return result

    @staticmethod
    def _convert_timestamps(row_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert datetime objects to ISO strings for dataclass compatibility."""
        for key, value in row_dict.items():
            if isinstance(value, datetime):
                row_dict[key] = value.isoformat()
        return row_dict

    @staticmethod
    def _cast_to_schema(
        df: pl.DataFrame,
        schema: Dict[str, pl.DataType]
    ) -> pl.DataFrame:
        """Cast DataFrame columns to match schema types."""
        existing_cols = set(df.columns)

        casts = []
        for col, dtype in schema.items():
            if col in existing_cols:
                try:
                    casts.append(pl.col(col).cast(dtype))
                except Exception:
                    # Keep original if cast fails
                    casts.append(pl.col(col))

        if casts:
            return df.select(casts)
        return df


# =============================================================================
# POLARS DATAFRAME BUILDERS
# =============================================================================

class DataFrameBuilder:
    """
    Factory methods for creating optimized Polars DataFrames.

    Provides builders for each data type with proper schema enforcement
    and memory-efficient construction.
    """

    @staticmethod
    def verses(
        data: Union[List[Dict], List[VerseSchema], pl.DataFrame]
    ) -> pl.DataFrame:
        """
        Build a verses DataFrame with optimized schema.

        Args:
            data: List of dicts, VerseSchema instances, or existing DataFrame

        Returns:
            Polars DataFrame with verse schema
        """
        if isinstance(data, pl.DataFrame):
            return DataFrameBuilder._ensure_verse_schema(data)

        if data and isinstance(data[0], VerseSchema):
            return SchemaConverter.dataclass_to_polars(data, VerseSchema)

        # From list of dicts
        df = pl.DataFrame(data)
        return DataFrameBuilder._ensure_verse_schema(df)

    @staticmethod
    def _ensure_verse_schema(df: pl.DataFrame) -> pl.DataFrame:
        """Ensure DataFrame has correct verse schema with defaults."""
        now = datetime.now(timezone.utc)

        # Add missing columns with defaults
        if "created_at" not in df.columns:
            df = df.with_columns(pl.lit(now).alias("created_at"))
        if "updated_at" not in df.columns:
            df = df.with_columns(pl.lit(now).alias("updated_at"))
        if "version" not in df.columns:
            df = df.with_columns(pl.lit("2.0.0").alias("version"))

        return df

    @staticmethod
    def cross_references(
        data: Union[List[Dict], List[CrossReferenceSchema], pl.DataFrame]
    ) -> pl.DataFrame:
        """
        Build a cross-references DataFrame with optimized schema.

        Args:
            data: List of dicts, CrossReferenceSchema instances, or existing DataFrame

        Returns:
            Polars DataFrame with cross-reference schema
        """
        if isinstance(data, pl.DataFrame):
            return DataFrameBuilder._ensure_crossref_schema(data)

        if data and isinstance(data[0], CrossReferenceSchema):
            return SchemaConverter.dataclass_to_polars(data, CrossReferenceSchema)

        df = pl.DataFrame(data)
        return DataFrameBuilder._ensure_crossref_schema(df)

    @staticmethod
    def _ensure_crossref_schema(df: pl.DataFrame) -> pl.DataFrame:
        """Ensure DataFrame has correct cross-reference schema."""
        now = datetime.now(timezone.utc)

        # Add missing columns with defaults
        if "created_at" not in df.columns:
            df = df.with_columns(pl.lit(now).alias("created_at"))
        if "updated_at" not in df.columns:
            df = df.with_columns(pl.lit(now).alias("updated_at"))
        if "version" not in df.columns:
            df = df.with_columns(pl.lit("2.0.0").alias("version"))
        if "confidence" not in df.columns:
            df = df.with_columns(pl.lit(1.0).cast(pl.Float32).alias("confidence"))
        if "bidirectional" not in df.columns:
            df = df.with_columns(pl.lit(False).alias("bidirectional"))
        if "verified" not in df.columns:
            df = df.with_columns(pl.lit(False).alias("verified"))

        return df

    @staticmethod
    def golden_records(
        data: Union[List[Dict], List[GoldenRecordSchema], pl.DataFrame]
    ) -> pl.DataFrame:
        """
        Build a golden records DataFrame with optimized schema.

        Args:
            data: List of dicts, GoldenRecordSchema instances, or existing DataFrame

        Returns:
            Polars DataFrame with golden record schema
        """
        if isinstance(data, pl.DataFrame):
            return data

        if data and isinstance(data[0], GoldenRecordSchema):
            return SchemaConverter.dataclass_to_polars(data, GoldenRecordSchema)

        return pl.DataFrame(data)

    @staticmethod
    def inference_candidates(
        data: Union[List[Dict], List[InferenceCandidateSchema], pl.DataFrame]
    ) -> pl.DataFrame:
        """
        Build an inference candidates DataFrame with optimized schema.

        Args:
            data: List of dicts, InferenceCandidateSchema instances, or DataFrame

        Returns:
            Polars DataFrame with inference candidate schema
        """
        if isinstance(data, pl.DataFrame):
            return data

        if data and isinstance(data[0], InferenceCandidateSchema):
            return SchemaConverter.dataclass_to_polars(data, InferenceCandidateSchema)

        return pl.DataFrame(data)

    @staticmethod
    def empty(schema_type: Type[BaseSchema]) -> pl.DataFrame:
        """
        Create empty DataFrame with correct schema.

        Args:
            schema_type: Dataclass type defining the schema

        Returns:
            Empty Polars DataFrame with correct column types
        """
        polars_schema = SchemaConverter.SCHEMA_MAP.get(schema_type)
        if polars_schema:
            return pl.DataFrame(schema=polars_schema)
        return pl.DataFrame()


# =============================================================================
# LAZY FRAME BUILDERS
# =============================================================================

class LazyFrameBuilder:
    """
    Factory methods for creating Polars LazyFrames for deferred execution.

    LazyFrames enable query optimization and memory-efficient processing
    of large datasets through lazy evaluation.
    """

    @staticmethod
    def from_json_files(
        paths: List[str],
        schema: Optional[Dict[str, pl.DataType]] = None
    ) -> pl.LazyFrame:
        """
        Create a LazyFrame from multiple JSON files.

        Args:
            paths: List of JSON file paths
            schema: Optional schema for type inference

        Returns:
            Polars LazyFrame for lazy evaluation
        """
        frames = []
        for path in paths:
            try:
                lf = pl.scan_ndjson(path)
                frames.append(lf)
            except Exception:
                # Try as regular JSON
                try:
                    df = pl.read_json(path)
                    frames.append(df.lazy())
                except Exception as e:
                    logger.warning(f"Failed to read {path}: {e}")
                    continue

        if not frames:
            if schema:
                return pl.DataFrame(schema=schema).lazy()
            return pl.DataFrame().lazy()

        return pl.concat(frames)

    @staticmethod
    def from_parquet_files(
        paths: List[str],
        columns: Optional[List[str]] = None
    ) -> pl.LazyFrame:
        """
        Create a LazyFrame from Parquet files (memory-mapped).

        Args:
            paths: List of Parquet file paths
            columns: Optional columns to select (projection pushdown)

        Returns:
            Polars LazyFrame with memory-mapped access
        """
        frames = []
        for path in paths:
            try:
                lf = pl.scan_parquet(path)
                if columns:
                    lf = lf.select(columns)
                frames.append(lf)
            except Exception as e:
                logger.warning(f"Failed to scan {path}: {e}")
                continue

        if not frames:
            return pl.DataFrame().lazy()

        return pl.concat(frames)

    @staticmethod
    def from_ipc_files(
        paths: List[str],
        columns: Optional[List[str]] = None,
        memory_map: bool = True
    ) -> pl.LazyFrame:
        """
        Create a LazyFrame from Arrow IPC files.

        Args:
            paths: List of IPC file paths
            columns: Optional columns to select
            memory_map: Whether to memory-map the files

        Returns:
            Polars LazyFrame with zero-copy access
        """
        frames = []
        for path in paths:
            try:
                lf = pl.scan_ipc(path, memory_map=memory_map)
                if columns:
                    lf = lf.select(columns)
                frames.append(lf)
            except Exception as e:
                logger.warning(f"Failed to scan IPC {path}: {e}")
                continue

        if not frames:
            return pl.DataFrame().lazy()

        return pl.concat(frames)


# =============================================================================
# SCHEMA VALIDATION
# =============================================================================

class SchemaValidator:
    """
    Validation utilities for Polars DataFrames against expected schemas.
    """

    @staticmethod
    def validate_verse_df(df: pl.DataFrame) -> List[str]:
        """
        Validate a verses DataFrame.

        Args:
            df: DataFrame to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        required_cols = {"verse_id", "book", "chapter", "verse", "text"}
        missing = required_cols - set(df.columns)
        if missing:
            errors.append(f"Missing required columns: {missing}")

        if "verse_id" in df.columns:
            null_ids = df.filter(pl.col("verse_id").is_null()).height
            if null_ids > 0:
                errors.append(f"Found {null_ids} rows with null verse_id")

        if "chapter" in df.columns:
            invalid_chapters = df.filter(pl.col("chapter") < 1).height
            if invalid_chapters > 0:
                errors.append(f"Found {invalid_chapters} rows with invalid chapter")

        if "verse" in df.columns:
            invalid_verses = df.filter(pl.col("verse") < 1).height
            if invalid_verses > 0:
                errors.append(f"Found {invalid_verses} rows with invalid verse number")

        return errors

    @staticmethod
    def validate_crossref_df(df: pl.DataFrame) -> List[str]:
        """
        Validate a cross-references DataFrame.

        Args:
            df: DataFrame to validate

        Returns:
            List of validation error messages
        """
        errors = []

        required_cols = {"source_ref", "target_ref", "connection_type"}
        missing = required_cols - set(df.columns)
        if missing:
            errors.append(f"Missing required columns: {missing}")

        if "source_ref" in df.columns:
            null_sources = df.filter(pl.col("source_ref").is_null()).height
            if null_sources > 0:
                errors.append(f"Found {null_sources} rows with null source_ref")

        if "target_ref" in df.columns:
            null_targets = df.filter(pl.col("target_ref").is_null()).height
            if null_targets > 0:
                errors.append(f"Found {null_targets} rows with null target_ref")

        if "confidence" in df.columns:
            invalid_conf = df.filter(
                (pl.col("confidence") < 0) | (pl.col("confidence") > 1)
            ).height
            if invalid_conf > 0:
                errors.append(f"Found {invalid_conf} rows with invalid confidence")

        # Validate connection types
        valid_types = {e.value for e in ConnectionType}
        if "connection_type" in df.columns:
            unique_types = set(df["connection_type"].unique().to_list())
            invalid_types = unique_types - valid_types - {None}
            if invalid_types:
                errors.append(f"Invalid connection types: {invalid_types}")

        return errors

    @staticmethod
    def validate_golden_record_df(df: pl.DataFrame) -> List[str]:
        """
        Validate a golden records DataFrame.

        Args:
            df: DataFrame to validate

        Returns:
            List of validation error messages
        """
        errors = []

        required_cols = {"verse_id", "text"}
        missing = required_cols - set(df.columns)
        if missing:
            errors.append(f"Missing required columns: {missing}")

        if "certification_score" in df.columns:
            invalid_scores = df.filter(
                (pl.col("certification_score") < 0) |
                (pl.col("certification_score") > 1)
            ).height
            if invalid_scores > 0:
                errors.append(f"Found {invalid_scores} rows with invalid certification_score")

        return errors


# =============================================================================
# TYPE ALIASES
# =============================================================================

# Convenient type aliases for function signatures
VerseDataFrame = pl.DataFrame
CrossRefDataFrame = pl.DataFrame
GoldenRecordDataFrame = pl.DataFrame
InferenceCandidateDataFrame = pl.DataFrame

# Export all public classes and functions
__all__ = [
    # Schema definitions
    "PolarsSchemas",
    "ArrowSchemas",
    # Converters
    "SchemaConverter",
    # Builders
    "DataFrameBuilder",
    "LazyFrameBuilder",
    # Validators
    "SchemaValidator",
    # Type aliases
    "VerseDataFrame",
    "CrossRefDataFrame",
    "GoldenRecordDataFrame",
    "InferenceCandidateDataFrame",
]
