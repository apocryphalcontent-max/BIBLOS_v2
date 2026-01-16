"""
BIBLOS v2 - Data Module

Dataset classes, data loaders, and high-performance data processing utilities.

This module provides:
1. PyTorch-compatible datasets for ML training/inference
2. Polars/Arrow-based loaders for high-performance batch processing
3. Schema definitions and validation
4. Zero-copy IPC for inter-process communication

Architecture:
- schemas.py: Normalized dataclass schemas for system-wide uniformity
- polars_schemas.py: Polars DataFrame schemas with Arrow backend
- arrow_batch.py: High-performance batch processing with zero-copy IPC
- dataset.py: PyTorch Dataset implementations
- loaders.py: DataLoader factories with automatic backend selection
"""

# =============================================================================
# SCHEMAS - Dataclass definitions
# =============================================================================
from data.schemas import (
    # Enums
    Testament,
    Language,
    ConnectionType,
    ConnectionStrength,
    ProcessingStatus,
    CertificationLevel,
    # Base
    BaseSchema,
    # Core schemas
    VerseSchema,
    WordSchema,
    MorphologySchema,
    CrossReferenceSchema,
    # Pipeline schemas
    ExtractionResultSchema,
    GoldenRecordSchema,
    PhaseResultSchema,
    PipelineResultSchema,
    # Theological schemas
    PatristicCitationSchema,
    TypologicalConnectionSchema,
    # Inference schemas
    InferenceCandidateSchema,
    InferenceResultSchema,
    # Validation functions
    validate_verse_id,
    normalize_verse_id,
    validate_connection_type,
    validate_strength,
    # Registry
    SCHEMA_REGISTRY,
    get_schema,
    create_from_dict,
)

# =============================================================================
# DATASETS - PyTorch Dataset implementations
# =============================================================================
from data.dataset import (
    VerseRecord,
    CrossReferenceRecord,
    BibleDataset,
    CrossReferenceDataset,
    VerseDataset,
    PairDataset,
)

# =============================================================================
# LOADERS - DataLoader factories
# =============================================================================
from data.loaders import (
    # PyTorch loaders (original API)
    create_verse_loader,
    create_crossref_loader,
    create_training_loader,
    create_embedding_loader,
    # Collate functions
    verse_collate_fn,
    crossref_collate_fn,
    # Dataset splitting
    split_by_book,
    create_stratified_loader,
    # Factory
    DataLoaderFactory,
    # Adapter
    PolarsToContextAdapter,
)

# =============================================================================
# POLARS/ARROW SUPPORT (Conditional)
# =============================================================================
try:
    import polars as pl
    import pyarrow as pa
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None
    pa = None

if POLARS_AVAILABLE:
    # Polars schemas
    from data.polars_schemas import (
        PolarsSchemas,
        ArrowSchemas,
        SchemaConverter,
        DataFrameBuilder,
        LazyFrameBuilder,
        SchemaValidator,
        # Type aliases
        VerseDataFrame,
        CrossRefDataFrame,
        GoldenRecordDataFrame,
        InferenceCandidateDataFrame,
    )

    # Arrow batch processing
    from data.arrow_batch import (
        # Configuration
        BatchConfig,
        BatchMetrics,
        # IPC Reader/Writer
        ArrowIPCWriter,
        ArrowIPCReader,
        # Streaming processor
        StreamingBatchProcessor,
        # Specialized processors
        BatchVerseProcessor,
        BatchCrossRefProcessor,
        # Memory-mapped store
        MemoryMappedStore,
        # IPC channel
        IPCChannel,
        # Convenience functions
        read_verses_batch,
        write_verses_batch,
        read_crossrefs_batch,
    )

    # Polars loaders
    from data.loaders import (
        PolarsVerseLoader,
        PolarsCrossRefLoader,
        create_polars_verse_loader,
        create_polars_crossref_loader,
        scan_verses,
        scan_crossrefs,
    )

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Feature flag
    "POLARS_AVAILABLE",

    # Enums
    "Testament",
    "Language",
    "ConnectionType",
    "ConnectionStrength",
    "ProcessingStatus",
    "CertificationLevel",

    # Base schema
    "BaseSchema",

    # Core schemas
    "VerseSchema",
    "WordSchema",
    "MorphologySchema",
    "CrossReferenceSchema",

    # Pipeline schemas
    "ExtractionResultSchema",
    "GoldenRecordSchema",
    "PhaseResultSchema",
    "PipelineResultSchema",

    # Theological schemas
    "PatristicCitationSchema",
    "TypologicalConnectionSchema",

    # Inference schemas
    "InferenceCandidateSchema",
    "InferenceResultSchema",

    # Schema utilities
    "validate_verse_id",
    "normalize_verse_id",
    "validate_connection_type",
    "validate_strength",
    "SCHEMA_REGISTRY",
    "get_schema",
    "create_from_dict",

    # Dataset records
    "VerseRecord",
    "CrossReferenceRecord",

    # PyTorch datasets
    "BibleDataset",
    "CrossReferenceDataset",
    "VerseDataset",
    "PairDataset",

    # PyTorch loaders
    "create_verse_loader",
    "create_crossref_loader",
    "create_training_loader",
    "create_embedding_loader",

    # Collate functions
    "verse_collate_fn",
    "crossref_collate_fn",

    # Dataset splitting
    "split_by_book",
    "create_stratified_loader",

    # Factory and adapters
    "DataLoaderFactory",
    "PolarsToContextAdapter",
]

# Add Polars/Arrow exports if available
if POLARS_AVAILABLE:
    __all__.extend([
        # Polars schemas
        "PolarsSchemas",
        "ArrowSchemas",
        "SchemaConverter",
        "DataFrameBuilder",
        "LazyFrameBuilder",
        "SchemaValidator",

        # Type aliases
        "VerseDataFrame",
        "CrossRefDataFrame",
        "GoldenRecordDataFrame",
        "InferenceCandidateDataFrame",

        # Batch configuration
        "BatchConfig",
        "BatchMetrics",

        # IPC Reader/Writer
        "ArrowIPCWriter",
        "ArrowIPCReader",

        # Processors
        "StreamingBatchProcessor",
        "BatchVerseProcessor",
        "BatchCrossRefProcessor",

        # Memory-mapped store
        "MemoryMappedStore",

        # IPC channel
        "IPCChannel",

        # Convenience functions
        "read_verses_batch",
        "write_verses_batch",
        "read_crossrefs_batch",

        # Polars loaders
        "PolarsVerseLoader",
        "PolarsCrossRefLoader",
        "create_polars_verse_loader",
        "create_polars_crossref_loader",
        "scan_verses",
        "scan_crossrefs",
    ])


def get_polars_info() -> dict:
    """
    Get information about Polars/Arrow availability and versions.

    Returns:
        Dict with availability status and versions
    """
    info = {
        "polars_available": POLARS_AVAILABLE,
        "polars_version": None,
        "pyarrow_version": None,
    }

    if POLARS_AVAILABLE:
        info["polars_version"] = pl.__version__
        info["pyarrow_version"] = pa.__version__

    return info
