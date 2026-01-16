"""
BIBLOS v2 - Data Loaders

Factory functions for creating data loaders with support for both:
1. PyTorch DataLoader (for ML training/inference)
2. Polars LazyFrame/DataFrame (for high-performance batch processing)

This module provides backward-compatible loaders that work with the existing
PyTorch-based pipeline while enabling migration to the new Polars/Arrow
data plane for improved performance.
"""
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import torch
from torch.utils.data import DataLoader, Subset, random_split

try:
    import polars as pl
    import pyarrow as pa
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None
    pa = None

from data.dataset import (
    BibleDataset,
    CrossReferenceDataset,
    VerseDataset,
    PairDataset,
)

# Conditional imports for new Polars functionality
if POLARS_AVAILABLE:
    from data.polars_schemas import (
        SchemaConverter,
        DataFrameBuilder,
        LazyFrameBuilder,
        PolarsSchemas,
    )
    from data.arrow_batch import (
        BatchConfig,
        BatchVerseProcessor,
        BatchCrossRefProcessor,
        ArrowIPCReader,
    )


# =============================================================================
# PYTORCH DATALOADERS (Original API - Preserved for Backward Compatibility)
# =============================================================================

def create_verse_loader(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    books: Optional[List[str]] = None,
    transform: Optional[Callable] = None
) -> DataLoader:
    """
    Create a PyTorch DataLoader for verse data.

    Args:
        data_dir: Path to data directory
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        books: Optional list of book codes to include
        transform: Optional transform function

    Returns:
        DataLoader for verses
    """
    dataset = BibleDataset(
        data_dir=data_dir,
        books=books,
        transform=transform
    )
    dataset.load()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=verse_collate_fn
    )


def create_crossref_loader(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    connection_types: Optional[List[str]] = None,
    min_confidence: float = 0.0
) -> DataLoader:
    """
    Create a PyTorch DataLoader for cross-reference data.

    Args:
        data_dir: Path to data directory
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        connection_types: Optional list of connection types to include
        min_confidence: Minimum confidence threshold

    Returns:
        DataLoader for cross-references
    """
    dataset = CrossReferenceDataset(
        data_dir=data_dir,
        connection_types=connection_types,
        min_confidence=min_confidence
    )
    dataset.load()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=crossref_collate_fn
    )


def create_training_loader(
    positive_pairs: List[Tuple[str, str, str]],
    verse_texts: Dict[str, str],
    batch_size: int = 32,
    negative_ratio: float = 1.0,
    tokenizer: Optional[Callable] = None,
    max_length: int = 512,
    num_workers: int = 0,
    train_split: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders for pair classification.

    Args:
        positive_pairs: List of (source, target, type) tuples
        verse_texts: Dict mapping verse_id to text
        batch_size: Batch size
        negative_ratio: Ratio of negative to positive samples
        tokenizer: Optional tokenizer
        max_length: Maximum sequence length
        num_workers: Number of worker processes
        train_split: Fraction for training set

    Returns:
        Tuple of (train_loader, val_loader)
    """
    dataset = PairDataset(
        positive_pairs=positive_pairs,
        verse_texts=verse_texts,
        negative_ratio=negative_ratio,
        tokenizer=tokenizer,
        max_length=max_length
    )

    # Split dataset
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader


def create_embedding_loader(
    verses: List[Dict[str, str]],
    tokenizer: Callable,
    batch_size: int = 32,
    max_length: int = 512,
    num_workers: int = 0
) -> DataLoader:
    """
    Create a DataLoader for embedding generation.

    Args:
        verses: List of verse dicts with "verse_id" and "text"
        tokenizer: Tokenizer for encoding
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of worker processes

    Returns:
        DataLoader for embedding generation
    """
    dataset = VerseDataset(
        verses=verses,
        tokenizer=tokenizer,
        max_length=max_length
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )


# =============================================================================
# COLLATE FUNCTIONS
# =============================================================================

def verse_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for verse batches."""
    return {
        "verse_ids": [item["verse_id"] for item in batch],
        "books": [item["book"] for item in batch],
        "chapters": [item["chapter"] for item in batch],
        "verses": [item["verse"] for item in batch],
        "texts": [item["text"] for item in batch],
        "languages": [item["language"] for item in batch]
    }


def crossref_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for cross-reference batches."""
    return {
        "source_refs": [item["source_ref"] for item in batch],
        "target_refs": [item["target_ref"] for item in batch],
        "connection_types": [item["connection_type"] for item in batch],
        "strengths": [item["strength"] for item in batch],
        "confidences": torch.tensor([item["confidence"] for item in batch]),
        "labels": torch.tensor([item["label"] for item in batch])
    }


# =============================================================================
# DATASET SPLITTING UTILITIES
# =============================================================================

def split_by_book(
    dataset: BibleDataset,
    train_books: List[str],
    val_books: List[str],
    test_books: Optional[List[str]] = None
) -> Tuple[Subset, Subset, Optional[Subset]]:
    """
    Split dataset by book for train/val/test.

    Args:
        dataset: BibleDataset to split
        train_books: Books for training
        val_books: Books for validation
        test_books: Optional books for testing

    Returns:
        Tuple of (train_subset, val_subset, test_subset)
    """
    train_indices = []
    val_indices = []
    test_indices = []

    for i, verse in enumerate(dataset._verses):
        if verse.book in train_books:
            train_indices.append(i)
        elif verse.book in val_books:
            val_indices.append(i)
        elif test_books and verse.book in test_books:
            test_indices.append(i)

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices) if test_indices else None

    return train_subset, val_subset, test_subset


def create_stratified_loader(
    dataset: CrossReferenceDataset,
    batch_size: int = 32,
    stratify_by: str = "connection_type"
) -> DataLoader:
    """
    Create a stratified DataLoader ensuring each batch has balanced types.

    Args:
        dataset: CrossReferenceDataset
        batch_size: Batch size
        stratify_by: Field to stratify by

    Returns:
        DataLoader with stratified batching
    """
    from torch.utils.data import WeightedRandomSampler

    # Count samples per type
    type_counts = {}
    for ref in dataset._references:
        conn_type = ref.connection_type
        type_counts[conn_type] = type_counts.get(conn_type, 0) + 1

    # Calculate weights
    weights = []
    for ref in dataset._references:
        conn_type = ref.connection_type
        weight = 1.0 / type_counts[conn_type]
        weights.append(weight)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=crossref_collate_fn
    )


# =============================================================================
# POLARS LOADERS (New High-Performance API)
# =============================================================================

if POLARS_AVAILABLE:

    class PolarsVerseLoader:
        """
        High-performance verse loader using Polars DataFrames.

        Supports lazy evaluation, memory-mapped files, and streaming
        for efficient processing of large datasets.
        """

        def __init__(
            self,
            data_dir: Union[str, Path],
            batch_size: int = 1000,
            books: Optional[List[str]] = None,
            columns: Optional[List[str]] = None,
            memory_map: bool = True,
        ):
            """
            Initialize Polars verse loader.

            Args:
                data_dir: Path to data directory
                batch_size: Batch size for iteration
                books: Optional list of book codes to include
                columns: Optional columns to select (projection pushdown)
                memory_map: Whether to use memory-mapped I/O
            """
            self.data_dir = Path(data_dir)
            self.batch_size = batch_size
            self.books = books
            self.columns = columns
            self.memory_map = memory_map
            self._lazy_frame: Optional[pl.LazyFrame] = None
            self._processor = BatchVerseProcessor(
                BatchConfig(batch_size=batch_size, memory_map=memory_map)
            )

        def scan(self) -> "pl.LazyFrame":
            """
            Create a lazy scan of all verse files.

            Returns:
                Polars LazyFrame for deferred execution
            """
            if self._lazy_frame is not None:
                return self._lazy_frame

            # Find all supported files
            parquet_files = list(self.data_dir.glob("**/*.parquet"))
            ipc_files = list(self.data_dir.glob("**/*.arrow")) + \
                       list(self.data_dir.glob("**/*.ipc"))
            json_files = list(self.data_dir.glob("**/*.json"))

            frames = []

            # Scan parquet files (most efficient)
            for path in parquet_files:
                lf = pl.scan_parquet(path)
                if self.columns:
                    lf = lf.select(self.columns)
                frames.append(lf)

            # Scan IPC files
            for path in ipc_files:
                lf = pl.scan_ipc(path, memory_map=self.memory_map)
                if self.columns:
                    lf = lf.select(self.columns)
                frames.append(lf)

            # Scan JSON files (less efficient but supported)
            for path in json_files:
                try:
                    df = pl.read_json(path)
                    if self.columns:
                        available_cols = [c for c in self.columns if c in df.columns]
                        if available_cols:
                            df = df.select(available_cols)
                    frames.append(df.lazy())
                except Exception:
                    continue

            if not frames:
                self._lazy_frame = pl.DataFrame(schema=PolarsSchemas.VERSE).lazy()
            else:
                self._lazy_frame = pl.concat(frames)

            # Apply book filter if specified
            if self.books:
                self._lazy_frame = self._lazy_frame.filter(
                    pl.col("book").is_in(self.books)
                )

            return self._lazy_frame

        def collect(self, streaming: bool = True) -> "pl.DataFrame":
            """
            Collect all data into a DataFrame.

            Args:
                streaming: Use streaming execution for memory efficiency

            Returns:
                Polars DataFrame with all verse data
            """
            lf = self.scan()
            return lf.collect(streaming=streaming)

        def iter_batches(self) -> Iterator["pl.DataFrame"]:
            """
            Iterate over verses in batches.

            Yields:
                Polars DataFrame for each batch
            """
            df = self.collect()
            for batch in df.iter_slices(n_rows=self.batch_size):
                yield batch

        def to_pytorch_loader(
            self,
            tokenizer: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            num_workers: int = 0,
        ) -> DataLoader:
            """
            Convert to PyTorch DataLoader for ML training.

            Args:
                tokenizer: Optional tokenizer for encoding
                transform: Optional transform function
                num_workers: Number of worker processes

            Returns:
                PyTorch DataLoader
            """
            df = self.collect()

            # Convert to list of dicts for VerseDataset
            verses = df.to_dicts()

            if tokenizer:
                dataset = VerseDataset(
                    verses=verses,
                    tokenizer=tokenizer,
                )
            else:
                # Use BibleDataset-like wrapper
                dataset = _PolarsDatasetWrapper(df, transform)

            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=verse_collate_fn
            )

        def __len__(self) -> int:
            """Get total row count (requires collection)."""
            return self.scan().select(pl.count()).collect().item()


    class PolarsCrossRefLoader:
        """
        High-performance cross-reference loader using Polars DataFrames.
        """

        def __init__(
            self,
            data_dir: Union[str, Path],
            batch_size: int = 1000,
            connection_types: Optional[List[str]] = None,
            min_confidence: float = 0.0,
            memory_map: bool = True,
        ):
            """
            Initialize Polars cross-reference loader.

            Args:
                data_dir: Path to data directory
                batch_size: Batch size for iteration
                connection_types: Optional list of connection types
                min_confidence: Minimum confidence threshold
                memory_map: Whether to use memory-mapped I/O
            """
            self.data_dir = Path(data_dir)
            self.batch_size = batch_size
            self.connection_types = connection_types
            self.min_confidence = min_confidence
            self.memory_map = memory_map
            self._lazy_frame: Optional[pl.LazyFrame] = None
            self._processor = BatchCrossRefProcessor(
                BatchConfig(batch_size=batch_size, memory_map=memory_map)
            )

        def scan(self) -> "pl.LazyFrame":
            """
            Create a lazy scan of all cross-reference files.

            Returns:
                Polars LazyFrame for deferred execution
            """
            if self._lazy_frame is not None:
                return self._lazy_frame

            # Find all supported files
            parquet_files = list(self.data_dir.glob("**/*.parquet"))
            ipc_files = list(self.data_dir.glob("**/*.arrow")) + \
                       list(self.data_dir.glob("**/*.ipc"))
            json_files = list(self.data_dir.glob("**/*.json"))

            frames = []

            for path in parquet_files:
                frames.append(pl.scan_parquet(path))

            for path in ipc_files:
                frames.append(pl.scan_ipc(path, memory_map=self.memory_map))

            for path in json_files:
                try:
                    df = pl.read_json(path)
                    frames.append(df.lazy())
                except Exception:
                    continue

            if not frames:
                self._lazy_frame = pl.DataFrame(
                    schema=PolarsSchemas.CROSS_REFERENCE
                ).lazy()
            else:
                self._lazy_frame = pl.concat(frames)

            # Apply filters
            if self.connection_types:
                self._lazy_frame = self._lazy_frame.filter(
                    pl.col("connection_type").is_in(self.connection_types)
                )

            if self.min_confidence > 0:
                self._lazy_frame = self._lazy_frame.filter(
                    pl.col("confidence") >= self.min_confidence
                )

            return self._lazy_frame

        def collect(self, streaming: bool = True) -> "pl.DataFrame":
            """
            Collect all data into a DataFrame.

            Args:
                streaming: Use streaming execution for memory efficiency

            Returns:
                Polars DataFrame with all cross-reference data
            """
            lf = self.scan()
            return lf.collect(streaming=streaming)

        def iter_batches(self) -> Iterator["pl.DataFrame"]:
            """
            Iterate over cross-references in batches.

            Yields:
                Polars DataFrame for each batch
            """
            df = self.collect()
            for batch in df.iter_slices(n_rows=self.batch_size):
                yield batch

        def get_by_source(self, verse_id: str) -> "pl.DataFrame":
            """
            Get all cross-references from a source verse.

            Args:
                verse_id: Source verse ID

            Returns:
                DataFrame of cross-references from this verse
            """
            return self.collect().filter(
                pl.col("source_ref") == verse_id.upper()
            )

        def get_by_target(self, verse_id: str) -> "pl.DataFrame":
            """
            Get all cross-references to a target verse.

            Args:
                verse_id: Target verse ID

            Returns:
                DataFrame of cross-references to this verse
            """
            return self.collect().filter(
                pl.col("target_ref") == verse_id.upper()
            )

        def to_pytorch_loader(
            self,
            num_workers: int = 0,
        ) -> DataLoader:
            """
            Convert to PyTorch DataLoader for ML training.

            Args:
                num_workers: Number of worker processes

            Returns:
                PyTorch DataLoader
            """
            df = self.collect()
            dataset = _PolarsCrossRefDatasetWrapper(df)

            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=crossref_collate_fn
            )

        def __len__(self) -> int:
            """Get total row count (requires collection)."""
            return self.scan().select(pl.count()).collect().item()


    class PolarsPatristicLoader:
        """
        High-performance patristic text loader using Polars DataFrames.

        Loads normalized patristic text JSON files for integration with the
        BIBLOS data pipeline and arrow batch processing.
        """

        def __init__(
            self,
            data_dir: Union[str, Path],
            batch_size: int = 100,
            authors: Optional[List[str]] = None,
            memory_map: bool = True,
        ):
            """
            Initialize Polars patristic loader.

            Args:
                data_dir: Path to patristics data directory
                batch_size: Batch size for iteration
                authors: Optional list of authors to filter
                memory_map: Whether to use memory-mapped I/O
            """
            self.data_dir = Path(data_dir)
            self.batch_size = batch_size
            self.authors = authors
            self.memory_map = memory_map
            self._lazy_frame: Optional[pl.LazyFrame] = None

        def scan(self) -> "pl.LazyFrame":
            """
            Create a lazy scan of patristic text files.

            Returns:
                Polars LazyFrame for deferred execution
            """
            if self._lazy_frame is not None:
                return self._lazy_frame

            # Find all supported files
            json_files = list(self.data_dir.glob("**/*.json"))
            parquet_files = list(self.data_dir.glob("**/*.parquet"))

            frames = []

            # Scan parquet files (most efficient)
            for path in parquet_files:
                frames.append(pl.scan_parquet(path))

            # Scan JSON files
            for path in json_files:
                if "index" in path.name.lower():
                    continue  # Skip index files
                try:
                    df = pl.read_json(path)
                    frames.append(df.lazy())
                except Exception:
                    continue

            if not frames:
                self._lazy_frame = pl.DataFrame(
                    schema=PolarsSchemas.PATRISTIC_TEXT
                ).lazy()
            else:
                self._lazy_frame = pl.concat(frames)

            # Apply author filter if specified
            if self.authors:
                self._lazy_frame = self._lazy_frame.filter(
                    pl.col("author").is_in(self.authors)
                )

            return self._lazy_frame

        def collect(self, streaming: bool = True) -> "pl.DataFrame":
            """
            Collect all data into a DataFrame.

            Args:
                streaming: Use streaming execution for memory efficiency

            Returns:
                Polars DataFrame with all patristic text data
            """
            lf = self.scan()
            return lf.collect(streaming=streaming)

        def iter_batches(self) -> Iterator["pl.DataFrame"]:
            """
            Iterate over patristic texts in batches.

            Yields:
                Polars DataFrame for each batch
            """
            df = self.collect()
            for batch in df.iter_slices(n_rows=self.batch_size):
                yield batch

        def get_by_author(self, author: str) -> "pl.DataFrame":
            """
            Get all texts by a specific author.

            Args:
                author: Author name

            Returns:
                DataFrame of texts by this author
            """
            return self.collect().filter(
                pl.col("author").str.contains(author, literal=False)
            )

        def get_by_verse(self, verse_id: str) -> "pl.DataFrame":
            """
            Get all texts that reference a specific verse.

            Args:
                verse_id: Verse ID to search for

            Returns:
                DataFrame of texts referencing this verse
            """
            # verse_references is a list column
            return self.collect().filter(
                pl.col("verse_references").list.contains(verse_id.upper())
            )

        def search_content(self, query: str) -> "pl.DataFrame":
            """
            Search text content for a query string.

            Args:
                query: Search query

            Returns:
                DataFrame of matching texts
            """
            return self.collect().filter(
                pl.col("content_clean").str.contains(query, literal=False)
            )

        def __len__(self) -> int:
            """Get total row count (requires collection)."""
            return self.scan().select(pl.count()).collect().item()


    class _PolarsDatasetWrapper(torch.utils.data.Dataset):
        """Wrapper to use Polars DataFrame as PyTorch Dataset."""

        def __init__(
            self,
            df: "pl.DataFrame",
            transform: Optional[Callable] = None
        ):
            self.df = df
            self.transform = transform

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            row = self.df.row(idx, named=True)
            if self.transform:
                row = self.transform(row)
            return row


    class _PolarsCrossRefDatasetWrapper(torch.utils.data.Dataset):
        """Wrapper to use Polars cross-ref DataFrame as PyTorch Dataset."""

        def __init__(self, df: "pl.DataFrame"):
            self.df = df

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            row = self.df.row(idx, named=True)
            return {
                "source_ref": row.get("source_ref", ""),
                "target_ref": row.get("target_ref", ""),
                "connection_type": row.get("connection_type", "thematic"),
                "strength": row.get("strength", "moderate"),
                "confidence": row.get("confidence", 1.0),
                "label": 1,  # Positive sample
            }


    # =============================================================================
    # POLARS LOADER FACTORY FUNCTIONS
    # =============================================================================

    def create_polars_verse_loader(
        data_dir: Union[str, Path],
        batch_size: int = 1000,
        books: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
        memory_map: bool = True,
    ) -> PolarsVerseLoader:
        """
        Create a Polars-based verse loader for high-performance processing.

        Args:
            data_dir: Path to data directory
            batch_size: Batch size for iteration
            books: Optional list of book codes to include
            columns: Optional columns to select (projection pushdown)
            memory_map: Whether to use memory-mapped I/O

        Returns:
            PolarsVerseLoader instance
        """
        return PolarsVerseLoader(
            data_dir=data_dir,
            batch_size=batch_size,
            books=books,
            columns=columns,
            memory_map=memory_map,
        )


    def create_polars_crossref_loader(
        data_dir: Union[str, Path],
        batch_size: int = 1000,
        connection_types: Optional[List[str]] = None,
        min_confidence: float = 0.0,
        memory_map: bool = True,
    ) -> PolarsCrossRefLoader:
        """
        Create a Polars-based cross-reference loader for high-performance processing.

        Args:
            data_dir: Path to data directory
            batch_size: Batch size for iteration
            connection_types: Optional list of connection types
            min_confidence: Minimum confidence threshold
            memory_map: Whether to use memory-mapped I/O

        Returns:
            PolarsCrossRefLoader instance
        """
        return PolarsCrossRefLoader(
            data_dir=data_dir,
            batch_size=batch_size,
            connection_types=connection_types,
            min_confidence=min_confidence,
            memory_map=memory_map,
        )


    def create_polars_patristic_loader(
        data_dir: Union[str, Path],
        batch_size: int = 100,
        authors: Optional[List[str]] = None,
        memory_map: bool = True,
    ) -> PolarsPatristicLoader:
        """
        Create a Polars-based patristic text loader for high-performance processing.

        Args:
            data_dir: Path to patristics data directory
            batch_size: Batch size for iteration
            authors: Optional list of authors to filter
            memory_map: Whether to use memory-mapped I/O

        Returns:
            PolarsPatristicLoader instance
        """
        return PolarsPatristicLoader(
            data_dir=data_dir,
            batch_size=batch_size,
            authors=authors,
            memory_map=memory_map,
        )


    def scan_patristics(
        data_dir: Union[str, Path],
        authors: Optional[List[str]] = None,
    ) -> "pl.LazyFrame":
        """
        Create a lazy scan of patristic text files.

        This is the most efficient way to process large patristic datasets
        with filtering.

        Args:
            data_dir: Path to patristics data directory
            authors: Optional list of authors to filter

        Returns:
            Polars LazyFrame for deferred execution
        """
        loader = PolarsPatristicLoader(
            data_dir=data_dir,
            authors=authors,
        )
        return loader.scan()


    def scan_verses(
        data_dir: Union[str, Path],
        books: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
    ) -> "pl.LazyFrame":
        """
        Create a lazy scan of verse files.

        This is the most efficient way to process large verse datasets
        as it enables predicate pushdown and projection optimization.

        Args:
            data_dir: Path to data directory
            books: Optional list of book codes to filter
            columns: Optional columns to select

        Returns:
            Polars LazyFrame for deferred execution
        """
        loader = PolarsVerseLoader(
            data_dir=data_dir,
            books=books,
            columns=columns,
        )
        return loader.scan()


    def scan_crossrefs(
        data_dir: Union[str, Path],
        connection_types: Optional[List[str]] = None,
        min_confidence: float = 0.0,
    ) -> "pl.LazyFrame":
        """
        Create a lazy scan of cross-reference files.

        This is the most efficient way to process large cross-reference
        datasets with filtering.

        Args:
            data_dir: Path to data directory
            connection_types: Optional list of connection types
            min_confidence: Minimum confidence threshold

        Returns:
            Polars LazyFrame for deferred execution
        """
        loader = PolarsCrossRefLoader(
            data_dir=data_dir,
            connection_types=connection_types,
            min_confidence=min_confidence,
        )
        return loader.scan()


# =============================================================================
# UNIFIED LOADER INTERFACE
# =============================================================================

class DataLoaderFactory:
    """
    Factory for creating data loaders with automatic backend selection.

    Automatically chooses between PyTorch DataLoader and Polars loader
    based on use case and available libraries.
    """

    @staticmethod
    def create_verse_loader(
        data_dir: Union[str, Path],
        batch_size: int = 32,
        backend: str = "auto",
        **kwargs
    ) -> Union[DataLoader, "PolarsVerseLoader"]:
        """
        Create a verse loader with automatic backend selection.

        Args:
            data_dir: Path to data directory
            batch_size: Batch size
            backend: "pytorch", "polars", or "auto"
            **kwargs: Additional backend-specific options

        Returns:
            DataLoader or PolarsVerseLoader instance
        """
        if backend == "auto":
            # Use Polars for large batch sizes (batch processing)
            # Use PyTorch for small batch sizes (ML training)
            backend = "polars" if (POLARS_AVAILABLE and batch_size >= 100) else "pytorch"

        if backend == "polars" and POLARS_AVAILABLE:
            return create_polars_verse_loader(
                data_dir=data_dir,
                batch_size=batch_size,
                **kwargs
            )
        else:
            return create_verse_loader(
                data_dir=data_dir,
                batch_size=batch_size,
                **{k: v for k, v in kwargs.items()
                   if k in ("shuffle", "num_workers", "books", "transform")}
            )

    @staticmethod
    def create_crossref_loader(
        data_dir: Union[str, Path],
        batch_size: int = 32,
        backend: str = "auto",
        **kwargs
    ) -> Union[DataLoader, "PolarsCrossRefLoader"]:
        """
        Create a cross-reference loader with automatic backend selection.

        Args:
            data_dir: Path to data directory
            batch_size: Batch size
            backend: "pytorch", "polars", or "auto"
            **kwargs: Additional backend-specific options

        Returns:
            DataLoader or PolarsCrossRefLoader instance
        """
        if backend == "auto":
            backend = "polars" if (POLARS_AVAILABLE and batch_size >= 100) else "pytorch"

        if backend == "polars" and POLARS_AVAILABLE:
            return create_polars_crossref_loader(
                data_dir=data_dir,
                batch_size=batch_size,
                **kwargs
            )
        else:
            return create_crossref_loader(
                data_dir=data_dir,
                batch_size=batch_size,
                **{k: v for k, v in kwargs.items()
                   if k in ("shuffle", "num_workers", "connection_types", "min_confidence")}
            )


# =============================================================================
# ADAPTER FOR PIPELINE INTEGRATION
# =============================================================================

class PolarsToContextAdapter:
    """
    Adapter to convert Polars DataFrames to pipeline context format.

    Enables seamless integration of Polars data with the existing
    pipeline orchestrator that expects dict-based context.
    """

    @staticmethod
    def verses_to_context(df: "pl.DataFrame") -> List[Dict[str, Any]]:
        """
        Convert verses DataFrame to pipeline context format.

        Args:
            df: Polars DataFrame with verse data

        Returns:
            List of verse dicts for pipeline processing
        """
        return df.to_dicts()

    @staticmethod
    def context_to_verses(
        context_list: List[Dict[str, Any]]
    ) -> "pl.DataFrame":
        """
        Convert pipeline context to verses DataFrame.

        Args:
            context_list: List of verse dicts from pipeline

        Returns:
            Polars DataFrame
        """
        if not POLARS_AVAILABLE:
            raise ImportError("Polars is not available")
        return DataFrameBuilder.verses(context_list)

    @staticmethod
    def crossrefs_to_context(df: "pl.DataFrame") -> List[Dict[str, Any]]:
        """
        Convert cross-refs DataFrame to pipeline context format.

        Args:
            df: Polars DataFrame with cross-reference data

        Returns:
            List of cross-reference dicts for pipeline processing
        """
        return df.to_dicts()

    @staticmethod
    def context_to_crossrefs(
        context_list: List[Dict[str, Any]]
    ) -> "pl.DataFrame":
        """
        Convert pipeline context to cross-refs DataFrame.

        Args:
            context_list: List of cross-reference dicts from pipeline

        Returns:
            Polars DataFrame
        """
        if not POLARS_AVAILABLE:
            raise ImportError("Polars is not available")
        return DataFrameBuilder.cross_references(context_list)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Original PyTorch loaders (backward compatible)
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
    # Factory
    "DataLoaderFactory",
    # Adapter
    "PolarsToContextAdapter",
]

# Add Polars exports if available
if POLARS_AVAILABLE:
    __all__.extend([
        "PolarsVerseLoader",
        "PolarsCrossRefLoader",
        "create_polars_verse_loader",
        "create_polars_crossref_loader",
        "scan_verses",
        "scan_crossrefs",
    ])
