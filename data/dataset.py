"""
BIBLOS v2 - Dataset Classes

PyTorch-compatible datasets for biblical text processing and ML training.
Uses centralized schemas for system-wide uniformity.
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import random

import torch
from torch.utils.data import Dataset
import numpy as np

# Import centralized schemas
from data.schemas import (
    VerseSchema,
    CrossReferenceSchema,
    validate_verse_id,
    normalize_verse_id,
    validate_connection_type,
    validate_strength
)


@dataclass
class VerseRecord:
    """
    Record for a single verse.

    Aligned with VerseSchema for system-wide uniformity.
    """
    verse_id: str
    book: str
    chapter: int
    verse: int
    text: str
    language: str
    embeddings: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Normalize verse_id."""
        if self.verse_id:
            self.verse_id = normalize_verse_id(self.verse_id)

    def to_schema(self) -> VerseSchema:
        """Convert to VerseSchema."""
        ot_books = {
            "GEN", "EXO", "LEV", "NUM", "DEU", "JOS", "JDG", "RUT",
            "1SA", "2SA", "1KI", "2KI", "1CH", "2CH", "EZR", "NEH",
            "EST", "JOB", "PSA", "PRO", "ECC", "SNG", "ISA", "JER",
            "LAM", "EZK", "DAN", "HOS", "JOL", "AMO", "OBA", "JON",
            "MIC", "NAH", "HAB", "ZEP", "HAG", "ZEC", "MAL"
        }
        testament = "OT" if self.book.upper() in ot_books else "NT"

        return VerseSchema(
            verse_id=self.verse_id,
            book=self.book,
            book_name="",  # Would need lookup
            chapter=self.chapter,
            verse=self.verse,
            text=self.text,
            original_text="",
            testament=testament,
            language=self.language,
            metadata=self.metadata
        )


@dataclass
class CrossReferenceRecord:
    """
    Record for a cross-reference.

    Aligned with CrossReferenceSchema for system-wide uniformity.
    """
    source_ref: str
    target_ref: str
    connection_type: str
    strength: str
    confidence: float = 1.0
    notes: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Normalize and validate fields."""
        if self.source_ref:
            self.source_ref = normalize_verse_id(self.source_ref)
        if self.target_ref:
            self.target_ref = normalize_verse_id(self.target_ref)
        if not validate_connection_type(self.connection_type):
            self.connection_type = "thematic"
        if not validate_strength(self.strength):
            self.strength = "moderate"

    def to_schema(self) -> CrossReferenceSchema:
        """Convert to CrossReferenceSchema."""
        return CrossReferenceSchema(
            source_ref=self.source_ref,
            target_ref=self.target_ref,
            connection_type=self.connection_type,
            strength=self.strength,
            confidence=self.confidence,
            notes=self.notes,
            sources=[],
            verified=False,
            patristic_support=False
        )


class BibleDataset(Dataset):
    """
    Dataset for complete Bible verses.

    Provides access to verse text, metadata, and pre-computed features.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        books: Optional[List[str]] = None,
        transform: Optional[callable] = None
    ):
        self.data_dir = Path(data_dir)
        self.books = books
        self.transform = transform
        self._verses: List[VerseRecord] = []
        self._index: Dict[str, int] = {}

    def load(self) -> None:
        """Load dataset from files."""
        if not self.data_dir.exists():
            return

        # Load verse files
        json_files = list(self.data_dir.glob("**/*.json"))

        for file_path in json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for item in data:
                        self._add_verse(item)
                elif isinstance(data, dict):
                    if "verses" in data:
                        for item in data["verses"]:
                            self._add_verse(item)
                    else:
                        self._add_verse(data)

            except Exception:
                continue

    def _add_verse(self, data: Dict[str, Any]) -> None:
        """Add a verse record from dict."""
        verse_id = data.get("verse_id") or data.get("ref")
        if not verse_id:
            return

        # Parse verse ID
        parts = verse_id.upper().replace(" ", ".").replace(":", ".").split(".")
        if len(parts) < 3:
            return

        book = parts[0]
        if self.books and book not in self.books:
            return

        try:
            record = VerseRecord(
                verse_id=verse_id,
                book=book,
                chapter=int(parts[1]),
                verse=int(parts[2]),
                text=data.get("text", ""),
                language=data.get("language", "unknown"),
                metadata=data.get("metadata", {})
            )

            self._index[verse_id] = len(self._verses)
            self._verses.append(record)

        except (ValueError, KeyError):
            pass

    def __len__(self) -> int:
        return len(self._verses)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self._verses):
            raise IndexError(f"Index {idx} out of range")

        record = self._verses[idx]

        item = {
            "verse_id": record.verse_id,
            "book": record.book,
            "chapter": record.chapter,
            "verse": record.verse,
            "text": record.text,
            "language": record.language
        }

        if self.transform:
            item = self.transform(item)

        return item

    def get_by_id(self, verse_id: str) -> Optional[VerseRecord]:
        """Get verse by ID."""
        idx = self._index.get(verse_id.upper())
        if idx is not None:
            return self._verses[idx]
        return None

    def get_book(self, book: str) -> List[VerseRecord]:
        """Get all verses for a book."""
        return [v for v in self._verses if v.book == book.upper()]


class CrossReferenceDataset(Dataset):
    """
    Dataset for cross-references.

    Used for training cross-reference discovery models.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        connection_types: Optional[List[str]] = None,
        min_confidence: float = 0.0
    ):
        self.data_dir = Path(data_dir)
        self.connection_types = connection_types
        self.min_confidence = min_confidence
        self._references: List[CrossReferenceRecord] = []

    def load(self) -> None:
        """Load cross-reference data."""
        if not self.data_dir.exists():
            return

        json_files = list(self.data_dir.glob("**/*.json"))

        for file_path in json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for item in data:
                        self._add_reference(item)
                elif isinstance(data, dict):
                    if "cross_references" in data:
                        for item in data["cross_references"]:
                            self._add_reference(item)
                    else:
                        self._add_reference(data)

            except Exception:
                continue

    def _add_reference(self, data: Dict[str, Any]) -> None:
        """Add a cross-reference record."""
        source = data.get("source_ref")
        target = data.get("target_ref")
        conn_type = data.get("connection_type", "thematic")

        if not source or not target:
            return

        if self.connection_types and conn_type not in self.connection_types:
            return

        confidence = data.get("confidence", 1.0)
        if confidence < self.min_confidence:
            return

        record = CrossReferenceRecord(
            source_ref=source,
            target_ref=target,
            connection_type=conn_type,
            strength=data.get("strength", "moderate"),
            confidence=confidence,
            notes=data.get("notes", [])
        )

        self._references.append(record)

    def __len__(self) -> int:
        return len(self._references)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self._references):
            raise IndexError(f"Index {idx} out of range")

        record = self._references[idx]

        return {
            "source_ref": record.source_ref,
            "target_ref": record.target_ref,
            "connection_type": record.connection_type,
            "strength": record.strength,
            "confidence": record.confidence,
            "label": 1  # Positive sample
        }

    def get_by_source(self, source_ref: str) -> List[CrossReferenceRecord]:
        """Get all cross-references from a source."""
        return [r for r in self._references if r.source_ref == source_ref]

    def get_by_type(self, conn_type: str) -> List[CrossReferenceRecord]:
        """Get all cross-references of a type."""
        return [r for r in self._references if r.connection_type == conn_type]


class VerseDataset(Dataset):
    """
    Simple verse text dataset for embedding models.
    """

    def __init__(
        self,
        verses: List[Dict[str, str]],
        tokenizer: Optional[callable] = None,
        max_length: int = 512
    ):
        self.verses = verses
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.verses)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        verse = self.verses[idx]

        item = {
            "verse_id": verse.get("verse_id", f"verse_{idx}"),
            "text": verse.get("text", "")
        }

        if self.tokenizer:
            encoded = self.tokenizer(
                item["text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            item["input_ids"] = encoded["input_ids"].squeeze()
            item["attention_mask"] = encoded["attention_mask"].squeeze()

        return item


class PairDataset(Dataset):
    """
    Dataset for verse pair classification.

    Used for training cross-reference classifiers.
    """

    def __init__(
        self,
        positive_pairs: List[Tuple[str, str, str]],  # (source, target, type)
        verse_texts: Dict[str, str],
        negative_ratio: float = 1.0,
        tokenizer: Optional[callable] = None,
        max_length: int = 512
    ):
        self.positive_pairs = positive_pairs
        self.verse_texts = verse_texts
        self.negative_ratio = negative_ratio
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._samples: List[Dict[str, Any]] = []
        self._build_samples()

    def _build_samples(self) -> None:
        """Build positive and negative samples."""
        all_verses = list(self.verse_texts.keys())

        # Add positive samples
        for source, target, conn_type in self.positive_pairs:
            if source in self.verse_texts and target in self.verse_texts:
                self._samples.append({
                    "source": source,
                    "target": target,
                    "source_text": self.verse_texts[source],
                    "target_text": self.verse_texts[target],
                    "connection_type": conn_type,
                    "label": 1
                })

        # Add negative samples
        positive_set = {(s, t) for s, t, _ in self.positive_pairs}
        num_negatives = int(len(self.positive_pairs) * self.negative_ratio)

        attempts = 0
        max_attempts = num_negatives * 10

        while len([s for s in self._samples if s["label"] == 0]) < num_negatives and attempts < max_attempts:
            source = random.choice(all_verses)
            target = random.choice(all_verses)

            if source != target and (source, target) not in positive_set:
                self._samples.append({
                    "source": source,
                    "target": target,
                    "source_text": self.verse_texts.get(source, ""),
                    "target_text": self.verse_texts.get(target, ""),
                    "connection_type": "none",
                    "label": 0
                })

            attempts += 1

        random.shuffle(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._samples[idx]

        item = {
            "source": sample["source"],
            "target": sample["target"],
            "source_text": sample["source_text"],
            "target_text": sample["target_text"],
            "connection_type": sample["connection_type"],
            "label": sample["label"]
        }

        if self.tokenizer:
            # Encode pair
            encoded = self.tokenizer(
                sample["source_text"],
                sample["target_text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            item["input_ids"] = encoded["input_ids"].squeeze()
            item["attention_mask"] = encoded["attention_mask"].squeeze()
            if "token_type_ids" in encoded:
                item["token_type_ids"] = encoded["token_type_ids"].squeeze()

        item["label"] = torch.tensor(sample["label"], dtype=torch.long)

        return item

    def get_positive_count(self) -> int:
        """Get number of positive samples."""
        return sum(1 for s in self._samples if s["label"] == 1)

    def get_negative_count(self) -> int:
        """Get number of negative samples."""
        return sum(1 for s in self._samples if s["label"] == 0)
