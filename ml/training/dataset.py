"""
BIBLOS v2 - Training Dataset

PyTorch Dataset implementation for BIBLOS training data.
"""
from typing import Dict, List, Any, Optional, Callable, TYPE_CHECKING
from dataclasses import dataclass
import logging

# Handle optional torch dependency
TORCH_AVAILABLE = False
try:
    from torch.utils.data import Dataset as TorchDataset
    TORCH_AVAILABLE = True
    _BaseClass = TorchDataset
except ImportError:
    _BaseClass = object


@dataclass
class DatasetConfig:
    """Configuration for BiblosDataset."""
    data_path: str = "data/training"
    max_length: int = 512
    include_metadata: bool = True
    cache_size: int = 10000


class BiblosDataset(_BaseClass):
    """
    PyTorch Dataset for BIBLOS training data.

    Supports:
    - Cross-reference pairs for contrastive learning
    - Verse embeddings with metadata
    - Agent extraction results
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        config: Optional[DatasetConfig] = None,
        transform: Optional[Callable] = None
    ):
        """
        Initialize BiblosDataset.

        Args:
            data: List of data samples
            config: Dataset configuration
            transform: Optional transform to apply to samples
        """
        self.data = data
        self.config = config or DatasetConfig()
        self.transform = transform
        self.logger = logging.getLogger("biblos.ml.dataset")

        self.logger.info(f"Initialized BiblosDataset with {len(data)} samples")

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample by index."""
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

    @classmethod
    def from_json(cls, path: str, config: Optional[DatasetConfig] = None) -> "BiblosDataset":
        """Load dataset from JSON file."""
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data, config)

    @classmethod
    def from_verses(
        cls,
        verses: List[Dict[str, Any]],
        config: Optional[DatasetConfig] = None
    ) -> "BiblosDataset":
        """Create dataset from verse data."""
        data = []
        for verse in verses:
            data.append({
                "verse_id": verse.get("verse_id", ""),
                "text": verse.get("text", ""),
                "metadata": verse.get("metadata", {})
            })
        return cls(data, config)

    def get_collate_fn(self) -> Optional[Callable]:
        """Get collate function for DataLoader."""
        if not TORCH_AVAILABLE:
            return None

        def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Collate batch of samples."""
            verse_ids = [sample.get("verse_id", "") for sample in batch]
            texts = [sample.get("text", "") for sample in batch]
            metadata = [sample.get("metadata", {}) for sample in batch]

            return {
                "verse_ids": verse_ids,
                "texts": texts,
                "metadata": metadata
            }

        return collate_fn
