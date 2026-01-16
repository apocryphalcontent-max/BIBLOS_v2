"""
BIBLOS v2 - Base Embedder Interface
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class EmbeddingResult:
    """Result from a single embedding computation."""
    text: str
    model_name: str
    embedding: np.ndarray
    dimensions: int
    cached: bool = False
    processing_time_ms: float = 0.0


class BaseEmbedder(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    async def embed(self, text: str) -> np.ndarray:
        """Embed a single text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a batch of texts."""
        pass

    @abstractmethod
    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get model name."""
        pass
