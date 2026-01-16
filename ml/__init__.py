"""
BIBLOS v2 - Machine Learning Pipeline

Components:
- embeddings: Multi-model ensemble embeddings for biblical texts
- models: GNN for cross-reference discovery, classifiers
- training: MLflow-integrated training pipelines
- inference: Production serving infrastructure
- cache: O(1) LRU cache with TTL and memory-aware eviction
- batch_processor: True parallel batch processing for GPU optimization
"""

from ml.config import MLConfig
from ml.embeddings.ensemble import EnsembleEmbedder, EnsembleResult
from ml.models.gnn_discovery import CrossRefGNN
from ml.training.trainer import BiblosTrainer

# New optimized components
from ml.cache import (
    LRUCache,
    AsyncLRUCache,
    CacheStats,
    CacheEntry,
    embedding_cache_key,
    cached_embedding,
    cached_embedding_async,
    get_embedding_cache,
)

from ml.batch_processor import (
    BatchConfig,
    BatchStats,
    GPUMemoryManager,
    DynamicBatcher,
    EmbeddingBatcher,
    true_batch_embed,
)

__version__ = "2.0.0"
__all__ = [
    "MLConfig",
    "EnsembleEmbedder",
    "EnsembleResult",
    "CrossRefGNN",
    "BiblosTrainer",
    # Cache
    "LRUCache",
    "AsyncLRUCache",
    "CacheStats",
    "CacheEntry",
    "embedding_cache_key",
    "cached_embedding",
    "cached_embedding_async",
    "get_embedding_cache",
    # Batch processing
    "BatchConfig",
    "BatchStats",
    "GPUMemoryManager",
    "DynamicBatcher",
    "EmbeddingBatcher",
    "true_batch_embed",
]
