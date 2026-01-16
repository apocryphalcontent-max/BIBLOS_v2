# BIBLOS v2 ML Pipeline Optimization Report

**Version:** 2.0.0
**Date:** 2026-01-15
**Author:** MLOps Engineering Review

---

## Executive Summary

This report provides a comprehensive review of the BIBLOS v2 ML pipeline with specific optimization recommendations for elite-level performance. The analysis covers embedding pipelines, GNN models, inference systems, training infrastructure, and data handling.

**Key Findings:**
- Embedding pipeline lacks GPU batching and has inefficient cache implementation
- GNN model has O(n^2) candidate generation bottleneck
- Inference pipeline missing async batching and warm-up strategies
- Training pipeline needs distributed training support (DDP/FSDP)
- Data loaders not optimized for Polars integration

---

## 1. Embedding Pipeline Optimization

### Current State Analysis

**File:** `ml/embeddings/ensemble.py`

**Issues Identified:**

1. **Sequential Batch Processing (Lines 324-334):**
```python
async def embed_batch(self, texts: List[str], use_cache: bool = True) -> List[EnsembleResult]:
    results = []
    for text in texts:
        result = await self.embed(text, use_cache)  # Sequential!
        results.append(result)
    return results
```

2. **Inefficient LRU Cache (Lines 97-106):**
```python
def _add_to_memory(self, key: str, embedding: np.ndarray) -> None:
    if key in self._memory_cache:
        self._access_order.remove(key)  # O(n) operation!
```

3. **No GPU Batching:**
The `encode()` call happens one text at a time, not utilizing GPU parallelism.

### Recommended Optimizations

#### 1.1 Async GPU Batching with Proper Queuing

```python
# ml/embeddings/optimized_ensemble.py

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import hashlib


@dataclass
class EmbeddingBatchConfig:
    """Configuration for batched embedding generation."""
    max_batch_size: int = 64
    max_queue_wait_ms: float = 50.0  # Max time to wait for batch to fill
    prefetch_factor: int = 2
    pin_memory: bool = True
    use_fp16: bool = True


class OptimizedEmbeddingCache:
    """
    High-performance LRU cache using OrderedDict for O(1) operations.
    Supports Redis backend for distributed caching.
    """

    def __init__(
        self,
        max_memory_items: int = 100000,
        redis_client: Optional['redis.Redis'] = None,
        embedding_dim: int = 768
    ):
        self._memory_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.max_memory_items = max_memory_items
        self.redis_client = redis_client
        self.embedding_dim = embedding_dim
        self._hits = 0
        self._misses = 0

    def _make_key(self, text: str, model: str) -> str:
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(self, text: str, model: str) -> Optional[np.ndarray]:
        key = self._make_key(text, model)

        # Check memory cache (O(1) with OrderedDict)
        if key in self._memory_cache:
            self._memory_cache.move_to_end(key)
            self._hits += 1
            return self._memory_cache[key]

        # Check Redis if available
        if self.redis_client:
            try:
                cached = self.redis_client.get(f"emb:{key}")
                if cached:
                    embedding = np.frombuffer(cached, dtype=np.float32)
                    self._add_to_memory(key, embedding)
                    self._hits += 1
                    return embedding
            except Exception:
                pass

        self._misses += 1
        return None

    def get_batch(
        self,
        texts: List[str],
        model: str
    ) -> Tuple[List[np.ndarray], List[int], List[str]]:
        """
        Batch cache lookup.

        Returns:
            - cached_embeddings: List of cached embeddings
            - miss_indices: Indices of texts not in cache
            - miss_texts: Texts not in cache
        """
        cached = []
        miss_indices = []
        miss_texts = []

        for i, text in enumerate(texts):
            emb = self.get(text, model)
            if emb is not None:
                cached.append(emb)
            else:
                cached.append(None)
                miss_indices.append(i)
                miss_texts.append(text)

        return cached, miss_indices, miss_texts

    def put(self, text: str, model: str, embedding: np.ndarray) -> None:
        key = self._make_key(text, model)
        self._add_to_memory(key, embedding)

        # Async write to Redis
        if self.redis_client:
            try:
                self.redis_client.setex(
                    f"emb:{key}",
                    3600,  # 1 hour TTL
                    embedding.tobytes()
                )
            except Exception:
                pass

    def put_batch(
        self,
        texts: List[str],
        model: str,
        embeddings: List[np.ndarray]
    ) -> None:
        """Batch cache insertion."""
        for text, emb in zip(texts, embeddings):
            self.put(text, model, emb)

    def _add_to_memory(self, key: str, embedding: np.ndarray) -> None:
        if key in self._memory_cache:
            self._memory_cache.move_to_end(key)
        else:
            if len(self._memory_cache) >= self.max_memory_items:
                self._memory_cache.popitem(last=False)
            self._memory_cache[key] = embedding


class BatchingEmbedder:
    """
    GPU-optimized batching embedder with async request queuing.
    """

    def __init__(
        self,
        model: SentenceTransformer,
        config: EmbeddingBatchConfig,
        cache: OptimizedEmbeddingCache,
        model_name: str
    ):
        self.model = model
        self.config = config
        self.cache = cache
        self.model_name = model_name
        self._queue: asyncio.Queue = asyncio.Queue()
        self._batch_event = asyncio.Event()
        self._processor_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        """Start the batch processor."""
        self._running = True
        self._processor_task = asyncio.create_task(self._process_batches())

    async def stop(self):
        """Stop the batch processor."""
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()

    async def embed(self, text: str) -> np.ndarray:
        """Embed single text with batching."""
        # Check cache first
        cached = self.cache.get(text, self.model_name)
        if cached is not None:
            return cached

        # Create future for result
        future = asyncio.Future()
        await self._queue.put((text, future))

        # Trigger batch processing if queue is full
        if self._queue.qsize() >= self.config.max_batch_size:
            self._batch_event.set()

        return await future

    async def embed_batch_direct(self, texts: List[str]) -> List[np.ndarray]:
        """
        Direct batch embedding bypassing queue (for large batches).
        Uses cache-aware batching.
        """
        # Check cache for all texts
        cached, miss_indices, miss_texts = self.cache.get_batch(
            texts, self.model_name
        )

        if not miss_texts:
            # All cached
            return [c for c in cached if c is not None]

        # Encode missing texts in batches
        new_embeddings = []
        for i in range(0, len(miss_texts), self.config.max_batch_size):
            batch = miss_texts[i:i + self.config.max_batch_size]
            batch_embs = self._encode_batch(batch)
            new_embeddings.extend(batch_embs)

        # Update cache
        self.cache.put_batch(miss_texts, self.model_name, new_embeddings)

        # Merge results
        result = cached.copy()
        for idx, emb in zip(miss_indices, new_embeddings):
            result[idx] = emb

        return [r for r in result if r is not None]

    def _encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """GPU-optimized batch encoding."""
        with torch.cuda.amp.autocast(enabled=self.config.use_fp16):
            embeddings = self.model.encode(
                texts,
                batch_size=len(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )

        if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
            return [embeddings[i] for i in range(embeddings.shape[0])]
        return [embeddings]

    async def _process_batches(self):
        """Background batch processor."""
        while self._running:
            try:
                # Wait for batch event or timeout
                await asyncio.wait_for(
                    self._batch_event.wait(),
                    timeout=self.config.max_queue_wait_ms / 1000
                )
            except asyncio.TimeoutError:
                pass

            self._batch_event.clear()

            # Collect items from queue
            batch_items = []
            while not self._queue.empty() and len(batch_items) < self.config.max_batch_size:
                try:
                    item = self._queue.get_nowait()
                    batch_items.append(item)
                except asyncio.QueueEmpty:
                    break

            if not batch_items:
                continue

            # Process batch
            texts = [item[0] for item in batch_items]
            futures = [item[1] for item in batch_items]

            try:
                embeddings = self._encode_batch(texts)

                # Cache and return results
                for text, emb, future in zip(texts, embeddings, futures):
                    self.cache.put(text, self.model_name, emb)
                    if not future.done():
                        future.set_result(emb)
            except Exception as e:
                for future in futures:
                    if not future.done():
                        future.set_exception(e)
```

#### 1.2 ONNX Export for Faster Inference

```python
# ml/embeddings/onnx_export.py

import torch
import onnx
import onnxruntime as ort
from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np


def export_to_onnx(
    model: SentenceTransformer,
    output_path: Path,
    max_seq_length: int = 512,
    opset_version: int = 14
) -> Path:
    """
    Export SentenceTransformer to ONNX for optimized inference.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    onnx_path = output_path / "model.onnx"

    # Get underlying transformer
    transformer = model._first_module()

    # Create dummy inputs
    dummy_input_ids = torch.zeros(1, max_seq_length, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, max_seq_length, dtype=torch.long)

    # Export to ONNX
    torch.onnx.export(
        transformer.auto_model,
        (dummy_input_ids, dummy_attention_mask),
        str(onnx_path),
        input_names=['input_ids', 'attention_mask'],
        output_names=['last_hidden_state'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'}
        },
        opset_version=opset_version,
        do_constant_folding=True
    )

    # Optimize ONNX model
    from onnxruntime.transformers import optimizer
    optimized_path = output_path / "model_optimized.onnx"

    optimizer.optimize_model(
        str(onnx_path),
        str(optimized_path),
        model_type='bert',
        opt_level=2,  # Extended optimizations
        use_gpu=True
    )

    return optimized_path


class ONNXEmbedder:
    """
    ONNX Runtime based embedder for production inference.
    """

    def __init__(
        self,
        model_path: Path,
        tokenizer,
        use_gpu: bool = True
    ):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 4

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers
        )
        self.tokenizer = tokenizer
        self.io_binding = None

        if use_gpu:
            self._setup_io_binding()

    def _setup_io_binding(self):
        """Pre-allocate GPU memory for zero-copy inference."""
        self.io_binding = self.session.io_binding()

    def embed_batch(
        self,
        texts: List[str],
        max_length: int = 512
    ) -> np.ndarray:
        """
        Batch embedding with ONNX Runtime.
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )

        # Run inference
        outputs = self.session.run(
            None,
            {
                'input_ids': encoded['input_ids'].astype(np.int64),
                'attention_mask': encoded['attention_mask'].astype(np.int64)
            }
        )

        # Mean pooling
        token_embeddings = outputs[0]
        attention_mask = encoded['attention_mask']

        # Expand attention mask
        mask_expanded = np.expand_dims(attention_mask, -1)
        mask_expanded = np.broadcast_to(mask_expanded, token_embeddings.shape)

        # Weighted sum
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)

        embeddings = sum_embeddings / sum_mask

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        return embeddings
```

#### 1.3 Memory-Efficient Quantization

```python
# ml/embeddings/quantization.py

import torch
from torch.quantization import quantize_dynamic
import numpy as np


def quantize_model_int8(model: torch.nn.Module) -> torch.nn.Module:
    """
    Apply INT8 dynamic quantization to embedding model.
    Reduces memory by ~4x with minimal accuracy loss.
    """
    quantized = quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    return quantized


class HalfPrecisionEmbedder:
    """
    FP16 embedder for 2x memory reduction on GPU.
    """

    def __init__(self, model: SentenceTransformer, device: str = "cuda"):
        self.model = model
        self.device = device

        # Convert to half precision
        self.model.half()
        self.model.to(device)

    @torch.cuda.amp.autocast()
    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Encode with automatic mixed precision."""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            with torch.no_grad():
                embeddings = self.model.encode(
                    batch,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )

                # Convert back to float32 for compatibility
                embeddings = embeddings.float().cpu().numpy()
                all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)
```

---

## 2. GNN Model Optimization

### Current State Analysis

**File:** `ml/models/gnn_discovery.py`

**Issues Identified:**

1. **O(n^2) Candidate Generation (Lines 218-223):**
```python
for i in range(num_verses):
    for j in range(num_verses):
        if i != j and (i, j) not in existing_set:
            candidates.append((i, j))
```

2. **No Mini-Batch Training for Large Graphs**

3. **Missing PyTorch Geometric Optimizations**

### Recommended Optimizations

#### 2.1 Efficient Candidate Generation with FAISS

```python
# ml/models/optimized_gnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import NeighborLoader, ClusterData, ClusterLoader
import numpy as np
from typing import List, Tuple, Dict, Optional
import faiss


class FAISSCandidateGenerator:
    """
    Use FAISS for efficient approximate nearest neighbor search
    to generate cross-reference candidates.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        nlist: int = 100,  # Number of clusters
        nprobe: int = 10,   # Clusters to search
        use_gpu: bool = True
    ):
        self.embedding_dim = embedding_dim
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_gpu = use_gpu
        self.index = None
        self.verse_ids: List[str] = []

    def build_index(
        self,
        embeddings: np.ndarray,
        verse_ids: List[str]
    ) -> None:
        """
        Build FAISS index for efficient similarity search.
        O(n * log(n)) vs O(n^2) for brute force.
        """
        self.verse_ids = verse_ids
        n_samples = embeddings.shape[0]

        # Choose index type based on dataset size
        if n_samples < 10000:
            # Flat index for small datasets (exact search)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            # IVF index for large datasets (approximate search)
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer,
                self.embedding_dim,
                min(self.nlist, n_samples // 39)
            )

            # Train index
            self.index.train(embeddings.astype(np.float32))
            self.index.nprobe = self.nprobe

        # Add vectors
        faiss.normalize_L2(embeddings.astype(np.float32))
        self.index.add(embeddings.astype(np.float32))

        # Move to GPU if available
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def find_candidates(
        self,
        query_embedding: np.ndarray,
        query_verse_id: str,
        top_k: int = 100,
        existing_edges: Optional[set] = None
    ) -> List[Tuple[str, float]]:
        """
        Find top-k candidate verses for a query.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index first.")

        # Normalize query
        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)

        # Search
        scores, indices = self.index.search(query, top_k + 1)

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            target_id = self.verse_ids[idx]

            # Skip self and existing edges
            if target_id == query_verse_id:
                continue

            if existing_edges and (query_verse_id, target_id) in existing_edges:
                continue

            candidates.append((target_id, float(score)))

            if len(candidates) >= top_k:
                break

        return candidates

    def batch_find_candidates(
        self,
        query_embeddings: np.ndarray,
        query_verse_ids: List[str],
        top_k: int = 100
    ) -> List[List[Tuple[str, float]]]:
        """
        Batch candidate generation for multiple queries.
        """
        queries = query_embeddings.astype(np.float32)
        faiss.normalize_L2(queries)

        scores, indices = self.index.search(queries, top_k + 1)

        all_candidates = []
        for i, query_id in enumerate(query_verse_ids):
            candidates = []
            for score, idx in zip(scores[i], indices[i]):
                if idx < 0:
                    continue
                target_id = self.verse_ids[idx]
                if target_id != query_id:
                    candidates.append((target_id, float(score)))
            all_candidates.append(candidates[:top_k])

        return all_candidates


class OptimizedCrossRefGNN(nn.Module):
    """
    Optimized GNN with mini-batch training support and
    efficient message passing.
    """

    CONNECTION_TYPES = [
        "thematic", "verbal", "conceptual", "historical", "typological",
        "prophetic", "liturgical", "narrative", "genealogical", "geographical"
    ]

    def __init__(
        self,
        in_channels: int = 768,
        hidden_channels: int = 256,
        num_layers: int = 3,
        heads: int = 8,
        dropout: float = 0.2,
        num_connection_types: int = 10,
        use_checkpoint: bool = False  # Gradient checkpointing
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint

        # Input projection with LayerNorm
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # GAT layers with residual connections
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.residual_projs = nn.ModuleList()

        for i in range(num_layers):
            in_dim = hidden_channels if i == 0 else hidden_channels * heads
            out_dim = hidden_channels

            self.gat_layers.append(
                GATv2Conv(
                    in_dim,
                    out_dim,
                    heads=heads,
                    dropout=dropout,
                    concat=True if i < num_layers - 1 else False,
                    add_self_loops=True,
                    share_weights=True  # Memory optimization
                )
            )

            out_features = out_dim * heads if i < num_layers - 1 else out_dim
            self.layer_norms.append(nn.LayerNorm(out_features))

            # Residual projection if dimensions change
            if in_dim != out_features:
                self.residual_projs.append(nn.Linear(in_dim, out_features))
            else:
                self.residual_projs.append(nn.Identity())

        # Edge prediction MLP
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )

        # Type classifier with label smoothing friendly design
        self.type_classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_connection_types)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with gradient checkpointing support.
        """
        # Input projection
        x = self.input_proj(x)

        # GAT layers with residual connections
        for i, (gat, norm, res_proj) in enumerate(
            zip(self.gat_layers, self.layer_norms, self.residual_projs)
        ):
            residual = res_proj(x)

            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    gat, x, edge_index,
                    use_reentrant=False
                )
            else:
                x = gat(x, edge_index)

            x = norm(x + residual)  # Residual connection

            if i < len(self.gat_layers) - 1:
                x = F.gelu(x)

        return x

    def predict_edges_batched(
        self,
        node_embeddings: torch.Tensor,
        source_idx: torch.Tensor,
        target_idx: torch.Tensor,
        batch_size: int = 10000
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Memory-efficient batched edge prediction for large candidate sets.
        """
        num_pairs = source_idx.shape[0]
        all_scores = []
        all_logits = []

        for i in range(0, num_pairs, batch_size):
            end_idx = min(i + batch_size, num_pairs)

            src = source_idx[i:end_idx]
            tgt = target_idx[i:end_idx]

            source_emb = node_embeddings[src]
            target_emb = node_embeddings[tgt]
            pair_emb = torch.cat([source_emb, target_emb], dim=-1)

            with torch.cuda.amp.autocast():
                scores = self.edge_predictor(pair_emb).squeeze(-1)
                logits = self.type_classifier(pair_emb)

            all_scores.append(scores)
            all_logits.append(logits)

        return torch.cat(all_scores), torch.cat(all_logits)


class MiniBatchGNNTrainer:
    """
    Mini-batch training for large biblical verse graphs.
    Uses NeighborLoader for efficient subgraph sampling.
    """

    def __init__(
        self,
        model: OptimizedCrossRefGNN,
        num_neighbors: List[int] = [25, 10],
        batch_size: int = 1024,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.device = device

    def create_loader(
        self,
        data: Data,
        shuffle: bool = True
    ) -> NeighborLoader:
        """
        Create NeighborLoader for mini-batch training.
        """
        return NeighborLoader(
            data,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )

    def train_epoch(
        self,
        loader: NeighborLoader,
        optimizer: torch.optim.Optimizer,
        edge_criterion: nn.Module,
        type_criterion: nn.Module,
        scaler: Optional[torch.cuda.amp.GradScaler] = None
    ) -> Dict[str, float]:
        """
        Train one epoch with mini-batches.
        """
        self.model.train()
        total_loss = 0
        total_edge_loss = 0
        total_type_loss = 0
        num_batches = 0

        for batch in loader:
            batch = batch.to(self.device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                # Forward pass
                node_emb = self.model(batch.x, batch.edge_index)

                # Sample edges for training
                pos_edges = batch.edge_index
                neg_edges = self._sample_negative_edges(batch, pos_edges.shape[1])

                # Edge prediction
                pos_src, pos_tgt = pos_edges
                neg_src, neg_tgt = neg_edges

                all_src = torch.cat([pos_src, neg_src])
                all_tgt = torch.cat([pos_tgt, neg_tgt])

                scores, type_logits = self.model.predict_edges_batched(
                    node_emb, all_src, all_tgt
                )

                # Labels
                edge_labels = torch.cat([
                    torch.ones(pos_edges.shape[1], device=self.device),
                    torch.zeros(neg_edges.shape[1], device=self.device)
                ])

                # Losses
                edge_loss = edge_criterion(scores, edge_labels)

                # Type loss only for positive edges
                if hasattr(batch, 'edge_type'):
                    type_loss = type_criterion(
                        type_logits[:pos_edges.shape[1]],
                        batch.edge_type
                    )
                else:
                    type_loss = torch.tensor(0.0, device=self.device)

                loss = edge_loss + 0.5 * type_loss

            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_edge_loss += edge_loss.item()
            total_type_loss += type_loss.item()
            num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "edge_loss": total_edge_loss / num_batches,
            "type_loss": total_type_loss / num_batches
        }

    def _sample_negative_edges(
        self,
        batch: Data,
        num_neg: int
    ) -> torch.Tensor:
        """
        Efficient negative edge sampling.
        """
        num_nodes = batch.num_nodes

        # Random negative edges
        neg_src = torch.randint(0, num_nodes, (num_neg,), device=self.device)
        neg_tgt = torch.randint(0, num_nodes, (num_neg,), device=self.device)

        # Remove self-loops
        mask = neg_src != neg_tgt
        neg_src = neg_src[mask]
        neg_tgt = neg_tgt[mask]

        return torch.stack([neg_src, neg_tgt])
```

---

## 3. Inference Pipeline Optimization

### Current State Analysis

**File:** `ml/inference/pipeline.py`

**Issues Identified:**

1. **No Async Batching (Lines 583-614):**
Sequential processing in `infer_batch` despite using `asyncio.gather`

2. **Cold Start Issues:**
No model warm-up strategy

3. **Inefficient Embedding Cache:**
Simple dict without TTL or size limits

### Recommended Optimizations

#### 3.1 Async Batching with Request Coalescing

```python
# ml/inference/optimized_pipeline.py

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import time
import torch
import numpy as np
from collections import deque
import threading


@dataclass
class InferenceRequest:
    """Single inference request."""
    verse_id: str
    text: str
    context: Optional[Dict[str, Any]]
    future: asyncio.Future
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizedInferenceConfig:
    """Enhanced inference configuration."""
    batch_size: int = 32
    max_batch_wait_ms: float = 100.0
    max_candidates: int = 100
    min_confidence: float = 0.5
    use_cache: bool = True
    cache_ttl_seconds: int = 3600
    warm_up_samples: int = 10
    device: str = "cuda"
    num_worker_threads: int = 2
    max_concurrent_batches: int = 4


class AsyncBatchingInferencePipeline:
    """
    Production inference pipeline with:
    - Request coalescing for efficient batching
    - Model warm-up for reduced cold start latency
    - Connection pooling for vector DB queries
    - Async I/O with proper backpressure
    """

    def __init__(self, config: Optional[OptimizedInferenceConfig] = None):
        self.config = config or OptimizedInferenceConfig()
        self._request_queue: asyncio.Queue[InferenceRequest] = asyncio.Queue()
        self._batch_semaphore = asyncio.Semaphore(self.config.max_concurrent_batches)
        self._embedding_model = None
        self._gnn_model = None
        self._candidate_generator = None
        self._initialized = False
        self._warm = False
        self._processor_tasks: List[asyncio.Task] = []

        # Metrics
        self._total_requests = 0
        self._total_latency_ms = 0.0
        self._batch_sizes: deque = deque(maxlen=1000)

    async def initialize(self) -> None:
        """Initialize models and warm up."""
        if self._initialized:
            return

        # Load models
        await self._load_models()

        # Start batch processors
        for _ in range(self.config.num_worker_threads):
            task = asyncio.create_task(self._batch_processor())
            self._processor_tasks.append(task)

        # Warm up models
        await self._warm_up()

        self._initialized = True

    async def _load_models(self) -> None:
        """Load and optimize models for inference."""
        # Load embedding model
        from ml.embeddings.optimized_ensemble import BatchingEmbedder
        # ... model loading

        # Load GNN model and set to eval mode
        from ml.models.optimized_gnn import OptimizedCrossRefGNN
        self._gnn_model = OptimizedCrossRefGNN()
        self._gnn_model.eval()

        # Compile model for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            self._gnn_model = torch.compile(
                self._gnn_model,
                mode="reduce-overhead",  # Optimized for inference
                fullgraph=True
            )

        # Move to device
        self._gnn_model.to(self.config.device)

    async def _warm_up(self) -> None:
        """
        Warm up models with dummy data to:
        - Compile CUDA kernels
        - Warm up JIT compilation
        - Fill GPU caches
        """
        dummy_texts = [f"Warm up text {i}" for i in range(self.config.warm_up_samples)]

        # Warm up embedding model
        for text in dummy_texts:
            _ = await self._get_embedding("WARM.1.1", text)

        # Warm up GNN with dummy graph
        dummy_embeddings = torch.randn(
            100, 768,
            device=self.config.device,
            dtype=torch.float32
        )
        dummy_edges = torch.randint(0, 100, (2, 500), device=self.config.device)

        with torch.no_grad():
            for _ in range(3):  # Multiple warm-up passes
                _ = self._gnn_model(dummy_embeddings, dummy_edges)

        # Synchronize CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self._warm = True

    async def infer(
        self,
        verse_id: str,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> 'InferenceResult':
        """
        Queue inference request for batch processing.
        """
        if not self._initialized:
            await self.initialize()

        # Create request with future
        future = asyncio.Future()
        request = InferenceRequest(
            verse_id=verse_id,
            text=text,
            context=context,
            future=future
        )

        await self._request_queue.put(request)

        # Wait for result
        return await future

    async def _batch_processor(self) -> None:
        """
        Background task that collects requests into batches
        and processes them efficiently.
        """
        while True:
            batch: List[InferenceRequest] = []

            # Collect first request (blocking)
            try:
                first_request = await asyncio.wait_for(
                    self._request_queue.get(),
                    timeout=1.0
                )
                batch.append(first_request)
            except asyncio.TimeoutError:
                continue

            # Collect more requests until batch full or timeout
            deadline = time.time() + self.config.max_batch_wait_ms / 1000

            while len(batch) < self.config.batch_size:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break

                try:
                    request = await asyncio.wait_for(
                        self._request_queue.get(),
                        timeout=remaining
                    )
                    batch.append(request)
                except asyncio.TimeoutError:
                    break

            # Process batch with semaphore for backpressure
            async with self._batch_semaphore:
                await self._process_batch(batch)

    async def _process_batch(self, batch: List[InferenceRequest]) -> None:
        """
        Process a batch of inference requests.
        """
        start_time = time.time()
        self._batch_sizes.append(len(batch))

        try:
            # Extract texts and verse IDs
            texts = [r.text for r in batch]
            verse_ids = [r.verse_id for r in batch]

            # Batch embedding generation
            embeddings = await self._get_embeddings_batch(verse_ids, texts)

            # Batch candidate generation
            all_candidates = await self._find_candidates_batch(
                verse_ids, embeddings, [r.context for r in batch]
            )

            # Batch classification
            all_results = await self._classify_candidates_batch(
                verse_ids, texts, all_candidates, [r.context for r in batch]
            )

            # Return results
            for request, result in zip(batch, all_results):
                if not request.future.done():
                    request.future.set_result(result)

            # Update metrics
            elapsed_ms = (time.time() - start_time) * 1000
            self._total_requests += len(batch)
            self._total_latency_ms += elapsed_ms

        except Exception as e:
            # Propagate errors to all futures
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)

    async def _get_embeddings_batch(
        self,
        verse_ids: List[str],
        texts: List[str]
    ) -> np.ndarray:
        """
        Batch embedding generation with caching.
        """
        # Check cache
        embeddings = []
        uncached_indices = []
        uncached_texts = []

        for i, (verse_id, text) in enumerate(zip(verse_ids, texts)):
            cached = self._embedding_cache.get(verse_id)
            if cached is not None:
                embeddings.append(cached)
            else:
                embeddings.append(None)
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Generate missing embeddings
        if uncached_texts:
            new_embeddings = await self._embedding_model.embed_batch_direct(uncached_texts)

            # Update cache and results
            for idx, emb in zip(uncached_indices, new_embeddings):
                verse_id = verse_ids[idx]
                self._embedding_cache.put(verse_id, emb)
                embeddings[idx] = emb

        return np.array(embeddings)

    def get_metrics(self) -> Dict[str, Any]:
        """Get inference pipeline metrics."""
        avg_batch_size = (
            sum(self._batch_sizes) / len(self._batch_sizes)
            if self._batch_sizes else 0
        )
        avg_latency = (
            self._total_latency_ms / self._total_requests
            if self._total_requests else 0
        )

        return {
            "total_requests": self._total_requests,
            "avg_latency_ms": avg_latency,
            "avg_batch_size": avg_batch_size,
            "queue_size": self._request_queue.qsize(),
            "is_warm": self._warm
        }
```

#### 3.2 Connection Pooling for Vector DB

```python
# ml/inference/vector_db_pool.py

import asyncio
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import numpy as np


class QdrantConnectionPool:
    """
    Connection pool for Qdrant vector database.
    Provides efficient connection reuse and query batching.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        pool_size: int = 10,
        collection_name: str = "verse_embeddings"
    ):
        self.host = host
        self.port = port
        self.pool_size = pool_size
        self.collection_name = collection_name
        self._pool: asyncio.Queue[AsyncQdrantClient] = asyncio.Queue()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize connection pool."""
        for _ in range(self.pool_size):
            client = AsyncQdrantClient(host=self.host, port=self.port)
            await self._pool.put(client)
        self._initialized = True

    @asynccontextmanager
    async def get_client(self):
        """Get a client from the pool."""
        if not self._initialized:
            await self.initialize()

        client = await self._pool.get()
        try:
            yield client
        finally:
            await self._pool.put(client)

    async def batch_search(
        self,
        query_vectors: np.ndarray,
        top_k: int = 100,
        score_threshold: float = 0.5
    ) -> List[List[Dict[str, Any]]]:
        """
        Batch similarity search with connection pooling.
        """
        async with self.get_client() as client:
            # Qdrant batch search
            results = await client.search_batch(
                collection_name=self.collection_name,
                requests=[
                    {
                        "vector": vec.tolist(),
                        "limit": top_k,
                        "score_threshold": score_threshold,
                        "with_payload": True
                    }
                    for vec in query_vectors
                ]
            )

        return [
            [
                {
                    "verse_id": hit.payload.get("verse_id"),
                    "score": hit.score,
                    "metadata": hit.payload
                }
                for hit in batch_result
            ]
            for batch_result in results
        ]

    async def upsert_embeddings(
        self,
        verse_ids: List[str],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict]] = None
    ) -> None:
        """
        Batch upsert embeddings with metadata.
        """
        points = [
            PointStruct(
                id=i,
                vector=emb.tolist(),
                payload={
                    "verse_id": verse_id,
                    **(meta or {})
                }
            )
            for i, (verse_id, emb, meta) in enumerate(
                zip(verse_ids, embeddings, metadata or [{}] * len(verse_ids))
            )
        ]

        async with self.get_client() as client:
            await client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
```

---

## 4. Training Pipeline Optimization

### Current State Analysis

**File:** `ml/training/trainer.py`

**Issues Identified:**

1. **No Distributed Training Support**
2. **Manual Gradient Accumulation**
3. **Basic Learning Rate Scheduling**

### Recommended Optimizations

#### 4.1 Distributed Training with FSDP

```python
# ml/training/distributed_trainer.py

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable
import os


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    strategy: str = "ddp"  # ddp, fsdp, deepspeed
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = "nccl"

    # FSDP specific
    sharding_strategy: str = "full_shard"  # full_shard, shard_grad_op, no_shard
    cpu_offload: bool = False
    backward_prefetch: str = "backward_pre"

    # Memory optimization
    activation_checkpointing: bool = True
    param_dtype: str = "fp32"  # fp32, fp16, bf16
    reduce_dtype: str = "fp32"
    buffer_dtype: str = "fp32"


class DistributedBiblosTrainer:
    """
    Production-grade distributed trainer supporting:
    - DDP (DistributedDataParallel)
    - FSDP (Fully Sharded Data Parallel)
    - DeepSpeed integration
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: DistributedConfig,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: Optional[torch.utils.data.Dataset] = None
    ):
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Initialize distributed
        self._setup_distributed()

        # Wrap model
        self.model = self._wrap_model(model)

        # Create data loaders
        self.train_loader = self._create_loader(train_dataset, shuffle=True)
        self.val_loader = (
            self._create_loader(val_dataset, shuffle=False)
            if val_dataset else None
        )

    def _setup_distributed(self) -> None:
        """Initialize distributed training environment."""
        if self.config.world_size > 1:
            # Get rank from environment
            self.config.rank = int(os.environ.get("RANK", 0))
            self.config.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.config.world_size = int(os.environ.get("WORLD_SIZE", 1))

            # Initialize process group
            dist.init_process_group(
                backend=self.config.backend,
                world_size=self.config.world_size,
                rank=self.config.rank
            )

            # Set device
            torch.cuda.set_device(self.config.local_rank)

    def _wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap model for distributed training."""
        device = torch.device(f"cuda:{self.config.local_rank}")
        model = model.to(device)

        if self.config.strategy == "ddp":
            return self._wrap_ddp(model)
        elif self.config.strategy == "fsdp":
            return self._wrap_fsdp(model)
        else:
            return model

    def _wrap_ddp(self, model: torch.nn.Module) -> DDP:
        """Wrap with DistributedDataParallel."""
        return DDP(
            model,
            device_ids=[self.config.local_rank],
            output_device=self.config.local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True  # Optimization for fixed graph
        )

    def _wrap_fsdp(self, model: torch.nn.Module) -> FSDP:
        """Wrap with Fully Sharded Data Parallel."""
        # Configure mixed precision
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16
        }

        mixed_precision = MixedPrecision(
            param_dtype=dtype_map[self.config.param_dtype],
            reduce_dtype=dtype_map[self.config.reduce_dtype],
            buffer_dtype=dtype_map[self.config.buffer_dtype]
        )

        # Configure sharding strategy
        sharding_map = {
            "full_shard": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "no_shard": ShardingStrategy.NO_SHARD
        }

        # Auto-wrap policy
        auto_wrap_policy = size_based_auto_wrap_policy

        return FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            sharding_strategy=sharding_map[self.config.sharding_strategy],
            cpu_offload=CPUOffload(offload_params=self.config.cpu_offload),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            use_orig_params=True  # Required for some optimizers
        )

    def _create_loader(
        self,
        dataset: torch.utils.data.Dataset,
        shuffle: bool = True
    ) -> torch.utils.data.DataLoader:
        """Create distributed data loader."""
        sampler = None
        if self.config.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.config.world_size,
                rank=self.config.rank,
                shuffle=shuffle
            )
            shuffle = False  # Sampler handles shuffling

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

    def train(
        self,
        epochs: int,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Run distributed training loop.
        """
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(epochs):
            # Set epoch for sampler
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            # Training
            train_loss = self._train_epoch(loss_fn, optimizer, scaler)

            # Validation
            val_loss = self._validate_epoch(loss_fn) if self.val_loader else None

            # Learning rate scheduling
            if scheduler:
                scheduler.step()

            # Log on rank 0
            if self.config.rank == 0:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        return {"final_train_loss": train_loss, "final_val_loss": val_loss}

    def _train_epoch(
        self,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler
    ) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            batch = {k: v.cuda() for k, v in batch.items()}

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = self.model(batch)
                loss = loss_fn(outputs, batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        # All-reduce loss across ranks
        if self.config.world_size > 1:
            loss_tensor = torch.tensor(total_loss, device="cuda")
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            total_loss = loss_tensor.item() / self.config.world_size

        return total_loss / len(self.train_loader)

    def _validate_epoch(self, loss_fn: Callable) -> float:
        """Validate one epoch."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.cuda() for k, v in batch.items()}
                outputs = self.model(batch)
                loss = loss_fn(outputs, batch)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)
```

#### 4.2 Optuna Hyperparameter Tuning

```python
# ml/training/hyperopt.py

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.samplers import TPESampler
from typing import Dict, Any, Callable, Optional
import mlflow


class OptunaHyperparameterTuner:
    """
    Hyperparameter optimization with Optuna.
    Integrates with MLflow for experiment tracking.
    """

    def __init__(
        self,
        model_factory: Callable,
        train_fn: Callable,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        study_name: str = "biblos_hyperparam_study",
        storage: str = "sqlite:///optuna.db"
    ):
        self.model_factory = model_factory
        self.train_fn = train_fn
        self.n_trials = n_trials
        self.timeout = timeout

        # Create study with pruning
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            sampler=TPESampler(multivariate=True),
            pruner=HyperbandPruner(
                min_resource=1,
                max_resource=100,
                reduction_factor=3
            ),
            load_if_exists=True
        )

    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define hyperparameter search space."""
        return {
            # Model architecture
            "hidden_channels": trial.suggest_categorical(
                "hidden_channels", [128, 256, 512]
            ),
            "num_layers": trial.suggest_int("num_layers", 2, 5),
            "heads": trial.suggest_categorical("heads", [4, 8, 16]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),

            # Training
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-5, 1e-2, log=True
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay", 1e-6, 1e-2, log=True
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size", [32, 64, 128, 256]
            ),

            # Optimizer
            "optimizer": trial.suggest_categorical(
                "optimizer", ["adam", "adamw", "sgd"]
            ),

            # Scheduler
            "scheduler": trial.suggest_categorical(
                "scheduler", ["cosine", "linear", "step", "none"]
            ),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        }

    def objective(self, trial: optuna.Trial) -> float:
        """Optimization objective function."""
        # Get hyperparameters
        params = self.define_search_space(trial)

        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)

            # Create model
            model = self.model_factory(params)

            # Train and evaluate
            try:
                val_loss = self.train_fn(
                    model=model,
                    params=params,
                    trial=trial  # For pruning callbacks
                )

                mlflow.log_metric("val_loss", val_loss)

                return val_loss

            except optuna.TrialPruned:
                raise
            except Exception as e:
                # Log failure
                mlflow.log_param("error", str(e))
                return float("inf")

    def optimize(self) -> Dict[str, Any]:
        """Run optimization."""
        with mlflow.start_run(run_name="hyperparameter_search"):
            self.study.optimize(
                self.objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                n_jobs=1,  # Use 1 for GPU training
                gc_after_trial=True
            )

            # Log best parameters
            best_params = self.study.best_params
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metric("best_val_loss", self.study.best_value)

            return {
                "best_params": best_params,
                "best_value": self.study.best_value,
                "n_trials": len(self.study.trials)
            }
```

---

## 5. Polars Integration

### Data Pipeline with Polars

```python
# data/polars_integration.py

import polars as pl
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


class PolarsDataPipeline:
    """
    High-performance data pipeline using Polars DataFrames.
    Supports zero-copy transfer to PyTorch tensors.
    """

    @staticmethod
    def load_verses(data_dir: Path) -> pl.LazyFrame:
        """
        Load verse data into Polars LazyFrame for efficient processing.
        """
        return (
            pl.scan_ndjson(data_dir / "**/*.json")
            .select([
                pl.col("verse_id"),
                pl.col("book"),
                pl.col("chapter").cast(pl.Int32),
                pl.col("verse").cast(pl.Int32),
                pl.col("text"),
                pl.col("language").fill_null("unknown")
            ])
            .with_columns([
                # Normalize verse_id
                pl.col("verse_id")
                .str.to_uppercase()
                .str.replace_all(" ", ".")
                .str.replace_all(":", ".")
                .alias("verse_id_normalized")
            ])
        )

    @staticmethod
    def load_crossrefs(data_dir: Path) -> pl.LazyFrame:
        """
        Load cross-reference data into Polars LazyFrame.
        """
        return (
            pl.scan_ndjson(data_dir / "**/*.json")
            .select([
                pl.col("source_ref"),
                pl.col("target_ref"),
                pl.col("connection_type").fill_null("thematic"),
                pl.col("strength").fill_null("moderate"),
                pl.col("confidence").cast(pl.Float64).fill_null(1.0)
            ])
            .filter(pl.col("confidence") >= 0.5)
        )

    @staticmethod
    def to_pytorch_tensor(
        df: pl.DataFrame,
        columns: List[str],
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Zero-copy conversion from Polars DataFrame to PyTorch tensor.
        """
        # Select numeric columns
        numeric_df = df.select(columns)

        # Convert to numpy (zero-copy if possible)
        numpy_array = numeric_df.to_numpy()

        # Create tensor from numpy (shares memory)
        tensor = torch.from_numpy(numpy_array).to(dtype)

        return tensor

    @staticmethod
    def to_pytorch_dataset(
        df: pl.DataFrame,
        feature_columns: List[str],
        label_column: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert Polars DataFrame to PyTorch tensors for training.
        """
        features = PolarsDataPipeline.to_pytorch_tensor(df, feature_columns)
        labels = PolarsDataPipeline.to_pytorch_tensor(
            df, [label_column], torch.long
        ).squeeze()

        return features, labels

    @staticmethod
    def batch_collate_polars(
        batch: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Custom collate function that uses Polars for efficient batching.
        """
        # Convert batch to Polars DataFrame
        df = pl.DataFrame(batch)

        result = {}

        # Handle text columns
        text_columns = [c for c in df.columns if df[c].dtype == pl.Utf8]
        for col in text_columns:
            result[col] = df[col].to_list()

        # Handle numeric columns - zero-copy to tensor
        numeric_columns = [c for c in df.columns if df[c].dtype in [pl.Int64, pl.Float64]]
        for col in numeric_columns:
            result[col] = torch.from_numpy(df[col].to_numpy())

        return result


class PolarsEmbeddingStore:
    """
    Store and retrieve embeddings using Polars for efficient I/O.
    """

    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def save_embeddings(
        self,
        verse_ids: List[str],
        embeddings: np.ndarray,
        metadata: Optional[Dict[str, List]] = None
    ) -> None:
        """
        Save embeddings to Parquet format for efficient storage.
        """
        # Create DataFrame with embeddings
        data = {
            "verse_id": verse_ids,
        }

        # Add embedding columns
        for i in range(embeddings.shape[1]):
            data[f"emb_{i}"] = embeddings[:, i].tolist()

        # Add metadata
        if metadata:
            data.update(metadata)

        df = pl.DataFrame(data)

        # Save to Parquet with compression
        df.write_parquet(
            self.storage_path / "embeddings.parquet",
            compression="zstd",
            statistics=True,
            row_group_size=10000
        )

    def load_embeddings(
        self,
        verse_ids: Optional[List[str]] = None
    ) -> Tuple[List[str], np.ndarray]:
        """
        Load embeddings from Parquet storage.
        """
        # Lazy load for efficiency
        lf = pl.scan_parquet(self.storage_path / "embeddings.parquet")

        # Filter if verse_ids provided
        if verse_ids:
            lf = lf.filter(pl.col("verse_id").is_in(verse_ids))

        # Collect
        df = lf.collect()

        # Extract embeddings
        emb_columns = [c for c in df.columns if c.startswith("emb_")]
        embeddings = df.select(emb_columns).to_numpy()
        verse_ids = df["verse_id"].to_list()

        return verse_ids, embeddings

    def streaming_load(
        self,
        batch_size: int = 10000
    ):
        """
        Stream embeddings in batches for memory efficiency.
        """
        lf = pl.scan_parquet(self.storage_path / "embeddings.parquet")

        # Get total rows
        total_rows = lf.select(pl.count()).collect().item()

        for offset in range(0, total_rows, batch_size):
            batch = (
                lf
                .slice(offset, batch_size)
                .collect()
            )

            emb_columns = [c for c in batch.columns if c.startswith("emb_")]
            embeddings = batch.select(emb_columns).to_numpy()
            verse_ids = batch["verse_id"].to_list()

            yield verse_ids, embeddings
```

---

## 6. Feature Store Design

### Feature Caching with Redis and DVC

```python
# ml/features/feature_store.py

import redis
import hashlib
import json
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import subprocess


@dataclass
class FeatureStoreConfig:
    """Feature store configuration."""
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 1
    cache_ttl_seconds: int = 86400  # 24 hours
    dvc_remote: str = "s3://biblos-features"
    local_cache_dir: str = "features/cache"


class BiblosFeatureStore:
    """
    Feature store for BIBLOS ML pipeline.

    Features:
    - Redis for online serving (low latency)
    - DVC for versioned offline storage
    - Automatic feature refresh
    """

    FEATURE_SCHEMAS = {
        "verse_embedding": {"dim": 768, "dtype": "float32"},
        "morphological_features": {"dim": 64, "dtype": "float32"},
        "syntactic_features": {"dim": 32, "dtype": "float32"},
        "theological_features": {"dim": 128, "dtype": "float32"},
    }

    def __init__(self, config: Optional[FeatureStoreConfig] = None):
        self.config = config or FeatureStoreConfig()
        self._redis = redis.Redis(
            host=self.config.redis_host,
            port=self.config.redis_port,
            db=self.config.redis_db,
            decode_responses=False
        )
        self._local_cache = Path(self.config.local_cache_dir)
        self._local_cache.mkdir(parents=True, exist_ok=True)

    def _make_key(self, verse_id: str, feature_name: str, version: str = "latest") -> str:
        """Create cache key."""
        return f"feat:{feature_name}:{version}:{verse_id}"

    def get_feature(
        self,
        verse_id: str,
        feature_name: str,
        version: str = "latest"
    ) -> Optional[np.ndarray]:
        """
        Get single feature from store.
        Tries Redis first, then local cache, then DVC.
        """
        key = self._make_key(verse_id, feature_name, version)

        # Try Redis
        cached = self._redis.get(key)
        if cached:
            return np.frombuffer(cached, dtype=np.float32)

        # Try local cache
        local_path = self._local_cache / feature_name / version / f"{verse_id}.npy"
        if local_path.exists():
            feature = np.load(local_path)
            # Populate Redis
            self._redis.setex(key, self.config.cache_ttl_seconds, feature.tobytes())
            return feature

        return None

    def get_features_batch(
        self,
        verse_ids: List[str],
        feature_name: str,
        version: str = "latest"
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Batch feature retrieval with pipeline for efficiency.

        Returns:
            - features: numpy array of features
            - missing_ids: list of verse_ids not found
        """
        keys = [self._make_key(vid, feature_name, version) for vid in verse_ids]

        # Batch get from Redis
        pipeline = self._redis.pipeline()
        for key in keys:
            pipeline.get(key)
        results = pipeline.execute()

        features = []
        missing_ids = []

        for verse_id, cached in zip(verse_ids, results):
            if cached:
                feature = np.frombuffer(cached, dtype=np.float32)
                features.append(feature)
            else:
                missing_ids.append(verse_id)
                features.append(None)

        # Load missing from local cache
        for i, vid in enumerate(verse_ids):
            if features[i] is None:
                local_path = self._local_cache / feature_name / version / f"{vid}.npy"
                if local_path.exists():
                    feature = np.load(local_path)
                    features[i] = feature
                    # Populate Redis
                    key = keys[i]
                    self._redis.setex(
                        key,
                        self.config.cache_ttl_seconds,
                        feature.tobytes()
                    )

        # Filter out still-missing
        valid_features = [f for f in features if f is not None]
        missing_ids = [vid for vid, f in zip(verse_ids, features) if f is None]

        return np.array(valid_features) if valid_features else np.array([]), missing_ids

    def put_feature(
        self,
        verse_id: str,
        feature_name: str,
        feature: np.ndarray,
        version: str = "latest"
    ) -> None:
        """Store single feature."""
        key = self._make_key(verse_id, feature_name, version)

        # Store in Redis
        self._redis.setex(key, self.config.cache_ttl_seconds, feature.tobytes())

        # Store in local cache
        local_dir = self._local_cache / feature_name / version
        local_dir.mkdir(parents=True, exist_ok=True)
        np.save(local_dir / f"{verse_id}.npy", feature)

    def put_features_batch(
        self,
        verse_ids: List[str],
        feature_name: str,
        features: np.ndarray,
        version: str = "latest"
    ) -> None:
        """Batch feature storage."""
        # Redis pipeline
        pipeline = self._redis.pipeline()
        for verse_id, feature in zip(verse_ids, features):
            key = self._make_key(verse_id, feature_name, version)
            pipeline.setex(key, self.config.cache_ttl_seconds, feature.tobytes())
        pipeline.execute()

        # Local cache
        local_dir = self._local_cache / feature_name / version
        local_dir.mkdir(parents=True, exist_ok=True)
        for verse_id, feature in zip(verse_ids, features):
            np.save(local_dir / f"{verse_id}.npy", feature)

    def version_features(self, feature_name: str, version_tag: str) -> None:
        """
        Version features using DVC.
        Creates a new version and pushes to remote.
        """
        feature_dir = self._local_cache / feature_name / "latest"

        if not feature_dir.exists():
            raise FileNotFoundError(f"No features found for {feature_name}")

        # Add to DVC
        subprocess.run(
            ["dvc", "add", str(feature_dir)],
            check=True
        )

        # Create version directory
        version_dir = self._local_cache / feature_name / version_tag
        version_dir.parent.mkdir(parents=True, exist_ok=True)

        # Copy files
        import shutil
        if version_dir.exists():
            shutil.rmtree(version_dir)
        shutil.copytree(feature_dir, version_dir)

        # Push to remote
        subprocess.run(
            ["dvc", "push", str(feature_dir) + ".dvc"],
            check=True
        )

        # Git commit
        subprocess.run([
            "git", "add",
            str(feature_dir) + ".dvc",
            str(feature_dir) + ".gitignore"
        ], check=True)
        subprocess.run([
            "git", "commit", "-m",
            f"Version features: {feature_name} @ {version_tag}"
        ], check=True)

    def load_version(self, feature_name: str, version_tag: str) -> None:
        """Load a specific feature version from DVC."""
        subprocess.run(
            ["dvc", "checkout",
             str(self._local_cache / feature_name / version_tag) + ".dvc"],
            check=True
        )
```

---

## 7. Model Serving Optimization

### TorchServe Configuration

```python
# ml/serving/torchserve_handler.py

import torch
import json
import logging
from abc import ABC
from typing import List, Dict, Any
from ts.torch_handler.base_handler import BaseHandler
import numpy as np


class BiblosEmbeddingHandler(BaseHandler, ABC):
    """
    TorchServe handler for BIBLOS embedding model.
    Supports batched inference with dynamic batching.
    """

    def __init__(self):
        super().__init__()
        self.initialized = False
        self.tokenizer = None
        self.embedding_model = None

    def initialize(self, context):
        """Initialize model and tokenizer."""
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Load model
        self.embedding_model = torch.jit.load(
            f"{model_dir}/model.pt",
            map_location=self.device
        )
        self.embedding_model.eval()

        self.initialized = True
        logging.info("BIBLOS Embedding Handler initialized")

    def preprocess(self, requests: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Preprocess batch of requests.
        """
        texts = []
        for request in requests:
            data = request.get("data") or request.get("body")
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8")
            if isinstance(data, str):
                data = json.loads(data)
            texts.append(data.get("text", ""))

        # Tokenize batch
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        return {k: v.to(self.device) for k, v in encoded.items()}

    def inference(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Run inference with mixed precision.
        """
        with torch.no_grad(), torch.cuda.amp.autocast():
            embeddings = self.embedding_model(**inputs)
        return embeddings

    def postprocess(self, inference_output: torch.Tensor) -> List[Dict]:
        """
        Convert embeddings to JSON response.
        """
        embeddings = inference_output.cpu().numpy()

        return [
            {
                "embedding": emb.tolist(),
                "dimensions": len(emb)
            }
            for emb in embeddings
        ]


# TorchServe config: config.properties
"""
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
number_of_netty_threads=4
job_queue_size=100
default_workers_per_model=4
model_store=/models
load_models=biblos_embedding.mar

# Dynamic batching
batch_size=32
max_batch_delay=100

# GPU settings
enable_envvars_config=true
prefer_direct_buffer=true
"""
```

### Triton Inference Server Configuration

```python
# ml/serving/triton_config.py

"""
Triton Inference Server configuration for BIBLOS models.

Directory structure:
models/
  biblos_embedding/
    config.pbtxt
    1/
      model.onnx
  biblos_gnn/
    config.pbtxt
    1/
      model.pt
  biblos_ensemble/
    config.pbtxt
"""

EMBEDDING_CONFIG = """
name: "biblos_embedding"
platform: "onnxruntime_onnx"
max_batch_size: 64

input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }
]

output [
  {
    name: "embeddings"
    data_type: TYPE_FP32
    dims: [ 768 ]
  }
]

instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 16, 32, 64 ]
  max_queue_delay_microseconds: 100000
}

optimization {
  input_pinned_memory { enable: true }
  output_pinned_memory { enable: true }
  cuda {
    graphs: true
    graph_spec {
      batch_size: 1
      input [
        { key: "input_ids", value: { dim: [ 512 ] } },
        { key: "attention_mask", value: { dim: [ 512 ] } }
      ]
    }
    graph_spec {
      batch_size: 32
      input [
        { key: "input_ids", value: { dim: [ 512 ] } },
        { key: "attention_mask", value: { dim: [ 512 ] } }
      ]
    }
  }
}
"""

GNN_CONFIG = """
name: "biblos_gnn"
platform: "pytorch_libtorch"
max_batch_size: 1

input [
  {
    name: "node_features"
    data_type: TYPE_FP32
    dims: [ -1, 768 ]
  },
  {
    name: "edge_index"
    data_type: TYPE_INT64
    dims: [ 2, -1 ]
  }
]

output [
  {
    name: "node_embeddings"
    data_type: TYPE_FP32
    dims: [ -1, 256 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
"""

ENSEMBLE_CONFIG = """
name: "biblos_ensemble"
platform: "ensemble"
max_batch_size: 64

input [
  {
    name: "text"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]

output [
  {
    name: "predictions"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }
]

ensemble_scheduling {
  step [
    {
      model_name: "biblos_embedding"
      model_version: -1
      input_map {
        key: "input_ids"
        value: "input_ids"
      }
      input_map {
        key: "attention_mask"
        value: "attention_mask"
      }
      output_map {
        key: "embeddings"
        value: "verse_embeddings"
      }
    },
    {
      model_name: "biblos_gnn"
      model_version: -1
      input_map {
        key: "node_features"
        value: "verse_embeddings"
      }
      output_map {
        key: "node_embeddings"
        value: "gnn_embeddings"
      }
    },
    {
      model_name: "biblos_classifier"
      model_version: -1
      input_map {
        key: "embeddings"
        value: "gnn_embeddings"
      }
      output_map {
        key: "predictions"
        value: "predictions"
      }
    }
  ]
}
"""
```

---

## 8. Configuration Updates

### Updated ml/config.py

```python
# ml/config.py - Optimized configuration

from typing import Dict, List, Optional, Any
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class OptimizedEmbeddingConfig(BaseModel):
    """Optimized embedding configuration."""
    models: Dict[str, 'EmbeddingModelConfig'] = Field(default_factory=dict)

    # Batching
    batch_size: int = 64
    max_batch_wait_ms: float = 50.0
    prefetch_factor: int = 2

    # Caching
    cache_backend: str = "redis"  # redis, disk, memory
    cache_dir: str = "data/embeddings_cache"
    redis_host: str = "localhost"
    redis_port: int = 6379
    cache_ttl_seconds: int = 86400
    max_cache_items: int = 500000

    # Optimization
    use_onnx: bool = True
    use_fp16: bool = True
    use_quantization: bool = False

    # Fusion
    normalize_embeddings: bool = True
    fusion_method: str = "weighted_average"  # weighted_average, concat, attention


class OptimizedGNNConfig(BaseModel):
    """Optimized GNN configuration."""
    hidden_channels: int = 256
    num_layers: int = 3
    dropout: float = 0.2
    heads: int = 8

    # Optimization
    use_checkpoint: bool = True  # Gradient checkpointing
    compile_model: bool = True  # torch.compile

    # Mini-batch training
    neighbor_sample_sizes: List[int] = [25, 10]
    batch_size: int = 1024

    # Candidate generation
    use_faiss: bool = True
    faiss_nlist: int = 100
    faiss_nprobe: int = 10
    top_k_candidates: int = 100


class OptimizedTrainingConfig(BaseModel):
    """Optimized training configuration."""
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 100

    # Distributed
    distributed_strategy: str = "ddp"  # ddp, fsdp, deepspeed
    world_size: int = 1

    # Optimization
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    use_amp: bool = True
    use_compile: bool = True

    # Scheduling
    scheduler: str = "cosine"
    warmup_ratio: float = 0.1

    # Early stopping
    early_stopping_patience: int = 10
    min_delta: float = 1e-4


class OptimizedInferenceConfig(BaseModel):
    """Optimized inference configuration."""
    batch_size: int = 32
    max_batch_wait_ms: float = 100.0
    max_candidates: int = 100
    min_confidence: float = 0.5

    # Workers
    num_worker_threads: int = 2
    max_concurrent_batches: int = 4

    # Warm-up
    warm_up_samples: int = 10

    # Serving
    serving_backend: str = "torchserve"  # torchserve, triton, custom


class OptimizedMLConfig(BaseSettings):
    """Master optimized ML configuration."""
    embeddings: OptimizedEmbeddingConfig = Field(default_factory=OptimizedEmbeddingConfig)
    gnn: OptimizedGNNConfig = Field(default_factory=OptimizedGNNConfig)
    training: OptimizedTrainingConfig = Field(default_factory=OptimizedTrainingConfig)
    inference: OptimizedInferenceConfig = Field(default_factory=OptimizedInferenceConfig)

    # General
    device: str = "cuda"
    seed: int = 42
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")

    # MLflow
    mlflow_tracking_uri: str = "sqlite:///mlruns.db"
    mlflow_experiment: str = "biblos-v2-optimized"

    # Feature store
    feature_store_backend: str = "redis"
    feature_store_dvc_remote: str = "s3://biblos-features"

    model_config = {
        "env_prefix": "BIBLOS_ML_",
        "env_nested_delimiter": "__"
    }


# Create optimized config singleton
optimized_config = OptimizedMLConfig()
```

---

## 9. Benchmark Methodology

### Performance Benchmarking Suite

```python
# ml/benchmarks/benchmark_suite.py

import torch
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Callable
import json
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Result from a single benchmark."""
    name: str
    throughput: float  # items/second
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    memory_mb: float
    gpu_utilization: float


class BiblosBenchmarkSuite:
    """
    Comprehensive benchmark suite for BIBLOS ML pipeline.
    """

    def __init__(self, output_dir: Path = Path("benchmarks")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []

    def benchmark_embeddings(
        self,
        embedder,
        texts: List[str],
        batch_sizes: List[int] = [1, 8, 16, 32, 64, 128],
        warmup_iterations: int = 10,
        test_iterations: int = 100
    ) -> Dict[int, BenchmarkResult]:
        """
        Benchmark embedding throughput and latency.
        """
        results = {}

        for batch_size in batch_sizes:
            # Warmup
            for _ in range(warmup_iterations):
                batch = texts[:batch_size]
                _ = embedder.encode(batch)

            # Benchmark
            latencies = []

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            for _ in range(test_iterations):
                batch = texts[:batch_size]

                start = time.perf_counter()
                _ = embedder.encode(batch)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                latencies.append((time.perf_counter() - start) * 1000)

            # Calculate metrics
            latencies = np.array(latencies)
            throughput = batch_size / (np.mean(latencies) / 1000)

            result = BenchmarkResult(
                name=f"embedding_batch_{batch_size}",
                throughput=throughput,
                latency_p50_ms=np.percentile(latencies, 50),
                latency_p95_ms=np.percentile(latencies, 95),
                latency_p99_ms=np.percentile(latencies, 99),
                memory_mb=self._get_gpu_memory_mb(),
                gpu_utilization=self._get_gpu_utilization()
            )

            results[batch_size] = result
            self.results.append(result)

        return results

    def benchmark_gnn(
        self,
        model: torch.nn.Module,
        node_counts: List[int] = [100, 1000, 10000],
        edge_densities: List[float] = [0.01, 0.05, 0.1],
        test_iterations: int = 50
    ) -> Dict[str, BenchmarkResult]:
        """
        Benchmark GNN forward pass.
        """
        model.eval()
        device = next(model.parameters()).device
        results = {}

        for num_nodes in node_counts:
            for density in edge_densities:
                num_edges = int(num_nodes * num_nodes * density)

                # Create dummy data
                x = torch.randn(num_nodes, 768, device=device)
                edge_index = torch.randint(
                    0, num_nodes, (2, num_edges), device=device
                )

                # Warmup
                with torch.no_grad():
                    for _ in range(10):
                        _ = model(x, edge_index)

                # Benchmark
                latencies = []

                torch.cuda.synchronize()

                for _ in range(test_iterations):
                    start = time.perf_counter()

                    with torch.no_grad():
                        _ = model(x, edge_index)

                    torch.cuda.synchronize()
                    latencies.append((time.perf_counter() - start) * 1000)

                latencies = np.array(latencies)
                key = f"gnn_nodes_{num_nodes}_density_{density}"

                result = BenchmarkResult(
                    name=key,
                    throughput=num_nodes / (np.mean(latencies) / 1000),
                    latency_p50_ms=np.percentile(latencies, 50),
                    latency_p95_ms=np.percentile(latencies, 95),
                    latency_p99_ms=np.percentile(latencies, 99),
                    memory_mb=self._get_gpu_memory_mb(),
                    gpu_utilization=self._get_gpu_utilization()
                )

                results[key] = result
                self.results.append(result)

        return results

    def benchmark_inference_pipeline(
        self,
        pipeline,
        test_verses: List[Dict[str, str]],
        concurrency_levels: List[int] = [1, 4, 8, 16, 32],
        test_duration_seconds: int = 30
    ) -> Dict[int, BenchmarkResult]:
        """
        Benchmark end-to-end inference pipeline.
        """
        import asyncio

        results = {}

        for concurrency in concurrency_levels:
            async def run_benchmark():
                completed = 0
                latencies = []

                start_time = time.time()

                while time.time() - start_time < test_duration_seconds:
                    tasks = []
                    for _ in range(concurrency):
                        verse = test_verses[completed % len(test_verses)]
                        task_start = time.perf_counter()
                        task = asyncio.create_task(
                            pipeline.infer(verse["verse_id"], verse["text"])
                        )
                        tasks.append((task, task_start))

                    for task, task_start in tasks:
                        await task
                        latencies.append((time.perf_counter() - task_start) * 1000)
                        completed += 1

                return completed, latencies

            completed, latencies = asyncio.run(run_benchmark())
            latencies = np.array(latencies)

            result = BenchmarkResult(
                name=f"inference_concurrency_{concurrency}",
                throughput=completed / test_duration_seconds,
                latency_p50_ms=np.percentile(latencies, 50),
                latency_p95_ms=np.percentile(latencies, 95),
                latency_p99_ms=np.percentile(latencies, 99),
                memory_mb=self._get_gpu_memory_mb(),
                gpu_utilization=self._get_gpu_utilization()
            )

            results[concurrency] = result
            self.results.append(result)

        return results

    def _get_gpu_memory_mb(self) -> float:
        """Get current GPU memory usage."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0.0

    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu / 100.0
        except Exception:
            return 0.0

    def save_results(self, filename: str = "benchmark_results.json") -> None:
        """Save benchmark results to JSON."""
        results_dict = [
            {
                "name": r.name,
                "throughput": r.throughput,
                "latency_p50_ms": r.latency_p50_ms,
                "latency_p95_ms": r.latency_p95_ms,
                "latency_p99_ms": r.latency_p99_ms,
                "memory_mb": r.memory_mb,
                "gpu_utilization": r.gpu_utilization
            }
            for r in self.results
        ]

        with open(self.output_dir / filename, "w") as f:
            json.dump(results_dict, f, indent=2)

    def generate_report(self) -> str:
        """Generate benchmark report."""
        report = "# BIBLOS ML Pipeline Benchmark Report\n\n"

        # Group by benchmark type
        embedding_results = [r for r in self.results if r.name.startswith("embedding")]
        gnn_results = [r for r in self.results if r.name.startswith("gnn")]
        inference_results = [r for r in self.results if r.name.startswith("inference")]

        if embedding_results:
            report += "## Embedding Benchmarks\n\n"
            report += "| Batch Size | Throughput (items/s) | P50 (ms) | P95 (ms) | P99 (ms) |\n"
            report += "|------------|---------------------|----------|----------|----------|\n"
            for r in embedding_results:
                batch_size = r.name.split("_")[-1]
                report += f"| {batch_size} | {r.throughput:.2f} | {r.latency_p50_ms:.2f} | {r.latency_p95_ms:.2f} | {r.latency_p99_ms:.2f} |\n"
            report += "\n"

        if gnn_results:
            report += "## GNN Benchmarks\n\n"
            report += "| Configuration | Throughput (nodes/s) | P50 (ms) | P95 (ms) | Memory (MB) |\n"
            report += "|---------------|---------------------|----------|----------|-------------|\n"
            for r in gnn_results:
                report += f"| {r.name} | {r.throughput:.2f} | {r.latency_p50_ms:.2f} | {r.latency_p95_ms:.2f} | {r.memory_mb:.2f} |\n"
            report += "\n"

        if inference_results:
            report += "## Inference Pipeline Benchmarks\n\n"
            report += "| Concurrency | Throughput (req/s) | P50 (ms) | P95 (ms) | P99 (ms) |\n"
            report += "|-------------|-------------------|----------|----------|----------|\n"
            for r in inference_results:
                concurrency = r.name.split("_")[-1]
                report += f"| {concurrency} | {r.throughput:.2f} | {r.latency_p50_ms:.2f} | {r.latency_p95_ms:.2f} | {r.latency_p99_ms:.2f} |\n"

        return report
```

---

## 10. Migration Plan

### Phase 1: Foundation (Week 1-2)

1. **Update Dependencies**
   - Add Polars, FAISS, ONNX Runtime to `pyproject.toml`
   - Update PyTorch to 2.1+
   - Add Redis client

2. **Implement Optimized Cache**
   - Replace list-based LRU with OrderedDict
   - Add Redis integration
   - Implement batch cache operations

### Phase 2: Embedding Optimization (Week 2-3)

1. **Implement Batching Embedder**
   - Create `ml/embeddings/optimized_ensemble.py`
   - Add async request coalescing
   - Implement GPU batching

2. **ONNX Export**
   - Export embedding models to ONNX
   - Benchmark against PyTorch baseline

### Phase 3: GNN Optimization (Week 3-4)

1. **FAISS Integration**
   - Replace O(n^2) candidate generation
   - Build verse embedding index
   - Benchmark ANN accuracy vs. exact

2. **Mini-Batch Training**
   - Implement NeighborLoader integration
   - Add gradient checkpointing
   - Enable torch.compile

### Phase 4: Inference Pipeline (Week 4-5)

1. **Async Batching**
   - Implement request coalescing
   - Add model warm-up
   - Implement connection pooling

2. **Vector DB Integration**
   - Set up Qdrant
   - Implement batch search
   - Add caching layer

### Phase 5: Training & Serving (Week 5-6)

1. **Distributed Training**
   - Implement DDP wrapper
   - Add FSDP support
   - Integrate Optuna

2. **Model Serving**
   - Configure TorchServe
   - Set up Triton (optional)
   - Implement A/B testing

### Phase 6: Feature Store & Validation (Week 6-7)

1. **Feature Store**
   - Implement Redis-backed store
   - Add DVC versioning
   - Create feature pipelines

2. **Benchmarking**
   - Run full benchmark suite
   - Compare against baseline
   - Document improvements

---

## Summary of Expected Improvements

| Component | Current | Optimized | Improvement |
|-----------|---------|-----------|-------------|
| Embedding Throughput | ~50 items/s | ~500 items/s | 10x |
| Embedding Latency (P50) | ~200ms | ~20ms | 10x |
| GNN Training | Single GPU | Multi-GPU FSDP | 4-8x |
| Candidate Generation | O(n^2) | O(n log n) | 100x+ |
| Inference Latency (P95) | ~500ms | ~50ms | 10x |
| Memory Usage | High | 50% reduction | 2x |
| Cache Hit Rate | ~60% | ~90% | 1.5x |

---

## References

1. PyTorch 2.0 Compile: https://pytorch.org/docs/stable/torch.compiler.html
2. FSDP Documentation: https://pytorch.org/docs/stable/fsdp.html
3. FAISS Library: https://github.com/facebookresearch/faiss
4. ONNX Runtime: https://onnxruntime.ai/
5. Triton Inference Server: https://github.com/triton-inference-server/server
6. TorchServe: https://pytorch.org/serve/
7. Optuna: https://optuna.org/
8. Polars: https://pola.rs/
