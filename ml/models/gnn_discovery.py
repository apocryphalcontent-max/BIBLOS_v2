"""
BIBLOS v2 - Graph Neural Network for Cross-Reference Discovery

Uses PyTorch Geometric for learning over the biblical verse graph
to discover new cross-references based on structural patterns.
"""
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import logging

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    # Stub classes for type hints
    class nn:
        class Module:
            pass


@dataclass
class CrossRefPrediction:
    """Prediction for a potential cross-reference."""
    source_verse: str
    target_verse: str
    confidence: float
    connection_type: str
    connection_type_confidence: float
    features: Dict[str, float]


class CrossRefGNN(nn.Module if TORCH_GEOMETRIC_AVAILABLE else object):
    """
    Graph Neural Network for biblical cross-reference discovery.

    Architecture:
    - Input: Node features (verse embeddings)
    - Hidden: Multiple GAT layers with attention
    - Output: Edge prediction scores + connection type classification

    Edge Types:
    - thematic, verbal, conceptual, historical, typological,
    - prophetic, liturgical, narrative, genealogical, geographical
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
        num_connection_types: int = 10
    ):
        if not TORCH_GEOMETRIC_AVAILABLE:
            self.logger = logging.getLogger("biblos.ml.models.gnn")
            self.logger.warning("PyTorch Geometric not available")
            return

        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_connection_types = num_connection_types
        self.logger = logging.getLogger("biblos.ml.models.gnn")

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_channels if i == 0 else hidden_channels * heads
            self.gat_layers.append(
                GATv2Conv(
                    in_dim,
                    hidden_channels,
                    heads=heads,
                    dropout=dropout,
                    concat=True if i < num_layers - 1 else False
                )
            )

        # Edge prediction head
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )

        # Connection type classifier
        self.type_classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_connection_types)
        )

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels * heads if i < num_layers - 1 else hidden_channels)
            for i in range(num_layers)
        ])

        # Embedding cache for mutual transformation metric
        self._node_embeddings: Dict[str, np.ndarray] = {}
        self._verse_to_index: Dict[str, int] = {}
        self._last_embeddings: Optional['torch.Tensor'] = None

    def forward(
        self,
        x: 'torch.Tensor',
        edge_index: 'torch.Tensor',
        batch: Optional['torch.Tensor'] = None
    ) -> Tuple['torch.Tensor', 'torch.Tensor']:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]

        Returns:
            edge_scores: Edge prediction scores [num_edges]
            node_embeddings: Updated node embeddings [num_nodes, hidden_channels]
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GAT layers
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            x = gat(x, edge_index)
            x = norm(x)
            if i < len(self.gat_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def predict_edges(
        self,
        node_embeddings: 'torch.Tensor',
        source_idx: 'torch.Tensor',
        target_idx: 'torch.Tensor'
    ) -> Tuple['torch.Tensor', 'torch.Tensor']:
        """
        Predict edge scores and connection types.

        Args:
            node_embeddings: Node embeddings [num_nodes, hidden_channels]
            source_idx: Source node indices [num_candidates]
            target_idx: Target node indices [num_candidates]

        Returns:
            edge_scores: Prediction scores [num_candidates]
            type_logits: Connection type logits [num_candidates, num_types]
        """
        # Concatenate source and target embeddings
        source_emb = node_embeddings[source_idx]
        target_emb = node_embeddings[target_idx]
        pair_emb = torch.cat([source_emb, target_emb], dim=-1)

        # Predict edge existence
        edge_scores = self.edge_predictor(pair_emb).squeeze(-1)

        # Predict connection type
        type_logits = self.type_classifier(pair_emb)

        return edge_scores, type_logits

    def discover_crossrefs(
        self,
        node_embeddings: np.ndarray,
        verse_ids: List[str],
        existing_edges: Optional[List[Tuple[int, int]]] = None,
        top_k: int = 100,
        min_confidence: float = 0.5
    ) -> List[CrossRefPrediction]:
        """
        Discover potential cross-references.

        Args:
            node_embeddings: Verse embeddings [num_verses, embedding_dim]
            verse_ids: List of verse identifiers
            existing_edges: Known edges to exclude
            top_k: Maximum predictions to return
            min_confidence: Minimum confidence threshold

        Returns:
            List of CrossRefPrediction objects
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            return []

        self.eval()
        device = next(self.parameters()).device

        # Convert to tensor
        x = torch.tensor(node_embeddings, dtype=torch.float32, device=device)
        num_verses = len(verse_ids)

        # Build existing edge set for exclusion
        existing_set = set(existing_edges) if existing_edges else set()

        # Generate all candidate pairs
        candidates = []
        for i in range(num_verses):
            for j in range(num_verses):
                if i != j and (i, j) not in existing_set:
                    candidates.append((i, j))

        if not candidates:
            return []

        # Batch prediction
        source_idx = torch.tensor([c[0] for c in candidates], device=device)
        target_idx = torch.tensor([c[1] for c in candidates], device=device)

        with torch.no_grad():
            # Get node embeddings through GNN
            # For simplicity, use input embeddings directly if no edge_index
            node_emb = self.input_proj(x)
            node_emb = F.relu(node_emb)

            # Predict
            edge_scores, type_logits = self.predict_edges(
                node_emb, source_idx, target_idx
            )

            # Get type predictions
            type_probs = F.softmax(type_logits, dim=-1)
            type_preds = type_probs.argmax(dim=-1)
            type_confs = type_probs.max(dim=-1).values

        # Filter and format results
        predictions = []
        for idx, (i, j) in enumerate(candidates):
            confidence = edge_scores[idx].item()
            if confidence >= min_confidence:
                predictions.append(CrossRefPrediction(
                    source_verse=verse_ids[i],
                    target_verse=verse_ids[j],
                    confidence=confidence,
                    connection_type=self.CONNECTION_TYPES[type_preds[idx].item()],
                    connection_type_confidence=type_confs[idx].item(),
                    features={}
                ))

        # Sort by confidence and return top_k
        predictions.sort(key=lambda x: x.confidence, reverse=True)
        return predictions[:top_k]

    def cache_embeddings(
        self,
        verse_ids: List[str],
        embeddings: 'torch.Tensor'
    ) -> None:
        """
        Cache node embeddings for later retrieval.

        Used by the mutual transformation metric to access embeddings
        before and after GNN refinement.

        Args:
            verse_ids: List of verse identifiers.
            embeddings: Tensor of embeddings [num_verses, embedding_dim].
        """
        self._verse_to_index = {vid: i for i, vid in enumerate(verse_ids)}
        self._last_embeddings = embeddings.detach()

        # Also store as numpy for direct access
        emb_numpy = embeddings.detach().cpu().numpy()
        for vid, idx in self._verse_to_index.items():
            self._node_embeddings[vid] = emb_numpy[idx]

    def get_node_embedding(
        self,
        verse_id: str
    ) -> Optional[np.ndarray]:
        """
        Get cached embedding for a specific verse.

        Args:
            verse_id: The verse identifier (e.g., "GEN.1.1").

        Returns:
            Numpy array of embedding or None if not cached.
        """
        return self._node_embeddings.get(verse_id)

    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Get all cached embeddings.

        Returns:
            Dictionary mapping verse_id to embedding.
        """
        return self._node_embeddings.copy()

    def clear_embedding_cache(self) -> None:
        """Clear the embedding cache."""
        self._node_embeddings.clear()
        self._verse_to_index.clear()
        self._last_embeddings = None

    def get_embeddings_for_pair(
        self,
        source_id: str,
        target_id: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get embeddings for a verse pair.

        Convenience method for mutual transformation metric.

        Args:
            source_id: Source verse identifier.
            target_id: Target verse identifier.

        Returns:
            Tuple of (source_embedding, target_embedding).
        """
        return (
            self._node_embeddings.get(source_id),
            self._node_embeddings.get(target_id)
        )
