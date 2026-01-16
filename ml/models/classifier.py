"""
BIBLOS v2 - Connection Type Classifier

Multi-label classifier for biblical cross-reference connection types.
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import logging

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn:
        class Module:
            pass


@dataclass
class ClassificationResult:
    """Result from connection type classification."""
    source_ref: str
    target_ref: str
    primary_type: str
    primary_confidence: float
    all_types: Dict[str, float]
    multilabel: List[str]


class ConnectionTypeClassifier(nn.Module if TORCH_AVAILABLE else object):
    """
    Classifier for cross-reference connection types.

    Types:
    - thematic: Shared themes/concepts
    - verbal: Word/phrase similarities
    - conceptual: Conceptual parallels
    - historical: Historical references
    - typological: Type/antitype relationships
    - prophetic: Prophecy/fulfillment
    - liturgical: Liturgical connections
    - narrative: Narrative parallels
    - genealogical: Genealogical links
    - geographical: Geographic references
    """

    TYPES = [
        "thematic", "verbal", "conceptual", "historical", "typological",
        "prophetic", "liturgical", "narrative", "genealogical", "geographical"
    ]

    def __init__(
        self,
        input_dim: int = 768 * 2,  # Concatenated embeddings
        hidden_dim: int = 512,
        num_classes: int = 10,
        dropout: float = 0.3,
        multilabel: bool = True
    ):
        if not TORCH_AVAILABLE:
            self.logger = logging.getLogger("biblos.ml.models.classifier")
            self.logger.warning("PyTorch not available")
            return

        super().__init__()

        self.multilabel = multilabel
        self.logger = logging.getLogger("biblos.ml.models.classifier")

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Attention for embedding fusion
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        source_emb: 'torch.Tensor',
        target_emb: 'torch.Tensor'
    ) -> 'torch.Tensor':
        """
        Forward pass.

        Args:
            source_emb: Source verse embeddings [batch, embed_dim]
            target_emb: Target verse embeddings [batch, embed_dim]

        Returns:
            logits: Class logits [batch, num_classes]
        """
        # Concatenate embeddings
        combined = torch.cat([source_emb, target_emb], dim=-1)

        # Classify
        logits = self.classifier(combined)

        return logits

    def predict(
        self,
        source_emb: np.ndarray,
        target_emb: np.ndarray,
        source_ref: str,
        target_ref: str,
        threshold: float = 0.5
    ) -> ClassificationResult:
        """
        Predict connection types for a verse pair.

        Args:
            source_emb: Source embedding
            target_emb: Target embedding
            source_ref: Source reference
            target_ref: Target reference
            threshold: Multilabel threshold

        Returns:
            ClassificationResult
        """
        if not TORCH_AVAILABLE:
            return ClassificationResult(
                source_ref=source_ref,
                target_ref=target_ref,
                primary_type="unknown",
                primary_confidence=0.0,
                all_types={},
                multilabel=[]
            )

        self.eval()
        device = next(self.parameters()).device

        # Convert to tensors
        source_t = torch.tensor(source_emb, dtype=torch.float32, device=device).unsqueeze(0)
        target_t = torch.tensor(target_emb, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            logits = self.forward(source_t, target_t)

            if self.multilabel:
                probs = torch.sigmoid(logits)
            else:
                probs = F.softmax(logits, dim=-1)

            probs = probs.squeeze(0).cpu().numpy()

        # Build result
        all_types = {t: float(probs[i]) for i, t in enumerate(self.TYPES)}
        primary_idx = probs.argmax()
        primary_type = self.TYPES[primary_idx]
        primary_confidence = probs[primary_idx]

        # Multilabel predictions
        multilabel = [t for t, p in all_types.items() if p >= threshold]

        return ClassificationResult(
            source_ref=source_ref,
            target_ref=target_ref,
            primary_type=primary_type,
            primary_confidence=float(primary_confidence),
            all_types=all_types,
            multilabel=multilabel
        )
