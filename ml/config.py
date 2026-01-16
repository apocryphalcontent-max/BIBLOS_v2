"""
BIBLOS v2 - ML Configuration

Centralized configuration for all ML components with Pydantic validation
and environment variable support.
"""
from typing import Dict, List, Optional, Any
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class EmbeddingModelConfig(BaseModel):
    """Configuration for an embedding model."""
    name: str
    model_id: str
    dimensions: int
    languages: List[str]
    base_weight: float = 0.25
    custom_path: Optional[str] = None
    batch_size: int = 32
    max_seq_length: int = 512


class EmbeddingConfig(BaseModel):
    """Configuration for the embedding ensemble."""
    models: Dict[str, EmbeddingModelConfig] = Field(default_factory=lambda: {
        "mpnet": EmbeddingModelConfig(
            name="all-mpnet-base-v2",
            model_id="sentence-transformers/all-mpnet-base-v2",
            dimensions=768,
            languages=["en", "mixed"],
            base_weight=0.35
        ),
        "minilm": EmbeddingModelConfig(
            name="all-MiniLM-L6-v2",
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            dimensions=384,
            languages=["en"],
            base_weight=0.15
        ),
        "greek": EmbeddingModelConfig(
            name="custom-ancient-greek",
            model_id="custom-ancient-greek",
            dimensions=768,
            languages=["grc", "el"],
            base_weight=0.25,
            custom_path="models/greek_embeddings"
        ),
        "hebrew": EmbeddingModelConfig(
            name="custom-biblical-hebrew",
            model_id="custom-biblical-hebrew",
            dimensions=768,
            languages=["he", "hbo"],
            base_weight=0.25,
            custom_path="models/hebrew_embeddings"
        )
    })
    cache_dir: str = "data/embeddings_cache"
    max_cache_items: int = 100000
    enable_disk_cache: bool = True
    normalize_embeddings: bool = True


class GNNConfig(BaseModel):
    """Configuration for the Graph Neural Network."""
    hidden_channels: int = 256
    num_layers: int = 3
    dropout: float = 0.2
    heads: int = 8  # For GAT layers
    aggregation: str = "mean"  # mean, max, sum
    edge_types: List[str] = Field(default_factory=lambda: [
        "thematic", "verbal", "conceptual", "historical",
        "typological", "prophetic", "liturgical", "narrative",
        "genealogical", "geographical"
    ])


class TrainingConfig(BaseModel):
    """Configuration for model training."""
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 100
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 10
    checkpoint_interval: int = 5
    eval_interval: int = 1


class MLflowConfig(BaseModel):
    """Configuration for MLflow experiment tracking."""
    tracking_uri: str = "sqlite:///mlruns.db"
    experiment_name: str = "biblos-v2"
    artifact_location: str = "mlruns/artifacts"
    registry_uri: str = "sqlite:///mlruns.db"


class InferenceConfig(BaseModel):
    """Configuration for inference serving."""
    model_path: str = "models/production"
    batch_size: int = 32
    max_workers: int = 4
    timeout_seconds: int = 30
    cache_predictions: bool = True
    cache_ttl_seconds: int = 3600


class MLConfig(BaseSettings):
    """Master ML configuration with environment variable support."""

    # Component configs
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    gnn: GNNConfig = Field(default_factory=GNNConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)

    # General settings
    device: str = "cuda"  # cuda, cpu, mps
    seed: int = 42
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    logs_dir: Path = Path("logs")

    # Feature flags
    enable_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = False
    enable_distributed: bool = False

    model_config = {
        "env_prefix": "BIBLOS_ML_",
        "env_nested_delimiter": "__"
    }

    def get_device(self) -> str:
        """Get the appropriate device, with fallback."""
        import torch
        if self.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif self.device == "mps" and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


# Global config instance
config = MLConfig()
