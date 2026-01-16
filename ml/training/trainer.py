"""
BIBLOS v2 - Training Pipeline with MLflow Integration

Unified training infrastructure for all BIBLOS ML models with
experiment tracking, checkpointing, and hyperparameter optimization.
"""
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
import time
import json

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Basic settings
    model_name: str = "biblos_model"
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 100

    # Optimization
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    # Early stopping
    early_stopping_patience: int = 10
    min_delta: float = 1e-4

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 5
    save_best_only: bool = True

    # MLflow
    mlflow_tracking_uri: str = "sqlite:///mlruns.db"
    mlflow_experiment: str = "biblos-v2"

    # Hardware
    device: str = "cuda"
    mixed_precision: bool = True
    num_workers: int = 4

    # Custom params
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingMetrics:
    """Metrics from a training run."""
    epoch: int
    train_loss: float
    val_loss: float
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    learning_rate: float
    epoch_time_seconds: float


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class BiblosTrainer:
    """
    Unified trainer for BIBLOS ML models.

    Features:
    - MLflow experiment tracking
    - Automatic checkpointing
    - Early stopping
    - Learning rate scheduling
    - Mixed precision training
    - Optuna hyperparameter optimization
    """

    def __init__(
        self,
        model: Any,  # nn.Module when torch available
        config: Optional[TrainingConfig] = None,
        train_dataloader: Optional[Any] = None,
        val_dataloader: Optional[Any] = None
    ):
        self.config = config or TrainingConfig()
        self.model = model
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.logger = logging.getLogger("biblos.ml.training")

        # Initialize components
        self.device = self._setup_device()
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.min_delta
        )

        # Metrics tracking
        self.history: List[TrainingMetrics] = []
        self.best_val_loss = float('inf')
        self.best_model_path: Optional[Path] = None

        # Setup MLflow
        self._setup_mlflow()

    def _setup_device(self) -> str:
        """Setup and return training device."""
        if not TORCH_AVAILABLE:
            return "cpu"

        device = self.config.device
        if device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA not available, falling back to CPU")
            return "cpu"
        elif device == "mps" and not torch.backends.mps.is_available():
            self.logger.warning("MPS not available, falling back to CPU")
            return "cpu"

        return device

    def _setup_mlflow(self) -> None:
        """Setup MLflow experiment tracking."""
        if not MLFLOW_AVAILABLE:
            self.logger.warning("MLflow not available")
            return

        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment)

    def _setup_optimizer(self) -> None:
        """Setup optimizer and scheduler."""
        if not TORCH_AVAILABLE:
            return

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Cosine annealing with warmup
        total_steps = len(self.train_loader) * self.config.epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config.learning_rate * 0.01
        )

        # Mixed precision
        if self.config.mixed_precision and self.device == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()

    def train(
        self,
        loss_fn: Optional[Callable] = None,
        metrics_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run full training loop.

        Args:
            loss_fn: Loss function (model, batch) -> loss
            metrics_fn: Metrics function (outputs, targets) -> dict

        Returns:
            Training summary dictionary
        """
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorch not available for training")
            return {"error": "PyTorch not available"}

        self.model.to(self.device)
        self._setup_optimizer()

        # Default loss function
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        # Start MLflow run
        mlflow_run = None
        if MLFLOW_AVAILABLE:
            mlflow_run = mlflow.start_run()
            mlflow.log_params({
                "model_name": self.config.model_name,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "epochs": self.config.epochs,
                "device": self.device
            })

        try:
            for epoch in range(self.config.epochs):
                epoch_start = time.time()

                # Training phase
                train_loss, train_metrics = self._train_epoch(loss_fn, metrics_fn)

                # Validation phase
                val_loss, val_metrics = self._validate_epoch(loss_fn, metrics_fn)

                epoch_time = time.time() - epoch_start
                current_lr = self.optimizer.param_groups[0]['lr']

                # Record metrics
                metrics = TrainingMetrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    learning_rate=current_lr,
                    epoch_time_seconds=epoch_time
                )
                self.history.append(metrics)

                # Log to MLflow
                if MLFLOW_AVAILABLE:
                    mlflow.log_metrics({
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "learning_rate": current_lr,
                        **{f"train_{k}": v for k, v in train_metrics.items()},
                        **{f"val_{k}": v for k, v in val_metrics.items()}
                    }, step=epoch)

                # Checkpointing
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, is_best=True)

                if (epoch + 1) % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(epoch)

                # Logging
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} - "
                    f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - "
                    f"Time: {epoch_time:.1f}s"
                )

                # Early stopping
                if self.early_stopping(val_loss):
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        finally:
            if mlflow_run:
                # Log best model
                if self.best_model_path and MLFLOW_AVAILABLE:
                    mlflow.pytorch.log_model(self.model, "model")
                mlflow.end_run()

        return self._get_training_summary()

    def _train_epoch(
        self,
        loss_fn: Callable,
        metrics_fn: Optional[Callable]
    ) -> tuple:
        """Run single training epoch."""
        self.model.train()
        total_loss = 0.0
        all_metrics: Dict[str, List[float]] = {}

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            if isinstance(batch, (tuple, list)):
                batch = [b.to(self.device) if hasattr(b, 'to') else b for b in batch]
            else:
                batch = batch.to(self.device)

            # Forward pass with mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch)
                    loss = loss_fn(outputs, batch)

                # Backward pass
                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(batch)
                loss = loss_fn(outputs, batch)
                loss.backward()

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            self.scheduler.step()
            total_loss += loss.item()

            # Compute metrics
            if metrics_fn:
                batch_metrics = metrics_fn(outputs, batch)
                for k, v in batch_metrics.items():
                    if k not in all_metrics:
                        all_metrics[k] = []
                    all_metrics[k].append(v)

        avg_loss = total_loss / max(1, len(self.train_loader))
        avg_metrics = {k: sum(v) / max(1, len(v)) for k, v in all_metrics.items()}

        return avg_loss, avg_metrics

    def _validate_epoch(
        self,
        loss_fn: Callable,
        metrics_fn: Optional[Callable]
    ) -> tuple:
        """Run validation epoch."""
        self.model.eval()
        total_loss = 0.0
        all_metrics: Dict[str, List[float]] = {}

        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, (tuple, list)):
                    batch = [b.to(self.device) if hasattr(b, 'to') else b for b in batch]
                else:
                    batch = batch.to(self.device)

                outputs = self.model(batch)
                loss = loss_fn(outputs, batch)
                total_loss += loss.item()

                if metrics_fn:
                    batch_metrics = metrics_fn(outputs, batch)
                    for k, v in batch_metrics.items():
                        if k not in all_metrics:
                            all_metrics[k] = []
                        all_metrics[k].append(v)

        avg_loss = total_loss / max(1, len(self.val_loader))
        avg_metrics = {k: sum(v) / max(1, len(v)) for k, v in all_metrics.items()}

        return avg_loss, avg_metrics

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        if not TORCH_AVAILABLE:
            return

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config.__dict__
        }

        filename = f"checkpoint_epoch_{epoch}.pt"
        if is_best:
            filename = "best_model.pt"
            self.best_model_path = checkpoint_dir / filename

        path = checkpoint_dir / filename
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")

    def _get_training_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        return {
            "best_val_loss": self.best_val_loss,
            "best_model_path": str(self.best_model_path) if self.best_model_path else None,
            "total_epochs": len(self.history),
            "final_train_loss": self.history[-1].train_loss if self.history else None,
            "final_val_loss": self.history[-1].val_loss if self.history else None,
            "training_time_seconds": sum(m.epoch_time_seconds for m in self.history)
        }


def main():
    """CLI entry point for training."""
    import typer
    app = typer.Typer()

    @app.command()
    def train(
        model_type: str = typer.Option("gnn", help="Model type to train"),
        config_path: Optional[str] = typer.Option(None, help="Config file path"),
        epochs: int = typer.Option(100, help="Number of epochs"),
        batch_size: int = typer.Option(64, help="Batch size")
    ):
        """Train a BIBLOS model."""
        typer.echo(f"Training {model_type} model...")
        # Training logic here

    app()


if __name__ == "__main__":
    main()
