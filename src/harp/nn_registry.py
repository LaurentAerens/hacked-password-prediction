"""NNModelRegistry: Manages NN model artifacts (checkpoints, best model, metadata)."""

import torch
import json
import tempfile
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import joblib


class NNModelRegistry:
    """Manages NN model artifacts (checkpoints, best model, metadata)."""

    def __init__(self, registry_dir: Optional[str] = None):
        """
        Initialize registry.
        
        Args:
            registry_dir: Root directory for model artifacts
        """
        if registry_dir is None:
            repo_root = Path(__file__).parent.parent
            registry_dir = str(repo_root / 'models' / 'nn')
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.registry_dir / "checkpoints").mkdir(exist_ok=True)
        (self.registry_dir / "final").mkdir(exist_ok=True)
        (self.registry_dir / "history").mkdir(exist_ok=True)

    def save_checkpoint(
        self,
        run_id: str,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, Any]
    ) -> str:
        """
        Save epoch checkpoint atomically.
        
        Args:
            run_id: Unique run identifier
            epoch: Epoch number
            model: Trained model
            optimizer: Optimizer with current state
            metrics: Metrics dict (train_loss, val_loss, val_acc)
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = self.registry_dir / "checkpoints" / run_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_data = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics
        }
        
        checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pt"
        
        # Atomic save: write to temp, then rename
        with tempfile.NamedTemporaryFile(
            dir=checkpoint_dir,
            delete=False,
            suffix=".tmp"
        ) as tmp_file:
            tmp_path = tmp_file.name
            torch.save(checkpoint_data, tmp_path)
        
        Path(tmp_path).replace(checkpoint_path)
        return str(checkpoint_path)

    def load_checkpoint(self, run_id: str, epoch: int) -> Dict[str, Any]:
        """
        Load checkpoint for specific run and epoch.
        
        Args:
            run_id: Unique run identifier
            epoch: Epoch number
            
        Returns:
            Dictionary with checkpoint data
        """
        checkpoint_path = self.registry_dir / "checkpoints" / run_id / f"epoch_{epoch}.pt"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        return torch.load(checkpoint_path)

    def save_best_model(
        self,
        run_id: str,
        model: torch.nn.Module,
        architecture_config: Dict[str, Any],
        metrics: Dict[str, Any],
        training_config: Dict[str, Any]
    ) -> str:
        """
        Save best model with metadata.
        
        Args:
            run_id: Unique run identifier
            model: Best trained model
            architecture_config: Model architecture configuration
            metrics: Best metrics (val_loss, val_acc, etc.)
            training_config: Training configuration
            
        Returns:
            Path to best_model.pt
        """
        final_dir = self.registry_dir / "final" / run_id
        final_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = final_dir / "best_model.pt"
        torch.save(model.state_dict(), model_path)
        
        # Save metadata
        metadata = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "architecture": architecture_config,
            "metrics": metrics,
            "training": training_config,
            "pytorch_version": torch.__version__
        }
        
        metadata_path = final_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return str(model_path)

    def load_best_model(self, run_id: str) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """
        Load best model and metadata.
        
        Args:
            run_id: Unique run identifier
            
        Returns:
            Tuple of (model, metadata)
        """
        final_dir = self.registry_dir / "final" / run_id
        model_path = final_dir / "best_model.pt"
        metadata_path = final_dir / "metadata.json"
        
        if not model_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Best model not found for run_id={run_id}")
        
        # Load metadata
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Load model state
        model_state = torch.load(model_path, weights_only=True)
        
        # Reconstruct model from architecture config
        from .nn_models import PasswordCNN, build_model_from_spec
        
        arch_config = metadata.get("architecture", {})
        model_class = arch_config.get("model_class", "standard")

        if model_class == "configurable":
            model = build_model_from_spec(
                arch_config["layer_spec"],
                dropout=arch_config.get("dropout", 0.2),
            )
        else:
            model = PasswordCNN(
                embedding_dim=arch_config.get("embedding_dim", 8),
                hidden_dim=arch_config.get("hidden_dim", 64),
                dropout=arch_config.get("dropout", 0.2),
            )
        model.load_state_dict(model_state)
        
        return model, metadata

    def save_history(self, run_id: str, history: Dict[str, Any]) -> str:
        """
        Save epoch history to CSV.
        
        Args:
            run_id: Unique run identifier
            history: History dict with keys like 'epoch', 'train_loss', etc.
            
        Returns:
            Path to metrics.csv
        """
        history_dir = self.registry_dir / "history" / run_id
        history_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame
        df = pd.DataFrame(history)
        
        # Save to CSV
        csv_path = history_dir / "metrics.csv"
        df.to_csv(csv_path, index=False)
        
        return str(csv_path)

    def load_history(self, run_id: str) -> pd.DataFrame:
        """
        Load epoch history from CSV.
        
        Args:
            run_id: Unique run identifier
            
        Returns:
            DataFrame with history
        """
        csv_path = self.registry_dir / "history" / run_id / "metrics.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"History not found for run_id={run_id}")
        
        return pd.read_csv(csv_path)
