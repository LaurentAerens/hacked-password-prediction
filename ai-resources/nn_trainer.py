"""PasswordNNTrainer: Training orchestration with GPU support, checkpointing, and telemetry."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from nn_tokenizer import PasswordTokenizer
from nn_models import PasswordCNN, build_model_from_spec, validate_layer_spec
from nn_registry import NNModelRegistry
from shared_lib.control_signal import ControlSignal
from shared_lib.telemetry_emitter import TelemetryEmitter


class GPUManager:
    """Handles GPU detection, mixed precision, memory limits."""

    @staticmethod
    def detect_device() -> torch.device:
        """
        Detect available device.
        
        Returns:
            torch.device: CUDA if available, else CPU
        """
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_batch_size(device: torch.device, fallback: int = 32) -> int:
        """
        Get appropriate batch size for device.
        
        Args:
            device: torch.device
            fallback: Batch size for CPU (default 32)
            
        Returns:
            Batch size (128 for GPU, fallback for CPU)
        """
        if device.type == "cuda":
            return 128
        else:
            return fallback

    @staticmethod
    def get_mixed_precision_context(device: torch.device):
        """
        Get mixed precision context manager.
        
        Args:
            device: torch.device
            
        Returns:
            Context manager (autocast for GPU, nullcontext for CPU)
        """
        if device.type == "cuda":
            return torch.cuda.amp.autocast()
        else:
            from contextlib import nullcontext
            return nullcontext()


class PasswordNNTrainer:
    """
    Orchestrates NN training with pause/resume/stop, GPU support, checkpointing.
    """

    def __init__(self, model_dir: str = "models/nn", device: Optional[torch.device] = None):
        """
        Initialize trainer.
        
        Args:
            model_dir: Directory for model artifacts
            device: torch.device (auto-detected if None)
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device or GPUManager.detect_device()
        self.tokenizer = PasswordTokenizer()
        self.tokenizer = self.tokenizer.to(self.device)
        self.registry = NNModelRegistry(str(self.model_dir))
        
        self.control_signal: Optional[ControlSignal] = None
        self.telemetry_emitter: Optional[TelemetryEmitter] = None

    def train(
        self,
        X: pd.Series,
        y: pd.Series,
        epochs: int = 20,
        batch_size: Optional[int] = None,
        learning_rate: float = 0.001,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        val_split: float = 0.2,
        layer_spec: list = None,
        control_signal: Optional[ControlSignal] = None,
        telemetry_emitter: Optional[TelemetryEmitter] = None,
    ) -> Dict[str, Any]:
        """
        Train NN with pause/resume/stop support.
        
        Args:
            X: Passwords (pd.Series)
            y: Labels (pd.Series, 0/1)
            epochs: Number of training epochs
            batch_size: Batch size (auto-detected if None)
            learning_rate: Learning rate for optimizer
            hidden_dim: Hidden layer dimension for PasswordCNN
            dropout: Dropout rate for PasswordCNN
            val_split: Validation split fraction
            control_signal: Optional ControlSignal for pause/resume/stop
            telemetry_emitter: Optional TelemetryEmitter for progress events
            
        Returns:
            {
                "model": trained_model,
                "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...]},
                "best_epoch": best_epoch_num,
                "best_metrics": {"val_loss": ..., "val_acc": ...},
                "checkpoint_path": "...",
                "training_time_sec": ...,
                "device": "cuda" or "cpu"
            }
        """
        start_time = datetime.now(timezone.utc)
        run_id = str(uuid.uuid4())[:8]
        
        self.control_signal = control_signal
        self.telemetry_emitter = telemetry_emitter
        
        # Auto-detect batch size if not provided
        if batch_size is None:
            batch_size = GPUManager.get_batch_size(self.device)
        
        # Prepare data
        X_train, X_val, y_train, y_val = self._train_val_split(X, y, val_split)
        
        # Tokenize (detach to avoid issues with embedding layer gradient tracking)
        X_train_encoded = self.tokenizer.batch_encode(X_train.values).detach()
        X_val_encoded = self.tokenizer.batch_encode(X_val.values).detach()
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
        
        # Move to device
        X_train_encoded = X_train_encoded.to(self.device)
        X_val_encoded = X_val_encoded.to(self.device)
        y_train_tensor = y_train_tensor.to(self.device)
        y_val_tensor = y_val_tensor.to(self.device)
        
        # Create dataloaders
        train_dataset = TensorDataset(X_train_encoded, y_train_tensor)
        val_dataset = TensorDataset(X_val_encoded, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model, optimizer, loss
        if layer_spec:
            valid, reason = validate_layer_spec(layer_spec)
            if not valid:
                raise ValueError(f"Invalid layer_spec: {reason}")
            model = build_model_from_spec(layer_spec, dropout=dropout, device=self.device)
        else:
            model = PasswordCNN(hidden_dim=hidden_dim, dropout=dropout).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()
        
        # Emit training started event
        if self.telemetry_emitter:
            self.telemetry_emitter.emit_event(
                event_type="nn.training.started",
                phase="nn_training",
                unit="epoch",
                status="started",
                metrics={
                    "run_id": run_id,
                    "model_size": model.num_params,
                    "batch_size": batch_size,
                    "device": self.device.type,
                    "epochs": epochs,
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                }
            )
        
        # Training loop
        history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        best_epoch = -1
        best_model_state = None
        
        for epoch in range(epochs):
            # Check control signals
            if self.control_signal and self.control_signal.should_stop():
                if self.telemetry_emitter:
                    self.telemetry_emitter.emit_event(
                        event_type="nn.training.stopped",
                        phase="nn_training",
                        unit="epoch",
                        status="stopped",
                        metrics={"run_id": run_id, "epoch": epoch, "reason": "user_stop"}
                    )
                break
            
            if self.control_signal and self.control_signal.should_pause():
                # Wait for resume
                while self.control_signal.get_state() == ControlSignal.PAUSED:
                    import time
                    time.sleep(0.1)
                
                if self.telemetry_emitter:
                    self.telemetry_emitter.emit_event(
                        event_type="nn.training.resumed",
                        phase="nn_training",
                        unit="epoch",
                        status="resumed",
                        metrics={"run_id": run_id, "from_epoch": epoch}
                    )
            
            # Emit epoch started
            if self.telemetry_emitter:
                self.telemetry_emitter.emit_event(
                    event_type="nn.epoch.started",
                    phase="nn_training",
                    unit="epoch",
                    status="started",
                    current=epoch + 1,
                    total=epochs,
                    metrics={"run_id": run_id, "epoch": epoch}
                )
            
            # Train epoch
            train_loss = self._train_epoch(model, train_loader, optimizer, loss_fn)
            
            # Validate epoch
            val_loss, val_acc = self._validate_epoch(model, val_loader, loss_fn)
            
            # Track history
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state = model.state_dict().copy()
            
            # Save checkpoint
            self.registry.save_checkpoint(
                run_id=run_id,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                metrics={"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}
            )
            
            # Emit epoch completed
            if self.telemetry_emitter:
                self.telemetry_emitter.emit_event(
                    event_type="nn.epoch.completed",
                    phase="nn_training",
                    unit="epoch",
                    status="completed",
                    current=epoch + 1,
                    total=epochs,
                    metrics={
                        "run_id": run_id,
                        "epoch": epoch,
                        "train_loss": float(train_loss),
                        "val_loss": float(val_loss),
                        "val_acc": float(val_acc)
                    }
                )
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Save best model
        if layer_spec:
            architecture_config = {
                "embedding_dim": 8,
                "dropout": dropout,
                "kernel_sizes": [2, 3, 4],
                "model_class": "configurable",
                "layer_spec": layer_spec,
            }
        else:
            architecture_config = {
                "embedding_dim": 8,
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "kernel_sizes": [2, 3, 4],
                "model_class": "standard",
            }
        training_config = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "val_split": val_split
        }
        best_metrics = {"val_loss": float(best_val_loss), "val_acc": float(history["val_acc"][best_epoch])}
        
        best_model_path = self.registry.save_best_model(
            run_id=run_id,
            model=model,
            architecture_config=architecture_config,
            metrics=best_metrics,
            training_config=training_config
        )
        
        # Save history
        self.registry.save_history(run_id, history)
        
        # Calculate elapsed time
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        # Emit training completed
        if self.telemetry_emitter:
            self.telemetry_emitter.emit_event(
                event_type="nn.training.completed",
                phase="nn_training",
                unit="epoch",
                status="completed",
                metrics={
                    "run_id": run_id,
                    "final_loss": float(best_val_loss),
                    "final_acc": float(history["val_acc"][best_epoch]),
                    "training_time_sec": elapsed
                }
            )
        
        return {
            "model": model,
            "history": history,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "checkpoint_path": str(best_model_path),
            "training_time_sec": elapsed,
            "device": self.device.type
        }

    def _train_epoch(self, model, train_loader, optimizer, loss_fn) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0

    def _validate_epoch(self, model, val_loader, loss_fn) -> Tuple[float, float]:
        """Validate for one epoch, return (loss, accuracy)."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                logits = model(batch_x)
                loss = loss_fn(logits, batch_y)
                total_loss += loss.item()
                
                # Compute accuracy (sigmoid + threshold at 0.5)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
        
        avg_loss = total_loss / max(1, len(val_loader))
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy

    def _train_val_split(self, X, y, val_split):
        """Split data into train/val sets."""
        n = len(X)
        val_size = int(n * val_split)
        
        indices = np.random.permutation(n)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        X_train = X.iloc[train_indices].reset_index(drop=True)
        y_train = y.iloc[train_indices].reset_index(drop=True)
        X_val = X.iloc[val_indices].reset_index(drop=True)
        y_val = y.iloc[val_indices].reset_index(drop=True)
        
        return X_train, X_val, y_train, y_val
