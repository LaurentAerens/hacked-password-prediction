"""Tests for NNModelRegistry."""

import sys
from pathlib import Path
import torch
import json
import tempfile
import pytest

# Add ai-resources directory to path
ai_resources_path = str(Path(__file__).parent.parent / "ai-resources")
sys.path.insert(0, ai_resources_path)

from nn_registry import NNModelRegistry
from nn_models import PasswordCNN


class TestNNModelRegistry:
    """Unit tests for NNModelRegistry."""

    def test_init_creates_directories(self):
        """Test that init creates required directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = NNModelRegistry(registry_dir=tmpdir)
            
            # Check directories exist
            assert Path(tmpdir).exists()
            assert Path(tmpdir) / "checkpoints" != Path()
            assert Path(tmpdir) / "final" != Path()

    def test_save_checkpoint(self):
        """Test saving a training checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = NNModelRegistry(registry_dir=tmpdir)
            
            model = PasswordCNN()
            optimizer = torch.optim.Adam(model.parameters())
            metrics = {"train_loss": 0.5, "val_loss": 0.6, "val_acc": 0.85}
            
            run_id = "test-run-001"
            epoch = 5
            
            checkpoint_path = registry.save_checkpoint(
                run_id=run_id,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                metrics=metrics
            )
            
            assert checkpoint_path is not None
            assert Path(checkpoint_path).exists()
            assert run_id in checkpoint_path

    def test_save_checkpoint_multiple_epochs(self):
        """Test saving multiple epoch checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = NNModelRegistry(registry_dir=tmpdir)
            
            model = PasswordCNN()
            optimizer = torch.optim.Adam(model.parameters())
            run_id = "test-run-002"
            
            paths = []
            for epoch in range(3):
                metrics = {"train_loss": 0.5 - epoch * 0.1, "val_loss": 0.6 - epoch * 0.1}
                path = registry.save_checkpoint(
                    run_id=run_id,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    metrics=metrics
                )
                paths.append(path)
            
            # All checkpoints should exist
            assert len(paths) == 3
            for path in paths:
                assert Path(path).exists()

    def test_save_best_model(self):
        """Test saving best model with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = NNModelRegistry(registry_dir=tmpdir)
            
            model = PasswordCNN()
            architecture_config = {"embedding_dim": 8, "hidden_dim": 64}
            metrics = {"val_loss": 0.18, "val_acc": 0.92}
            training_config = {"epochs": 20, "batch_size": 32, "learning_rate": 0.001}
            
            run_id = "best-run-001"
            registry.save_best_model(
                run_id=run_id,
                model=model,
                architecture_config=architecture_config,
                metrics=metrics,
                training_config=training_config
            )
            
            # Check files exist
            final_dir = Path(tmpdir) / "final" / run_id
            assert final_dir.exists()
            assert (final_dir / "best_model.pt").exists()
            assert (final_dir / "metadata.json").exists()

    def test_save_best_model_metadata_content(self):
        """Test that saved metadata has expected structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = NNModelRegistry(registry_dir=tmpdir)
            
            model = PasswordCNN()
            architecture_config = {"embedding_dim": 8}
            metrics = {"val_loss": 0.18}
            training_config = {"epochs": 20}
            
            run_id = "test-meta-001"
            registry.save_best_model(
                run_id=run_id,
                model=model,
                architecture_config=architecture_config,
                metrics=metrics,
                training_config=training_config
            )
            
            # Load and verify metadata
            metadata_path = Path(tmpdir) / "final" / run_id / "metadata.json"
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            assert metadata["run_id"] == run_id
            assert "timestamp" in metadata
            assert "architecture" in metadata
            assert "metrics" in metadata
            assert metadata["metrics"]["val_loss"] == 0.18

    def test_load_best_model(self):
        """Test loading best model and metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = NNModelRegistry(registry_dir=tmpdir)
            
            model = PasswordCNN()
            architecture_config = {"embedding_dim": 8, "hidden_dim": 64}
            metrics = {"val_loss": 0.18, "val_acc": 0.92}
            training_config = {"epochs": 20}
            
            run_id = "load-test-001"
            registry.save_best_model(
                run_id=run_id,
                model=model,
                architecture_config=architecture_config,
                metrics=metrics,
                training_config=training_config
            )
            
            # Load model
            loaded_model, loaded_metadata = registry.load_best_model(run_id)
            
            assert loaded_model is not None
            assert loaded_metadata is not None
            assert loaded_metadata["run_id"] == run_id
            assert loaded_model.num_params > 0

    def test_load_best_model_not_found(self):
        """Test loading non-existent model raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = NNModelRegistry(registry_dir=tmpdir)
            
            with pytest.raises((FileNotFoundError, RuntimeError)):
                registry.load_best_model("nonexistent-run")

    def test_save_history(self):
        """Test saving epoch history to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = NNModelRegistry(registry_dir=tmpdir)
            
            history = {
                "epoch": [0, 1, 2],
                "train_loss": [0.5, 0.4, 0.3],
                "val_loss": [0.6, 0.5, 0.4],
                "val_acc": [0.80, 0.85, 0.90]
            }
            
            run_id = "history-test-001"
            registry.save_history(run_id, history)
            
            # Check CSV exists
            csv_path = Path(tmpdir) / "history" / run_id / "metrics.csv"
            assert csv_path.exists()

    def test_save_history_csv_content(self):
        """Test that saved history CSV has correct content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = NNModelRegistry(registry_dir=tmpdir)
            
            history = {
                "epoch": [0, 1, 2],
                "train_loss": [0.5, 0.4, 0.3],
                "val_loss": [0.6, 0.5, 0.4],
                "val_acc": [0.80, 0.85, 0.90]
            }
            
            run_id = "csv-test-001"
            registry.save_history(run_id, history)
            
            # Load CSV and verify
            import pandas as pd
            csv_path = Path(tmpdir) / "history" / run_id / "metrics.csv"
            df = pd.read_csv(csv_path)
            
            assert len(df) == 3
            assert list(df.columns) == ["epoch", "train_loss", "val_loss", "val_acc"]
            assert df["epoch"].tolist() == [0, 1, 2]

    def test_checkpoint_atomicity(self):
        """Test that checkpoint saves are atomic (no temp files left)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = NNModelRegistry(registry_dir=tmpdir)
            
            model = PasswordCNN()
            optimizer = torch.optim.Adam(model.parameters())
            metrics = {"train_loss": 0.5}
            
            checkpoint_path = registry.save_checkpoint(
                run_id="atomic-test",
                epoch=0,
                model=model,
                optimizer=optimizer,
                metrics=metrics
            )
            
            # Check no temp files left
            temp_files = list(Path(tmpdir).rglob("*.tmp"))
            assert len(temp_files) == 0

    def test_load_checkpoint(self):
        """Test loading a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = NNModelRegistry(registry_dir=tmpdir)
            
            model = PasswordCNN()
            optimizer = torch.optim.Adam(model.parameters())
            metrics = {"train_loss": 0.5, "val_loss": 0.6}
            
            run_id = "load-checkpoint-001"
            registry.save_checkpoint(
                run_id=run_id,
                epoch=5,
                model=model,
                optimizer=optimizer,
                metrics=metrics
            )
            
            # Load checkpoint
            loaded_data = registry.load_checkpoint(run_id, epoch=5)
            
            assert loaded_data is not None
            assert "model_state" in loaded_data
            assert "optimizer_state" in loaded_data
            assert "metrics" in loaded_data
            assert loaded_data["metrics"]["train_loss"] == 0.5

    def test_directory_structure_created(self):
        """Test that required directory structure is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = NNModelRegistry(registry_dir=tmpdir)
            
            registry_path = Path(tmpdir)
            assert (registry_path / "checkpoints").exists()
            assert (registry_path / "final").exists()
            assert (registry_path / "history").exists()

    def test_load_best_model_configurable(self):
        """load_best_model reconstructs PasswordCNNConfigurable when model_class=configurable."""
        from nn_models import PasswordCNNConfigurable, build_model_from_spec
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = NNModelRegistry(registry_dir=tmpdir)

            layer_spec = [{"units": 64, "activation": "relu"}, {"units": 32, "activation": "relu"}]
            model = build_model_from_spec(layer_spec, dropout=0.3)
            architecture_config = {
                "model_class": "configurable",
                "layer_spec": layer_spec,
                "dropout": 0.3,
            }
            run_id = "configurable-load-001"
            registry.save_best_model(
                run_id=run_id,
                model=model,
                architecture_config=architecture_config,
                metrics={"val_loss": 0.2},
                training_config={"epochs": 5},
            )

            loaded_model, loaded_metadata = registry.load_best_model(run_id)

            assert isinstance(loaded_model, PasswordCNNConfigurable)
            assert loaded_metadata["run_id"] == run_id

    def test_load_best_model_backward_compat_no_model_class(self):
        """load_best_model falls back to PasswordCNN when model_class key is absent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = NNModelRegistry(registry_dir=tmpdir)

            model = PasswordCNN()
            # Deliberately omit model_class (pre-existing saved model format)
            architecture_config = {"embedding_dim": 8, "hidden_dim": 64, "dropout": 0.2}
            run_id = "compat-load-001"
            registry.save_best_model(
                run_id=run_id,
                model=model,
                architecture_config=architecture_config,
                metrics={"val_loss": 0.15},
                training_config={"epochs": 10},
            )

            loaded_model, loaded_metadata = registry.load_best_model(run_id)

            assert isinstance(loaded_model, PasswordCNN)
            assert loaded_metadata["run_id"] == run_id
