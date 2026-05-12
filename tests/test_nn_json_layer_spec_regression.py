"""Regression tests for slider path and JSON layer_spec dispatch in NN training."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import torch


from harp.nn_models import PasswordCNN, PasswordCNNConfigurable
from harp.nn_registry import NNModelRegistry
from harp.nn_trainer import PasswordNNTrainer


def _make_tiny_data() -> tuple[pd.Series, pd.Series]:
    """Create tiny deterministic dataset for fast CPU-only tests."""
    passwords = ["pass1", "pass2", "pass3", "pass4", "pass5"] * 12
    labels = [0, 1] * 30
    return pd.Series(passwords), pd.Series(labels)


def _read_latest_metadata(model_dir: str) -> dict:
    meta_files = sorted(Path(model_dir).glob("final/*/metadata.json"))
    assert meta_files, "Expected metadata.json to be created"
    with open(meta_files[-1], encoding="utf-8") as f:
        return json.load(f)


def test_slider_path_no_layer_spec_uses_standard_passwordcnn():
    """No layer_spec should keep legacy slider path and model class metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = PasswordNNTrainer(model_dir=tmpdir, device=torch.device("cpu"))
        X, y = _make_tiny_data()

        result = trainer.train(X, y, epochs=1, batch_size=8, hidden_dim=64, dropout=0.2)

        assert isinstance(result["model"], PasswordCNN)
        metadata = _read_latest_metadata(tmpdir)
        assert metadata["architecture"]["model_class"] == "standard"


def test_layer_spec_path_uses_passwordcnnconfigurable():
    """Dict-based layer_spec should dispatch to configurable model path."""
    layer_spec = [{"units": 64, "activation": "relu"}, {"units": 32}]
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = PasswordNNTrainer(model_dir=tmpdir, device=torch.device("cpu"))
        X, y = _make_tiny_data()

        result = trainer.train(X, y, epochs=1, batch_size=8, layer_spec=layer_spec)

        assert isinstance(result["model"], PasswordCNNConfigurable)
        metadata = _read_latest_metadata(tmpdir)
        architecture = metadata["architecture"]
        assert architecture["model_class"] == "configurable"
        assert "layer_spec" in architecture
        assert architecture["layer_spec"] == layer_spec


def test_invalid_layer_spec_raises_value_error_before_training():
    """Invalid layer_spec values must fail fast with ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = PasswordNNTrainer(model_dir=tmpdir, device=torch.device("cpu"))
        X, y = _make_tiny_data()

        with pytest.raises(ValueError, match="Invalid layer_spec"):
            trainer.train(X, y, epochs=1, batch_size=8, layer_spec=[{"units": -5}])


def test_empty_layer_spec_falls_back_to_standard_path():
    """Empty layer_spec should remain backward-compatible with standard model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = PasswordNNTrainer(model_dir=tmpdir, device=torch.device("cpu"))
        X, y = _make_tiny_data()

        result = trainer.train(X, y, epochs=1, batch_size=8, layer_spec=[])

        assert isinstance(result["model"], PasswordCNN)


def test_load_best_model_backward_compat_without_model_class_key():
    """Legacy metadata without model_class should still load standard PasswordCNN."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = NNModelRegistry(tmpdir)
        run_id = "legacy-regression"
        final_dir = Path(tmpdir) / "final" / run_id
        final_dir.mkdir(parents=True, exist_ok=True)

        model = PasswordCNN(hidden_dim=64, dropout=0.2)
        torch.save(model.state_dict(), final_dir / "best_model.pt")

        legacy_metadata = {
            "run_id": run_id,
            "timestamp": "2026-05-12T00:00:00+00:00",
            "architecture": {
                "embedding_dim": 8,
                "hidden_dim": 64,
                "dropout": 0.2,
                "kernel_sizes": [2, 3, 4],
            },
            "metrics": {"val_loss": 0.5, "val_acc": 0.75},
            "training": {"epochs": 1, "batch_size": 8},
            "pytorch_version": torch.__version__,
        }
        with open(final_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(legacy_metadata, f)

        loaded_model, _ = registry.load_best_model(run_id)

        assert isinstance(loaded_model, PasswordCNN)
        assert not isinstance(loaded_model, PasswordCNNConfigurable)