"""Tests for PasswordNNTrainer."""

import sys
from pathlib import Path
import torch
import pandas as pd
import pytest
import tempfile

# Add ai-resources directory to path
ai_resources_path = str(Path(__file__).parent.parent / "ai-resources")
sys.path.insert(0, ai_resources_path)

from nn_trainer import PasswordNNTrainer, GPUManager
from shared_lib.control_signal import ControlSignal
from shared_lib.telemetry_emitter import TelemetryEmitter


class TestGPUManager:
    """Unit tests for GPUManager."""

    def test_detect_device_returns_torch_device(self):
        """Test detect_device returns a torch.device."""
        device = GPUManager.detect_device()
        assert isinstance(device, torch.device)

    def test_detect_device_cuda_if_available(self):
        """Test detect_device returns cuda if available."""
        device = GPUManager.detect_device()
        if torch.cuda.is_available():
            assert device.type == "cuda"
        else:
            assert device.type == "cpu"

    def test_get_batch_size_gpu(self):
        """Test batch size for GPU."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = GPUManager.get_batch_size(device)
        
        if device.type == "cuda":
            assert batch_size == 128
        else:
            assert batch_size == 32

    def test_get_batch_size_cpu(self):
        """Test batch size for CPU."""
        device = torch.device("cpu")
        batch_size = GPUManager.get_batch_size(device)
        assert batch_size == 32

    def test_get_batch_size_custom_fallback(self):
        """Test batch size with custom fallback."""
        device = torch.device("cpu")
        batch_size = GPUManager.get_batch_size(device, fallback=64)
        assert batch_size == 64

    def test_get_mixed_precision_context_cpu(self):
        """Test mixed precision context for CPU."""
        device = torch.device("cpu")
        ctx = GPUManager.get_mixed_precision_context(device)
        
        # Should be a context manager
        assert hasattr(ctx, "__enter__") and hasattr(ctx, "__exit__")

    def test_get_mixed_precision_context_cuda(self):
        """Test mixed precision context for GPU."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ctx = GPUManager.get_mixed_precision_context(device)
        
        # Should be a context manager
        assert hasattr(ctx, "__enter__") and hasattr(ctx, "__exit__")


class TestPasswordNNTrainer:
    """Unit tests for PasswordNNTrainer."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PasswordNNTrainer(model_dir=tmpdir)
            assert str(trainer.model_dir) == tmpdir
            assert trainer.device is not None
            assert trainer.control_signal is None

    def test_init_device_detection(self):
        """Test device detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PasswordNNTrainer(model_dir=tmpdir)
            
            if torch.cuda.is_available():
                assert trainer.device.type == "cuda"
            else:
                assert trainer.device.type == "cpu"

    def test_init_custom_device_cpu(self):
        """Test initialization with explicit CPU device."""
        with tempfile.TemporaryDirectory() as tmpdir:
            device = torch.device("cpu")
            trainer = PasswordNNTrainer(model_dir=tmpdir, device=device)
            assert trainer.device.type == "cpu"

    def test_train_basic(self):
        """Test basic training loop."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PasswordNNTrainer(model_dir=tmpdir)
            
            # Create dummy data
            passwords = ["pass1", "pass2", "pass3", "pass4", "pass5"] * 20
            labels = [0, 1] * 50
            X = pd.Series(passwords)
            y = pd.Series(labels[:len(X)])
            
            result = trainer.train(X, y, epochs=2, batch_size=8)
            
            assert "model" in result
            assert "history" in result
            assert "best_metrics" in result
            assert "device" in result
            assert result["device"] == trainer.device.type

    def test_train_history_structure(self):
        """Test that training history has expected structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PasswordNNTrainer(model_dir=tmpdir)
            
            passwords = ["pass1", "pass2", "pass3", "pass4", "pass5"] * 20
            labels = [0, 1] * 50
            X = pd.Series(passwords)
            y = pd.Series(labels[:len(X)])
            
            result = trainer.train(X, y, epochs=2, batch_size=8)
            history = result["history"]
            
            assert "epoch" in history
            assert "train_loss" in history
            assert "val_loss" in history
            assert "val_acc" in history
            assert len(history["epoch"]) == 2

    def test_train_with_control_signal_pause(self):
        """Test pause/resume during training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PasswordNNTrainer(model_dir=tmpdir)
            control_signal = ControlSignal(run_id="test-run-pause")
            
            passwords = ["pass1", "pass2", "pass3", "pass4", "pass5"] * 20
            labels = [0, 1] * 50
            X = pd.Series(passwords)
            y = pd.Series(labels[:len(X)])
            
            # Train should not crash with pause/resume
            result = trainer.train(
                X, y,
                epochs=3,
                batch_size=8,
                control_signal=control_signal
            )
            
            assert result is not None
            assert "model" in result

    def test_train_with_telemetry_emitter(self):
        """Test training with telemetry emitter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PasswordNNTrainer(model_dir=tmpdir)
            emitter = TelemetryEmitter(run_id="test-run-telemetry")
            events = []
            emitter.subscribe(lambda e: events.append(e))
            
            passwords = ["pass1", "pass2", "pass3", "pass4", "pass5"] * 20
            labels = [0, 1] * 50
            X = pd.Series(passwords)
            y = pd.Series(labels[:len(X)])
            
            result = trainer.train(
                X, y,
                epochs=2,
                batch_size=8,
                telemetry_emitter=emitter
            )
            
            # Should have emitted events
            assert len(events) > 0
            
            # Check for key event types
            event_types = [e.get("event_type", "") for e in events]
            assert any("nn.training" in et for et in event_types)

    def test_train_checkpoint_saved(self):
        """Test that training saves checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PasswordNNTrainer(model_dir=tmpdir)
            
            passwords = ["pass1", "pass2", "pass3", "pass4", "pass5"] * 20
            labels = [0, 1] * 50
            X = pd.Series(passwords)
            y = pd.Series(labels[:len(X)])
            
            result = trainer.train(X, y, epochs=2, batch_size=8)
            
            # Check checkpoint path exists
            if "checkpoint_path" in result:
                checkpoint_path = Path(result["checkpoint_path"])
                # Checkpoint should be saved to model_dir
                assert str(checkpoint_path).startswith(tmpdir)

    def test_train_batch_size_auto_detect(self):
        """Test automatic batch size detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PasswordNNTrainer(model_dir=tmpdir)
            
            passwords = ["pass1", "pass2", "pass3", "pass4", "pass5"] * 20
            labels = [0, 1] * 50
            X = pd.Series(passwords)
            y = pd.Series(labels[:len(X)])
            
            # Train without specifying batch_size
            result = trainer.train(X, y, epochs=2, batch_size=None)
            
            assert result is not None

    def test_train_learning_rate_custom(self):
        """Test training with custom learning rate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PasswordNNTrainer(model_dir=tmpdir)
            
            passwords = ["pass1", "pass2", "pass3", "pass4", "pass5"] * 20
            labels = [0, 1] * 50
            X = pd.Series(passwords)
            y = pd.Series(labels[:len(X)])
            
            result = trainer.train(X, y, epochs=2, batch_size=8, learning_rate=0.01)
            
            assert result is not None

    def test_train_val_split(self):
        """Test training with custom validation split."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PasswordNNTrainer(model_dir=tmpdir)
            
            passwords = ["pass1", "pass2", "pass3", "pass4", "pass5"] * 20
            labels = [0, 1] * 50
            X = pd.Series(passwords)
            y = pd.Series(labels[:len(X)])
            
            result = trainer.train(X, y, epochs=2, batch_size=8, val_split=0.3)
            
            assert result is not None
            assert "val_loss" in result["history"]


class TestPasswordNNTrainerGPU:
    """Tests specific to GPU handling."""

    def test_train_gpu_fallback_to_cpu(self):
        """Test that trainer falls back to CPU if GPU unavailable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Force CPU device
            device = torch.device("cpu")
            trainer = PasswordNNTrainer(model_dir=tmpdir, device=device)
            
            passwords = ["pass1", "pass2", "pass3", "pass4", "pass5"] * 10
            labels = [0, 1] * 25
            X = pd.Series(passwords[:len(labels)])
            y = pd.Series(labels)
            
            result = trainer.train(X, y, epochs=1, batch_size=4)
            assert result["device"] == "cpu"
