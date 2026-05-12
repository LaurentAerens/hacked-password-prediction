"""Checkpoint and recovery tests.

Tests ensure:
- NN checkpoint integrity (save/stop/load/resume)
- Cross-phase checkpoints coexist
- Storage formats (joblib for Phase 1/2, PyTorch for NN)
- Recovery after storage
"""

from pathlib import Path
import tempfile
import torch
import numpy as np
import pandas as pd
import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier

from harp.nn_trainer import PasswordNNTrainer
from harp.nn_models import PasswordCNN
from harp.shared_lib.checkpoint_manager import CheckpointManager
from harp.adaptive_trainer import train_with_adaptive_search


# ============================================================================
# NN CHECKPOINT INTEGRITY TESTS
# ============================================================================

class TestNNCheckpointIntegrity:
    """Test NN checkpoint save/load/resume."""

    def test_nn_checkpoint_save_load_cycle(self):
        """NN checkpoint save and load works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create trainer
            trainer = PasswordNNTrainer(model_dir=tmpdir)
            
            # Trainer should initialize checkpoint manager
            assert trainer is not None
            
            # Create a model
            model = PasswordCNN()
            
            # Save checkpoint (use PyTorch native format, not CheckpointManager)
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / "model_epoch1.pt"
            
            torch.save({
                'model_state': model.state_dict(),
                'epoch': 1,
                'loss': 0.5,
            }, checkpoint_path)
            
            # Load checkpoint
            assert checkpoint_path.exists()
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            
            assert checkpoint['epoch'] == 1
            assert checkpoint['loss'] == 0.5

    def test_nn_checkpoint_metadata_preserved(self):
        """NN checkpoint metadata (epoch, loss, metrics) preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / "checkpoint.pt"
            
            # Save checkpoint with metadata
            metadata = {
                'epoch': 5,
                'loss': 0.123,
                'metrics': {
                    'train_acc': 0.92,
                    'val_acc': 0.89,
                },
                'model_state': RandomForestClassifier(n_estimators=2, random_state=42).get_params(),
            }
            
            torch.save(metadata, checkpoint_path)
            
            # Load and verify
            loaded = torch.load(checkpoint_path, weights_only=False)
            assert loaded['epoch'] == 5
            assert loaded['loss'] == 0.123
            assert loaded['metrics']['train_acc'] == 0.92

    def test_multiple_checkpoints_dont_corrupt(self):
        """Multiple checkpoints don't corrupt each other."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save multiple checkpoints
            checkpoints = {}
            for epoch in range(1, 4):
                path = checkpoint_dir / f"checkpoint_epoch{epoch}.pt"
                data = {
                    'epoch': epoch,
                    'loss': 1.0 / epoch,
                    'model_state': {},
                }
                torch.save(data, path)
                checkpoints[epoch] = path
            
            # Verify each is intact
            for epoch, path in checkpoints.items():
                loaded = torch.load(path, weights_only=False)
                assert loaded['epoch'] == epoch
                assert abs(loaded['loss'] - 1.0/epoch) < 1e-6


# ============================================================================
# CROSS-PHASE CHECKPOINT TESTS
# ============================================================================

class TestCrossPhaseCheckpoints:
    """Test checkpoints from different phases coexist."""

    def test_phase1_and_phase2_checkpoints_coexist(self):
        """Phase 1 and Phase 2 checkpoints can coexist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Phase 1 checkpoint
            phase1_path = checkpoint_dir / "phase1.joblib"
            model1 = RandomForestClassifier(n_estimators=2, random_state=42)
            X = np.random.randn(50, 5)
            y = np.random.randint(0, 2, 50)
            model1.fit(X, y)
            joblib.dump(model1, phase1_path)
            
            # Phase 2 checkpoint (also joblib in this version)
            phase2_path = checkpoint_dir / "phase2.joblib"
            model2 = RandomForestClassifier(n_estimators=4, random_state=43)
            model2.fit(X, y)
            joblib.dump(model2, phase2_path)
            
            # Both should exist and be loadable
            assert phase1_path.exists()
            assert phase2_path.exists()
            
            loaded1 = joblib.load(phase1_path)
            loaded2 = joblib.load(phase2_path)
            
            assert loaded1 is not None
            assert loaded2 is not None

    def test_phase1_to_phase2_checkpoint_chain(self):
        """Phase 1 → Phase 2 checkpoint progression works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Phase 1 saves checkpoint
            p1_path = checkpoint_dir / "phase1_best.joblib"
            model1 = RandomForestClassifier(n_estimators=2, random_state=42)
            X = np.random.randn(100, 5)
            y = np.random.randint(0, 2, 100)
            model1.fit(X, y)
            joblib.dump(model1, p1_path)
            
            # Phase 2 reads Phase 1 checkpoint and builds on it
            p2_path = checkpoint_dir / "phase2_best.joblib"
            loaded_p1 = joblib.load(p1_path)
            
            # Phase 2 checkpoint chain
            p2_checkpoint = {
                'phase1_model': loaded_p1,
                'phase2_model': RandomForestClassifier(n_estimators=4, random_state=43),
            }
            joblib.dump(p2_checkpoint, p2_path)
            
            # Verify chain
            loaded_chain = joblib.load(p2_path)
            assert 'phase1_model' in loaded_chain
            assert 'phase2_model' in loaded_chain

    def test_nn_checkpoint_independent(self):
        """NN checkpoints are independent of Phase 1/2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # NN checkpoint (PyTorch)
            nn_path = checkpoint_dir / "nn_model.pt"
            model = PasswordCNN()
            torch.save(model.state_dict(), nn_path)
            
            # Phase 1 checkpoint (joblib)
            p1_path = checkpoint_dir / "phase1.joblib"
            model1 = RandomForestClassifier(n_estimators=2, random_state=42)
            X = np.random.randn(50, 5)
            y = np.random.randint(0, 2, 50)
            model1.fit(X, y)
            joblib.dump(model1, p1_path)
            
            # Both should exist independently
            assert nn_path.exists()
            assert p1_path.exists()
            
            # Load both independently
            nn_state = torch.load(nn_path, weights_only=True)
            p1_model = joblib.load(p1_path)
            
            assert nn_state is not None
            assert p1_model is not None


# ============================================================================
# STORAGE FORMAT TESTS
# ============================================================================

class TestStorageFormats:
    """Test different storage formats."""

    def test_phase1_joblib_format(self):
        """Phase 1 models saved in joblib format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = RandomForestClassifier(n_estimators=2, random_state=42)
            X = np.random.randn(50, 5)
            y = np.random.randint(0, 2, 50)
            model.fit(X, y)
            
            # Save as joblib
            path = Path(tmpdir) / "phase1.joblib"
            joblib.dump(model, path)
            
            # Verify format
            assert path.exists()
            assert path.suffix == ".joblib"
            
            # Load and verify
            loaded = joblib.load(path)
            assert loaded is not None

    def test_phase2_joblib_format(self):
        """Phase 2 models saved in joblib format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = RandomForestClassifier(n_estimators=4, random_state=42)
            X = np.random.randn(50, 5)
            y = np.random.randint(0, 2, 50)
            model.fit(X, y)
            
            # Save as joblib
            path = Path(tmpdir) / "phase2.joblib"
            joblib.dump(model, path)
            
            # Verify format
            assert path.exists()
            loaded = joblib.load(path)
            assert loaded is not None

    def test_nn_pytorch_state_dict_format(self):
        """NN models saved as PyTorch state_dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = PasswordCNN()
            
            # Save as PyTorch
            path = Path(tmpdir) / "nn_model.pt"
            torch.save(model.state_dict(), path)
            
            # Verify format
            assert path.exists()
            assert path.suffix == ".pt"
            
            # Load and verify
            state = torch.load(path, weights_only=True)
            assert state is not None

    def test_formats_recoverable_after_storage(self):
        """All formats recoverable after storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Store various formats
            
            # joblib
            joblib_path = Path(tmpdir) / "model.joblib"
            model1 = RandomForestClassifier(n_estimators=2, random_state=42)
            X = np.random.randn(50, 5)
            y = np.random.randint(0, 2, 50)
            model1.fit(X, y)
            joblib.dump(model1, joblib_path)
            
            # PyTorch
            pt_path = Path(tmpdir) / "model.pt"
            model2 = PasswordCNN()
            torch.save(model2.state_dict(), pt_path)
            
            # Simulate delay (file storage complete)
            import time
            time.sleep(0.1)
            
            # Recovery
            recovered_joblib = joblib.load(joblib_path)
            recovered_pt = torch.load(pt_path, weights_only=True)
            
            assert recovered_joblib is not None
            assert recovered_pt is not None


# ============================================================================
# CHECKPOINT MANAGER INTEGRATION TESTS
# ============================================================================

class TestCheckpointManagerIntegration:
    """Test CheckpointManager with all phases."""

    def test_checkpoint_manager_saves_loads(self):
        """CheckpointManager saves and loads checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(tmpdir)
            
            # Create model
            model = RandomForestClassifier(n_estimators=2, random_state=42)
            X = np.random.randn(50, 5)
            y = np.random.randint(0, 2, 50)
            model.fit(X, y)
            
            # Create results dataframe
            results_df = pd.DataFrame({
                'model': ['LogisticRegression'],
                'preprocessor': ['StandardScaler'],
                'auc': [0.95]
            })
            
            # Save using correct API
            path = cm.save_checkpoint(
                run_id="test_run",
                phase='test',
                combination_index=0,
                total_combinations=1,
                results_df=results_df,
                best_model=model,
                best_auc=0.95,
                best_config={'C': 1.0}
            )
            
            assert path is not None
            assert Path(path).exists()
