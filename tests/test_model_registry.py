"""Tests for UnifiedModelRegistry."""

from pathlib import Path
import pytest
import tempfile
import json
import joblib
import torch

from harp.model_registry import UnifiedModelRegistry
from harp.nn_models import PasswordCNN


class TestUnifiedModelRegistry:
    """Unit tests for UnifiedModelRegistry."""

    def test_init_defaults(self):
        """Test registry initialization."""
        registry = UnifiedModelRegistry()
        assert registry.base_dir == "models"
        assert "phase1" in registry.registry
        assert "phase2" in registry.registry
        assert "nn" in registry.registry

    def test_scan_models_empty(self):
        """Test scanning empty directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = UnifiedModelRegistry(base_dir=tmpdir)
            models = registry.scan_models()
            
            assert isinstance(models, dict)
            assert "phase1" in models or len(models) == 0

    def test_scan_models_phase1(self):
        """Test scanning Phase 1 models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = UnifiedModelRegistry(base_dir=tmpdir)
            
            # Create Phase 1 model structure
            p1_path = Path(tmpdir) / "phase1" / "final" / "run_001"
            p1_path.mkdir(parents=True, exist_ok=True)
            
            # Save dummy model
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=2, random_state=42)
            import numpy as np
            X = np.random.rand(5, 3)
            y = np.random.randint(0, 2, 5)
            model.fit(X, y)
            
            model_file = p1_path / "best_model.joblib"
            joblib.dump(model, model_file)
            
            models = registry.scan_models()
            
            assert "phase1" in models
            assert len(models["phase1"]) > 0
            assert models["phase1"][0]["run_id"] == "run_001"

    def test_get_latest_models(self):
        """Test getting latest model from each phase."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = UnifiedModelRegistry(base_dir=tmpdir)
            
            # Create Phase 1 models with timestamps
            from sklearn.ensemble import RandomForestClassifier
            import numpy as np
            
            for run_id in ["run_001", "run_002"]:
                p1_path = Path(tmpdir) / "phase1" / "final" / run_id
                p1_path.mkdir(parents=True, exist_ok=True)
                
                model = RandomForestClassifier(n_estimators=2, random_state=42)
                X = np.random.rand(5, 3)
                y = np.random.randint(0, 2, 5)
                model.fit(X, y)
                
                model_file = p1_path / "best_model.joblib"
                joblib.dump(model, model_file)
            
            latest = registry.get_latest_models()
            
            # Should get the latest run_id
            assert "phase1" in latest
            assert latest["phase1"]["run_id"] == "run_002"

    def test_get_model_metadata(self):
        """Test loading model metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = UnifiedModelRegistry(base_dir=tmpdir)
            
            # Create metadata file
            p1_path = Path(tmpdir) / "phase1" / "final" / "run_001"
            p1_path.mkdir(parents=True, exist_ok=True)
            
            metadata = {
                "metrics": {"val_acc": 0.85, "precision": 0.88, "recall": 0.82, "f1": 0.85},
                "architecture": {"param_count": 1000},
                "training_time_sec": 120
            }
            
            metadata_file = p1_path / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f)
            
            loaded = registry.get_model_metadata("phase1", "run_001")
            
            assert loaded["metrics"]["val_acc"] == 0.85
            assert loaded["training_time_sec"] == 120

    def test_get_model_metadata_missing(self):
        """Test handling missing metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = UnifiedModelRegistry(base_dir=tmpdir)
            
            # Try to load non-existent metadata
            loaded = registry.get_model_metadata("phase1", "nonexistent")
            
            assert loaded == {}
