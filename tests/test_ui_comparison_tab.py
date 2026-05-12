"""Integration tests for UI Comparison Tab."""

from pathlib import Path
import pytest
import tempfile
import json
import joblib
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from harp.model_comparison import ModelComparison
from harp.model_registry import UnifiedModelRegistry
from harp.ensemble import EnsemblePredictor


class TestUIComparisonTab:
    """Integration tests for comparison tab UI components."""

    @pytest.fixture
    def registry_with_all_models(self):
        """Create a registry with models from all phases."""
        tmpdir = tempfile.TemporaryDirectory()
        registry = UnifiedModelRegistry(base_dir=tmpdir.name)
        
        from sklearn.ensemble import RandomForestClassifier
        
        # Create Phase 1 model
        p1_path = Path(tmpdir.name) / "phase1" / "final" / "run_001"
        p1_path.mkdir(parents=True, exist_ok=True)
        
        model = RandomForestClassifier(n_estimators=2, random_state=42)
        X = np.random.rand(5, 3)
        y = np.random.randint(0, 2, 5)
        model.fit(X, y)
        
        joblib.dump(model, p1_path / "best_model.joblib")
        
        metadata = {
            "metrics": {"val_acc": 0.85, "precision": 0.88, "recall": 0.82, "f1": 0.85},
            "architecture": {"param_count": 1000},
            "training_time_sec": 120
        }
        
        with open(p1_path / "metadata.json", "w") as f:
            json.dump(metadata, f)
        
        # Create Phase 2 model
        p2_path = Path(tmpdir.name) / "phase2" / "final" / "run_001"
        p2_path.mkdir(parents=True, exist_ok=True)
        
        model2 = RandomForestClassifier(n_estimators=3, random_state=42)
        model2.fit(X, y)
        
        joblib.dump(model2, p2_path / "best_model.joblib")
        
        metadata2 = {
            "metrics": {"val_acc": 0.87, "precision": 0.89, "recall": 0.84, "f1": 0.86},
            "architecture": {"param_count": 1200},
            "training_time_sec": 150
        }
        
        with open(p2_path / "metadata.json", "w") as f:
            json.dump(metadata2, f)
        
        yield registry
        tmpdir.cleanup()

    def test_comparison_table_renders(self, registry_with_all_models):
        """Test that comparison table renders with all phases."""
        comparison = ModelComparison(registry_with_all_models)
        comp_df = comparison.build_comparison_table()
        
        assert len(comp_df) >= 1
        assert all(col in comp_df.columns for col in ["Phase", "Accuracy", "Precision", "Recall", "F1"])

    def test_best_model_identification(self, registry_with_all_models):
        """Test best model is correctly identified."""
        comparison = ModelComparison(registry_with_all_models)
        best_acc = comparison.get_best_model_by_metric("accuracy")
        
        # Should identify a phase
        assert best_acc is not None

    def test_ensemble_predictions_with_models(self, registry_with_all_models):
        """Test ensemble makes predictions."""
        from sklearn.ensemble import RandomForestClassifier
        
        ensemble = EnsemblePredictor()
        
        latest = registry_with_all_models.get_latest_models()
        
        if "phase1" in latest:
            ensemble.load_phase1_model(latest["phase1"]["path"])
        
        if "phase2" in latest:
            ensemble.load_phase2_model(latest["phase2"]["path"])
        
        # Test that ensemble has models loaded
        assert ensemble.get_active_model_count() > 0

    def test_ensemble_config_display(self):
        """Test ensemble configuration for UI display."""
        ensemble = EnsemblePredictor()
        config = ensemble.ensemble_config
        
        assert "phase1" in config
        assert "phase2" in config
        assert "nn" in config
        assert "method" in config
        assert "active_count" in config

    def test_model_loading_graceful_failure(self):
        """Test graceful handling of missing model paths."""
        ensemble = EnsemblePredictor()
        
        # Try to load from non-existent path
        try:
            ensemble.load_phase1_model("/nonexistent/path/model.joblib")
        except FileNotFoundError:
            pass
        
        # Should not crash; should remain disabled
        assert ensemble.enabled_models["phase1"] is False
