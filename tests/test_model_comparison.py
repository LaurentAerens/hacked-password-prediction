"""Tests for ModelComparison."""

from pathlib import Path
import pytest
import tempfile
import json
import joblib
import pandas as pd
import numpy as np

from harp.model_comparison import ModelComparison
from harp.model_registry import UnifiedModelRegistry


class TestModelComparison:
    """Unit tests for ModelComparison."""

    @pytest.fixture
    def registry_with_models(self):
        """Create a registry with sample models."""
        tmpdir = tempfile.TemporaryDirectory()
        registry = UnifiedModelRegistry(base_dir=tmpdir.name)
        
        # Create Phase 1 model
        from sklearn.ensemble import RandomForestClassifier
        
        p1_path = Path(tmpdir.name) / "phase1" / "final" / "run_001"
        p1_path.mkdir(parents=True, exist_ok=True)
        
        model = RandomForestClassifier(n_estimators=2, random_state=42)
        X = np.random.rand(5, 3)
        y = np.random.randint(0, 2, 5)
        model.fit(X, y)
        
        joblib.dump(model, p1_path / "best_model.joblib")
        
        # Create metadata for Phase 1
        metadata = {
            "metrics": {"val_acc": 0.85, "precision": 0.88, "recall": 0.82, "f1": 0.85},
            "architecture": {"param_count": 1000},
            "training_time_sec": 120
        }
        
        with open(p1_path / "metadata.json", "w") as f:
            json.dump(metadata, f)
        
        yield registry
        tmpdir.cleanup()

    def test_init(self, registry_with_models):
        """Test ModelComparison initialization."""
        comparison = ModelComparison(registry_with_models)
        assert comparison.registry is not None
        assert comparison.comparison_df is None

    def test_build_comparison_table(self, registry_with_models):
        """Test building comparison table."""
        comparison = ModelComparison(registry_with_models)
        comp_df = comparison.build_comparison_table()
        
        assert isinstance(comp_df, pd.DataFrame)
        assert "Phase" in comp_df.columns
        assert "Accuracy" in comp_df.columns
        assert "Precision" in comp_df.columns
        assert len(comp_df) > 0

    def test_get_best_model_by_metric(self, registry_with_models):
        """Test finding best model by metric."""
        comparison = ModelComparison(registry_with_models)
        best_phase = comparison.get_best_model_by_metric("accuracy")
        
        assert best_phase is not None
        assert best_phase.upper() in ["PHASE1", "PHASE2", "NN"]

    def test_plot_comparison(self, registry_with_models):
        """Test generating comparison plot."""
        comparison = ModelComparison(registry_with_models)
        fig = comparison.plot_comparison()
        
        # Should return a plotly figure
        assert fig is not None
        assert hasattr(fig, "add_trace")
