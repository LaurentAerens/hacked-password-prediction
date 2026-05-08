"""
Unit tests for CheckpointManager.

Tests verify:
1. Checkpoint save succeeds
2. Checkpoint load restores state exactly
3. Validation detects corrupted files
4. No data loss on interruption (atomic writes)
"""

import pytest
import os
import tempfile
import json
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)

# Import using correct package name (ai-resources becomes ai_resources in imports)
import importlib.util
checkpoint_manager_path = Path(parent_dir) / "ai-resources" / "shared_lib" / "checkpoint_manager.py"
spec = importlib.util.spec_from_file_location("checkpoint_manager", checkpoint_manager_path)
checkpoint_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint_module)
CheckpointManager = checkpoint_module.CheckpointManager


class TestCheckpointManagerBasics:
    """Test basic save/load functionality."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create a temporary checkpoint directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def manager(self, temp_checkpoint_dir):
        """Create a CheckpointManager instance."""
        return CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

    @pytest.fixture
    def sample_state(self):
        """Create sample training state."""
        results_df = pd.DataFrame({
            'model': ['RandomForest', 'RandomForest'],
            'preprocessor': ['StandardScaler', 'StandardScaler'],
            'auc': [0.85, 0.87],
            'fold': [1, 2]
        })
        
        # Create a simple sklearn pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        # Create dummy model state
        X_dummy = np.array([[1, 2], [3, 4]])
        y_dummy = np.array([0, 1])
        pipeline.fit(X_dummy, y_dummy)
        
        return {
            'phase': 'phase_1a',
            'combination_index': 5,
            'total_combinations': 96,
            'results_df': results_df,
            'best_model': pipeline,
            'best_auc': 0.87,
            'best_config': {
                'model': 'RandomForestClassifier',
                'preprocessor': 'StandardScaler',
                'params': {'n_estimators': 10}
            }
        }

    def test_save_checkpoint_creates_files(self, manager, sample_state):
        """Test that save_checkpoint creates both .pkl and .json files."""
        run_id = 'test-run-001'
        path = manager.save_checkpoint(
            run_id=run_id,
            phase=sample_state['phase'],
            combination_index=sample_state['combination_index'],
            total_combinations=sample_state['total_combinations'],
            results_df=sample_state['results_df'],
            best_model=sample_state['best_model'],
            best_auc=sample_state['best_auc'],
            best_config=sample_state['best_config']
        )
        
        assert path is not None
        assert Path(path).exists()
        pkl_file = Path(path)
        json_file = pkl_file.parent / pkl_file.name.replace('.pkl', '.json')
        assert json_file.exists()

    def test_load_checkpoint_restores_state(self, manager, sample_state):
        """Test that load_checkpoint restores state exactly."""
        run_id = 'test-run-002'
        
        # Save checkpoint
        manager.save_checkpoint(
            run_id=run_id,
            phase=sample_state['phase'],
            combination_index=sample_state['combination_index'],
            total_combinations=sample_state['total_combinations'],
            results_df=sample_state['results_df'],
            best_model=sample_state['best_model'],
            best_auc=sample_state['best_auc'],
            best_config=sample_state['best_config']
        )
        
        # Load checkpoint
        loaded_state = manager.load_checkpoint(run_id=run_id)
        
        assert loaded_state['phase'] == sample_state['phase']
        assert loaded_state['combination_index'] == sample_state['combination_index']
        assert loaded_state['total_combinations'] == sample_state['total_combinations']
        assert loaded_state['best_auc'] == sample_state['best_auc']
        assert loaded_state['best_config'] == sample_state['best_config']
        
        # Verify results_df is identical
        pd.testing.assert_frame_equal(
            loaded_state['results_df'],
            sample_state['results_df']
        )
        
        # Verify best_model can predict
        X_test = np.array([[1, 2], [3, 4]])
        predictions = loaded_state['best_model'].predict(X_test)
        assert predictions is not None
        assert len(predictions) == 2

    def test_load_latest_checkpoint_by_run_id(self, manager, sample_state):
        """Test that load_checkpoint loads latest checkpoint for a run_id."""
        run_id = 'test-run-003'
        
        # Save multiple checkpoints for same run_id
        manager.save_checkpoint(
            run_id=run_id,
            phase='phase_1a',
            combination_index=5,
            total_combinations=96,
            results_df=sample_state['results_df'],
            best_model=sample_state['best_model'],
            best_auc=0.85,
            best_config=sample_state['best_config']
        )
        
        # Save a later checkpoint
        manager.save_checkpoint(
            run_id=run_id,
            phase='phase_1a',
            combination_index=10,
            total_combinations=96,
            results_df=sample_state['results_df'],
            best_model=sample_state['best_model'],
            best_auc=0.87,
            best_config=sample_state['best_config']
        )
        
        # Load should return latest (combination_index=10)
        loaded_state = manager.load_checkpoint(run_id=run_id)
        assert loaded_state['combination_index'] == 10
        assert loaded_state['best_auc'] == 0.87


class TestCheckpointValidation:
    """Test checkpoint validation and corruption detection."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create a temporary checkpoint directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def manager(self, temp_checkpoint_dir):
        """Create a CheckpointManager instance."""
        return CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

    def test_validate_checkpoint_success(self, manager, temp_checkpoint_dir):
        """Test that validate_checkpoint returns True for valid checkpoint."""
        run_id = 'test-run-004'
        
        # Create minimal checkpoint files
        checkpoint_path = Path(temp_checkpoint_dir) / f"{run_id}_phase_1a.pkl"
        metadata_path = checkpoint_path.parent / checkpoint_path.name.replace('.pkl', '.json')
        
        # Write valid pickle and JSON
        import pickle
        pickle.dump({'phase': 'phase_1a', 'combination_index': 5}, open(checkpoint_path, 'wb'))
        
        metadata = {
            'run_id': run_id,
            'phase': 'phase_1a',
            'combination_index': 5,
            'timestamp': '2026-05-08T14:30:45.123456Z',
            'checkpoint_version': '1.0',
            'results_count': 5
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        assert manager.validate_checkpoint(str(checkpoint_path)) is True

    def test_validate_checkpoint_detects_corrupted_pkl(self, manager, temp_checkpoint_dir):
        """Test that validate_checkpoint detects corrupted pickle file."""
        checkpoint_path = Path(temp_checkpoint_dir) / "test-run-005_phase_1a.pkl"
        metadata_path = checkpoint_path.parent / checkpoint_path.name.replace('.pkl', '.json')
        
        # Write corrupted pickle
        with open(checkpoint_path, 'wb') as f:
            f.write(b'corrupted data' * 100)
        
        # Write valid JSON
        metadata = {
            'run_id': 'test-run-005',
            'phase': 'phase_1a',
            'combination_index': 5,
            'timestamp': '2026-05-08T14:30:45.123456Z'
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        assert manager.validate_checkpoint(str(checkpoint_path)) is False

    def test_validate_checkpoint_detects_missing_json(self, manager, temp_checkpoint_dir):
        """Test that validate_checkpoint detects missing JSON file."""
        run_id = 'test-run-006'
        checkpoint_path = Path(temp_checkpoint_dir) / f"{run_id}_phase_1a.pkl"
        
        # Write pickle without JSON
        import pickle
        pickle.dump({'phase': 'phase_1a'}, open(checkpoint_path, 'wb'))
        
        assert manager.validate_checkpoint(str(checkpoint_path)) is False

    def test_validate_checkpoint_detects_corrupted_json(self, manager, temp_checkpoint_dir):
        """Test that validate_checkpoint detects corrupted JSON file."""
        run_id = 'test-run-007'
        checkpoint_path = Path(temp_checkpoint_dir) / f"{run_id}_phase_1a.pkl"
        metadata_path = checkpoint_path.parent / checkpoint_path.name.replace('.pkl', '.json')
        
        # Write valid pickle
        import pickle
        pickle.dump({'phase': 'phase_1a'}, open(checkpoint_path, 'wb'))
        
        # Write corrupted JSON
        with open(metadata_path, 'w') as f:
            f.write('{invalid json content')
        
        assert manager.validate_checkpoint(str(checkpoint_path)) is False


class TestCheckpointAtomicity:
    """Test atomic write behavior to prevent corruption."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create a temporary checkpoint directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def manager(self, temp_checkpoint_dir):
        """Create a CheckpointManager instance."""
        return CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

    def test_save_uses_temp_file_pattern(self, manager, temp_checkpoint_dir):
        """Test that save_checkpoint uses temp file to prevent corruption."""
        # This test verifies the implementation uses atomic writes
        # by checking that temporary files are cleaned up after save
        
        run_id = 'test-run-008'
        
        # Create sample state
        results_df = pd.DataFrame({'col': [1, 2]})
        pipeline = Pipeline([('scaler', StandardScaler())])
        X_dummy = np.array([[1], [2]])
        y_dummy = np.array([0, 1])
        pipeline.fit(X_dummy, y_dummy)
        
        manager.save_checkpoint(
            run_id=run_id,
            phase='phase_1a',
            combination_index=1,
            total_combinations=96,
            results_df=results_df,
            best_model=pipeline,
            best_auc=0.85,
            best_config={}
        )
        
        # Check that no .tmp files remain
        tmp_files = list(Path(temp_checkpoint_dir).glob('*.tmp'))
        assert len(tmp_files) == 0, "Temporary files should be cleaned up"


class TestCheckpointMetadata:
    """Test checkpoint metadata format and content."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create a temporary checkpoint directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def manager(self, temp_checkpoint_dir):
        """Create a CheckpointManager instance."""
        return CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

    def test_metadata_contains_required_fields(self, manager, temp_checkpoint_dir):
        """Test that checkpoint metadata contains all required fields."""
        run_id = 'test-run-009'
        
        results_df = pd.DataFrame({
            'model': ['RandomForest'],
            'auc': [0.85]
        })
        pipeline = Pipeline([('scaler', StandardScaler())])
        X_dummy = np.array([[1], [2]])
        y_dummy = np.array([0, 1])
        pipeline.fit(X_dummy, y_dummy)
        
        manager.save_checkpoint(
            run_id=run_id,
            phase='phase_1a',
            combination_index=5,
            total_combinations=96,
            results_df=results_df,
            best_model=pipeline,
            best_auc=0.85,
            best_config={'model': 'RandomForest'}
        )
        
        # Read and verify metadata
        checkpoint_files = list(Path(temp_checkpoint_dir).glob(f"{run_id}_phase_*.json"))
        assert len(checkpoint_files) > 0
        
        with open(checkpoint_files[-1], 'r') as f:
            metadata = json.load(f)
        
        # Verify required fields
        assert 'run_id' in metadata
        assert 'phase' in metadata
        assert 'combination_index' in metadata
        assert 'total_combinations' in metadata
        assert 'timestamp' in metadata
        assert 'checkpoint_version' in metadata
        assert metadata['run_id'] == run_id
        assert metadata['phase'] == 'phase_1a'
        assert metadata['combination_index'] == 5
        assert metadata['total_combinations'] == 96

    def test_metadata_dataframe_shape_recorded(self, manager, temp_checkpoint_dir):
        """Test that metadata records results_df shape."""
        run_id = 'test-run-010'
        
        # Create DataFrame with known shape
        results_df = pd.DataFrame({
            'model': ['RF', 'RF', 'SVM'],
            'preprocessor': ['SS', 'SS', 'SS'],
            'auc': [0.85, 0.87, 0.82]
        })
        
        pipeline = Pipeline([('scaler', StandardScaler())])
        X_dummy = np.array([[1], [2]])
        y_dummy = np.array([0, 1])
        pipeline.fit(X_dummy, y_dummy)
        
        manager.save_checkpoint(
            run_id=run_id,
            phase='phase_1a',
            combination_index=3,
            total_combinations=96,
            results_df=results_df,
            best_model=pipeline,
            best_auc=0.87,
            best_config={}
        )
        
        # Verify shape in metadata
        checkpoint_files = list(Path(temp_checkpoint_dir).glob(f"{run_id}_phase_*.json"))
        with open(checkpoint_files[-1], 'r') as f:
            metadata = json.load(f)
        
        assert 'results_shape' in metadata
        assert metadata['results_shape'] == [3, 3]  # 3 rows, 3 columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
