"""
Integration tests for CheckpointManager - verify acceptance criteria.

Acceptance criteria:
1. Checkpoint save succeeds
2. Checkpoint load restores state exactly
3. Validation detects corrupted files
4. No data loss on interruption
"""

import pytest
import os
import tempfile
import shutil
import time
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import sys
from pathlib import Path as PathlibPath
parent_dir = str(PathlibPath(__file__).parent.parent)
sys.path.insert(0, parent_dir)

import importlib.util
checkpoint_manager_path = PathlibPath(parent_dir) / "ai-resources" / "shared_lib" / "checkpoint_manager.py"
spec = importlib.util.spec_from_file_location("checkpoint_manager", checkpoint_manager_path)
checkpoint_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint_module)
CheckpointManager = checkpoint_module.CheckpointManager


class TestAcceptanceCriteria:
    """Verify all acceptance criteria are met."""

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
    def sample_training_state(self):
        """Create realistic training state."""
        results_df = pd.DataFrame({
            'model': ['RandomForest', 'RandomForest', 'SVM', 'SVM'],
            'preprocessor': ['StandardScaler', 'MinMaxScaler', 'StandardScaler', 'MinMaxScaler'],
            'auc': [0.85, 0.84, 0.82, 0.83],
            'fold': [1, 1, 1, 1],
            'accuracy': [0.88, 0.87, 0.85, 0.86]
        })
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        X_dummy = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_dummy = np.array([0, 1, 0, 1])
        pipeline.fit(X_dummy, y_dummy)
        
        return {
            'results_df': results_df,
            'pipeline': pipeline,
        }

    def test_criterion_1_checkpoint_save_succeeds(self, manager, sample_training_state):
        """Acceptance Criterion 1: Checkpoint save succeeds."""
        run_id = 'ac-test-001'
        
        try:
            path = manager.save_checkpoint(
                run_id=run_id,
                phase='phase_1a',
                combination_index=4,
                total_combinations=96,
                results_df=sample_training_state['results_df'],
                best_model=sample_training_state['pipeline'],
                best_auc=0.85,
                best_config={'model': 'RandomForestClassifier', 'preprocessor': 'StandardScaler'}
            )
            assert path is not None
            assert Path(path).exists()
            # Verify both files exist
            json_path = Path(path).parent / Path(path).name.replace('.pkl', '.json')
            assert json_path.exists()
        except Exception as e:
            pytest.fail(f"Checkpoint save failed: {e}")

    def test_criterion_2_checkpoint_load_restores_state_exactly(self, manager, sample_training_state):
        """Acceptance Criterion 2: Checkpoint load restores state exactly."""
        run_id = 'ac-test-002'
        
        # Save
        manager.save_checkpoint(
            run_id=run_id,
            phase='phase_1a',
            combination_index=4,
            total_combinations=96,
            results_df=sample_training_state['results_df'],
            best_model=sample_training_state['pipeline'],
            best_auc=0.85,
            best_config={'model': 'RandomForestClassifier', 'preprocessor': 'StandardScaler'}
        )
        
        # Load
        loaded = manager.load_checkpoint(run_id=run_id)
        
        # Verify state restoration
        assert loaded['phase'] == 'phase_1a'
        assert loaded['combination_index'] == 4
        assert loaded['total_combinations'] == 96
        assert loaded['best_auc'] == 0.85
        
        # Verify DataFrame
        pd.testing.assert_frame_equal(
            loaded['results_df'],
            sample_training_state['results_df']
        )
        
        # Verify model can predict
        X_test = np.array([[1, 2], [3, 4]])
        pred1 = sample_training_state['pipeline'].predict(X_test)
        pred2 = loaded['best_model'].predict(X_test)
        np.testing.assert_array_equal(pred1, pred2)

    def test_criterion_3_validation_detects_corrupted_files(self, manager, temp_checkpoint_dir):
        """Acceptance Criterion 3: Validation detects corrupted files."""
        checkpoint_path = Path(temp_checkpoint_dir) / "corrupted_phase_1a.pkl"
        metadata_path = checkpoint_path.parent / checkpoint_path.name.replace('.pkl', '.json')
        
        # Write corrupted pickle
        with open(checkpoint_path, 'wb') as f:
            f.write(b'corrupted binary data')
        
        # Write valid JSON
        import json
        with open(metadata_path, 'w') as f:
            json.dump({'run_id': 'test', 'phase': 'phase_1a'}, f)
        
        # Validation should detect corruption
        is_valid = manager.validate_checkpoint(str(checkpoint_path))
        assert is_valid is False

    def test_criterion_4_no_data_loss_on_interruption(self, manager, temp_checkpoint_dir, sample_training_state):
        """Acceptance Criterion 4: No data loss on interruption (atomic writes)."""
        run_id = 'ac-test-004'
        
        # Simulate multiple checkpoint saves (like training progresses)
        for i in range(1, 6):
            manager.save_checkpoint(
                run_id=run_id,
                phase='phase_1a',
                combination_index=i,
                total_combinations=96,
                results_df=sample_training_state['results_df'].head(i),
                best_model=sample_training_state['pipeline'],
                best_auc=0.85 + (i * 0.001),
                best_config={}
            )
        
        # Verify latest checkpoint is intact
        latest = manager.load_checkpoint(run_id=run_id)
        assert latest['combination_index'] == 5
        assert latest['best_auc'] == pytest.approx(0.855, abs=0.001)
        
        # Verify no orphaned temp files exist
        temp_files = list(Path(temp_checkpoint_dir).glob('*.tmp'))
        assert len(temp_files) == 0, "Orphaned temporary files found - atomic write failed"

    def test_multiple_phases_per_run(self, manager, sample_training_state):
        """Test checkpoint across multiple phases (e.g., 1a → 1b)."""
        run_id = 'ac-test-005'
        
        # Phase 1a checkpoint
        manager.save_checkpoint(
            run_id=run_id,
            phase='phase_1a',
            combination_index=10,
            total_combinations=96,
            results_df=sample_training_state['results_df'],
            best_model=sample_training_state['pipeline'],
            best_auc=0.85,
            best_config={}
        )
        
        # Small delay to ensure different timestamps
        time.sleep(0.01)
        
        # Phase 1b checkpoint (later)
        manager.save_checkpoint(
            run_id=run_id,
            phase='phase_1b',
            combination_index=5,
            total_combinations=20,
            results_df=sample_training_state['results_df'],
            best_model=sample_training_state['pipeline'],
            best_auc=0.87,
            best_config={}
        )
        
        # Load latest (should be phase_1b due to later timestamp)
        latest = manager.load_checkpoint(run_id=run_id)
        # Should load the most recent, which is phase_1b with latest timestamp
        assert latest['phase'] == 'phase_1b'
        assert latest['best_auc'] == 0.87

    def test_checkpoint_format_integrity(self, manager, temp_checkpoint_dir, sample_training_state):
        """Test that checkpoint format follows specification."""
        run_id = 'ac-test-006'
        
        manager.save_checkpoint(
            run_id=run_id,
            phase='phase_1a',
            combination_index=7,
            total_combinations=96,
            results_df=sample_training_state['results_df'],
            best_model=sample_training_state['pipeline'],
            best_auc=0.86,
            best_config={'model': 'RandomForest', 'params': {'n_estimators': 10}}
        )
        
        # Verify format
        expected_pkl = Path(temp_checkpoint_dir) / f"{run_id}_phase_phase_1a.pkl"
        expected_json = Path(temp_checkpoint_dir) / f"{run_id}_phase_phase_1a.json"
        
        assert expected_pkl.exists(), "Pickle file format incorrect"
        assert expected_json.exists(), "JSON metadata file format incorrect"
        
        # Verify JSON structure
        import json
        with open(expected_json, 'r') as f:
            metadata = json.load(f)
        
        assert metadata['run_id'] == run_id
        assert metadata['phase'] == 'phase_1a'
        assert metadata['combination_index'] == 7
        assert metadata['checkpoint_version'] == '1.0'
        assert 'timestamp' in metadata


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
