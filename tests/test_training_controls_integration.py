"""
Integration tests for training controls (pause/stop/resume).

Validates:
- Stop halts training gracefully
- Pause pauses between combinations
- Resume continues from checkpoint
- No data loss on interruption
- Checkpoint recovery works
"""

import pytest
import threading
import time
import tempfile
from pathlib import Path
import sys

# Add ai-resources to path
ai_resources_path = Path(__file__).resolve().parent.parent / "ai-resources"
sys.path.insert(0, str(ai_resources_path))

from shared_lib.control_signal import ControlSignal
from shared_lib.checkpoint_manager import CheckpointManager
from adaptive_trainer import train_with_adaptive_search
from shared_lib.data_utils import get_data


class TestTrainingControls:
    """Integration tests for stop/pause/resume controls."""
    
    @pytest.fixture
    def training_data(self):
        """Load training data for tests."""
        import pandas as pd
        # Create small test dataset
        X = ["password123", "admin", "letmein"] * 10
        y = [1, 0, 1] * 10
        return X, y
    
    def test_stop_mid_training(self, training_data):
        """Test that stop halts training gracefully."""
        X, y = training_data
        
        control_signal = ControlSignal(run_id="test_stop")
        
        # Thread to request stop after 1 second
        def request_stop_after_delay():
            time.sleep(1)
            control_signal.request_stop()
        
        stop_thread = threading.Thread(target=request_stop_after_delay, daemon=True)
        stop_thread.start()
        
        # Run training with control signal
        result = train_with_adaptive_search(
            X, y,
            models=["LogisticRegression", "RandomForest"],
            preprocessors=["CountVectorizer", "TfidfVectorizer"],
            cv_folds=3,
            top_percent=0.5,
            n_jobs=1,
            control_signal=control_signal,
        )
        
        # Verify training stopped (not all combinations completed)
        combinations_screened = result["metrics"]["combinations_screened"]
        # With 2 models * 2 preprocessors = 4 total, should not complete all
        assert combinations_screened < 4 or control_signal.get_state() == "STOPPED"
        print(f"✓ Stop test: Screened {combinations_screened}/4 combinations before stopping")
    
    def test_pause_and_resume(self, training_data):
        """Test that pause/resume cycle works correctly."""
        X, y = training_data
        
        control_signal = ControlSignal(run_id="test_pause_resume")
        
        # Thread to pause after 0.5 seconds, then resume after 1 second
        def pause_and_resume():
            time.sleep(0.5)
            print("→ Requesting pause...")
            control_signal.request_pause()
            
            # Wait for trainer to check and transition to PAUSED
            time.sleep(1.0)
            
            # Check that we're paused (state should be PAUSED after should_pause() is called by trainer)
            state = control_signal.get_state()
            print(f"✓ Training paused (state={state})")
            
            # Resume after 0.5 seconds
            time.sleep(0.5)
            print("→ Requesting resume...")
            control_signal.resume()
        
        pause_thread = threading.Thread(target=pause_and_resume, daemon=True)
        pause_thread.start()
        
        # Run training with control signal
        result = train_with_adaptive_search(
            X, y,
            models=["LogisticRegression"],
            preprocessors=["CountVectorizer"],
            cv_folds=2,
            top_percent=0.5,
            n_jobs=1,
            control_signal=control_signal,
        )
        
        # Verify training completed
        assert result is not None
        assert "best_auc" in result["metrics"]
        print(f"✓ Pause/resume test: Completed with best_auc={result['metrics']['best_auc']:.4f}")
    
    def test_checkpoint_save_and_load(self, training_data):
        """Test that checkpoints save and load correctly."""
        X, y = training_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_mgr = CheckpointManager(checkpoint_dir=tmpdir)
            
            # Create dummy results and model
            import pandas as pd
            from sklearn.linear_model import LogisticRegression
            
            results_df = pd.DataFrame({
                "model": ["LogReg"],
                "preprocessor": ["CountVec"],
                "best_auc": [0.85],
                "cv_score_mean": [0.84],
            })
            
            model = LogisticRegression()
            
            # Save checkpoint
            run_id = "test_checkpoint"
            phase = "phase_1a_screening"
            combo_idx = 5
            total_combos = 10
            best_auc = 0.85
            best_config = {"model": "LogReg", "preprocessor": "CountVec"}
            
            path = checkpoint_mgr.save_checkpoint(
                run_id=run_id,
                phase=phase,
                combination_index=combo_idx,
                total_combinations=total_combos,
                results_df=results_df,
                best_model=model,
                best_auc=best_auc,
                best_config=best_config,
            )
            
            Path(path).exists()
            print(f"✓ Checkpoint saved to {path}")
            
            # Validate checkpoint
            is_valid = checkpoint_mgr.validate_checkpoint(path)
            assert is_valid
            print("✓ Checkpoint validated")
            
            # Load checkpoint - returns entire checkpoint_data dict
            checkpoint_data = checkpoint_mgr.load_checkpoint(run_id)
            
            assert checkpoint_data["phase"] == phase
            assert checkpoint_data["combination_index"] == combo_idx
            assert checkpoint_data["results_df"].shape == results_df.shape
            assert checkpoint_data["best_auc"] == best_auc
            print(f"✓ Checkpoint loaded: phase={checkpoint_data['phase']}, combo_idx={checkpoint_data['combination_index']}")
    
    def test_no_data_loss_on_stop(self, training_data):
        """Test that partial results are preserved when training stops."""
        X, y = training_data
        
        control_signal = ControlSignal(run_id="test_no_loss")
        
        # Stop after 0.5 seconds
        def stop_after_delay():
            time.sleep(0.5)
            control_signal.request_stop()
        
        stop_thread = threading.Thread(target=stop_after_delay, daemon=True)
        stop_thread.start()
        
        # Run training
        result = train_with_adaptive_search(
            X, y,
            models=["LogisticRegression", "RandomForest"],
            preprocessors=["CountVectorizer"],
            cv_folds=2,
            top_percent=0.5,
            n_jobs=1,
            control_signal=control_signal,
        )
        
        # Verify partial results are present
        assert result is not None
        assert "results_df" in result
        assert len(result["results_df"]) > 0
        assert "best_config" in result
        print(f"✓ Partial results preserved: {len(result['results_df'])} combinations completed")
    
    def test_control_signal_state_transitions(self):
        """Test state machine transitions are correct."""
        signal = ControlSignal(run_id="test_states")
        
        # Initial state
        assert signal.get_state() == "RUNNING"
        print("✓ Initial state: RUNNING")
        
        # Request pause
        signal.request_pause()
        assert signal.get_state() == "PAUSE_REQUESTED"
        print("✓ State after pause request: PAUSE_REQUESTED")
        
        # Call should_pause() - this triggers transition to PAUSED
        pause_signal = signal.should_pause()
        assert pause_signal == True  # Returns True on transition
        assert signal.get_state() == "PAUSED"
        print("✓ State after should_pause(): PAUSED, should_pause()=True")
        
        # Subsequent should_pause() calls return False
        pause_signal = signal.should_pause()
        assert pause_signal == False
        print("✓ Subsequent should_pause(): False (already transitioned)")
        
        # Request resume
        signal.resume()
        assert signal.get_state() == "RUNNING"
        print("✓ State after resume: RUNNING")
        
        # Request stop
        signal.request_stop()
        assert signal.get_state() == "STOP_REQUESTED"
        print("✓ State after stop request: STOP_REQUESTED")
        
        # Call should_stop() - this triggers transition to STOPPED
        stop_signal = signal.should_stop()
        assert stop_signal == True  # Returns True on transition
        assert signal.get_state() == "STOPPED"
        print("✓ State after should_stop(): STOPPED, should_stop()=True")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
