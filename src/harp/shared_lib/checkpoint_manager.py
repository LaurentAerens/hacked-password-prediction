"""
Checkpoint Manager for serializing partial training state to disk.

Supports pause/resume workflows by saving model state, results, and metadata
after each combination completes. Uses atomic writes (temp file → rename) to
prevent corruption on interruption.

Format:
  {run_id}_phase_{phase}.pkl    # joblib-serialized model state
  {run_id}_phase_{phase}.json   # metadata (run_id, phase, timestamp, etc.)
"""

import os
import json
import pickle
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import joblib


class CheckpointManager:
    """Manages checkpoint save/load with atomic writes and validation."""

    def __init__(self, checkpoint_dir: str):
        """
        Initialize CheckpointManager.
        
        Args:
            checkpoint_dir: Directory path for storing checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.version = "1.0"

    def save_checkpoint(
        self,
        run_id: str,
        phase: str,
        combination_index: int,
        total_combinations: int,
        results_df: pd.DataFrame,
        best_model: Any,
        best_auc: float,
        best_config: Dict[str, Any],
        error: Optional[str] = None
    ) -> str:
        """
        Save checkpoint atomically to disk.
        
        Uses temp file + rename pattern to ensure no corruption on interruption.
        
        Args:
            run_id: Unique run identifier
            phase: Phase name (e.g., 'phase_1a', 'phase_1b')
            combination_index: Current combination index
            total_combinations: Total number of combinations
            results_df: DataFrame with results so far
            best_model: Best model object (sklearn Pipeline or similar)
            best_auc: Best AUC score so far
            best_config: Best configuration dict
            error: Optional error message if checkpoint saved due to error
            
        Returns:
            Path to saved checkpoint (pkl file)
        """
        checkpoint_name = f"{run_id}_phase_{phase}"
        pkl_path = self.checkpoint_dir / f"{checkpoint_name}.pkl"
        json_path = self.checkpoint_dir / f"{checkpoint_name}.json"
        
        # Prepare checkpoint data
        checkpoint_data = {
            "phase": phase,
            "combination_index": combination_index,
            "total_combinations": total_combinations,
            "results_df": results_df,
            "best_model": best_model,
            "best_auc": best_auc,
            "best_config": best_config,
        }
        
        # Prepare metadata
        metadata = {
            "run_id": run_id,
            "phase": phase,
            "combination_index": combination_index,
            "total_combinations": total_combinations,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checkpoint_version": self.version,
            "best_auc": best_auc,
            "results_count": len(results_df) if results_df is not None else 0,
            "results_shape": results_df.shape if results_df is not None else None,
            "best_model_type": type(best_model).__module__ + "." + type(best_model).__name__,
            "error": error,
        }
        
        # Atomic write: save to temp file, then rename
        try:
            # Save pickle to temp file
            with tempfile.NamedTemporaryFile(
                dir=self.checkpoint_dir,
                delete=False,
                suffix=".tmp"
            ) as tmp_file:
                tmp_pkl_path = tmp_file.name
                joblib.dump(checkpoint_data, tmp_pkl_path)
            
            # Rename temp to final
            Path(tmp_pkl_path).replace(pkl_path)
            
            # Save metadata (non-critical, doesn't need atomic write)
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            return str(pkl_path)
        
        except Exception as e:
            # Clean up temp file if it exists
            if os.path.exists(tmp_pkl_path):
                os.remove(tmp_pkl_path)
            raise RuntimeError(
                f"Failed to save checkpoint for run_id={run_id}, phase={phase}: {e}"
            )

    def load_checkpoint(self, run_id: str) -> Dict[str, Any]:
        """
        Load latest checkpoint for a given run_id.
        
        Finds the most recent checkpoint file for the run_id by timestamp,
        then by combination_index as tiebreaker.
        
        Args:
            run_id: Unique run identifier
            
        Returns:
            Dictionary with checkpoint data (phase, combination_index, results_df, best_model, etc.)
            
        Raises:
            FileNotFoundError: If no checkpoint found for run_id
            RuntimeError: If checkpoint is corrupted or cannot be loaded
        """
        # Find all checkpoint files for this run_id
        pattern = f"{run_id}_phase_*.pkl"
        checkpoint_files = list(self.checkpoint_dir.glob(pattern))
        
        if not checkpoint_files:
            raise FileNotFoundError(
                f"No checkpoint found for run_id={run_id} in {self.checkpoint_dir}"
            )
        
        # Find latest checkpoint by timestamp, then combination_index as tiebreaker
        latest_checkpoint = None
        latest_timestamp = None
        latest_index = -1
        
        for pkl_file in checkpoint_files:
            json_file = pkl_file.parent / pkl_file.name.replace('.pkl', '.json')
            
            if not json_file.exists():
                continue
            
            try:
                with open(json_file, 'r') as f:
                    metadata = json.load(f)
                    timestamp = metadata.get('timestamp', '')
                    combo_index = metadata.get('combination_index', -1)
                    
                    # Compare by timestamp first, then combination_index
                    is_newer = (
                        latest_timestamp is None or
                        timestamp > latest_timestamp or
                        (timestamp == latest_timestamp and combo_index > latest_index)
                    )
                    
                    if is_newer:
                        latest_timestamp = timestamp
                        latest_index = combo_index
                        latest_checkpoint = pkl_file
            except (json.JSONDecodeError, IOError):
                continue
        
        if latest_checkpoint is None:
            raise FileNotFoundError(
                f"Could not determine latest checkpoint for run_id={run_id}"
            )
        
        # Load checkpoint
        try:
            checkpoint_data = joblib.load(str(latest_checkpoint))
            return checkpoint_data
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint {latest_checkpoint}: {e}"
            )

    def validate_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Validate checkpoint file integrity.
        
        Checks:
        1. Pickle file is readable
        2. Corresponding JSON metadata exists and is valid
        3. Metadata contains required fields
        
        Args:
            checkpoint_path: Path to checkpoint pkl file
            
        Returns:
            True if valid, False otherwise
        """
        checkpoint_path = Path(checkpoint_path)
        json_path = checkpoint_path.parent / checkpoint_path.name.replace('.pkl', '.json')
        
        # Check pickle file
        try:
            joblib.load(str(checkpoint_path))
        except Exception:
            return False
        
        # Check JSON file exists
        if not json_path.exists():
            return False
        
        # Validate JSON content
        try:
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            
            # Verify required fields
            required_fields = ['run_id', 'phase', 'timestamp', 'checkpoint_version']
            for field in required_fields:
                if field not in metadata:
                    return False
            
            return True
        except (json.JSONDecodeError, IOError):
            return False

    def list_checkpoints(self, run_id: Optional[str] = None) -> list:
        """
        List all checkpoints, optionally filtered by run_id.
        
        Args:
            run_id: Optional run identifier to filter by
            
        Returns:
            List of checkpoint metadata dicts, sorted by timestamp
        """
        if run_id:
            pattern = f"{run_id}_phase_*.json"
        else:
            pattern = "*_phase_*.json"
        
        checkpoints = []
        for json_file in self.checkpoint_dir.glob(pattern):
            try:
                with open(json_file, 'r') as f:
                    metadata = json.load(f)
                    checkpoints.append(metadata)
            except (json.JSONDecodeError, IOError):
                continue
        
        # Sort by timestamp descending
        checkpoints.sort(
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )
        
        return checkpoints

    def delete_checkpoint(self, run_id: str, phase: str) -> bool:
        """
        Delete a specific checkpoint.
        
        Args:
            run_id: Run identifier
            phase: Phase identifier (e.g., 'phase_1a')
            
        Returns:
            True if deleted, False if not found
        """
        checkpoint_name = f"{run_id}_phase_{phase}"
        pkl_path = self.checkpoint_dir / f"{checkpoint_name}.pkl"
        json_path = self.checkpoint_dir / f"{checkpoint_name}.json"
        
        deleted = False
        
        if pkl_path.exists():
            pkl_path.unlink()
            deleted = True
        
        if json_path.exists():
            json_path.unlink()
            deleted = True
        
        return deleted

    def cleanup_old_checkpoints(self, keep_count: int = 5) -> int:
        """
        Clean up old checkpoints, keeping only recent ones per run_id.
        
        Args:
            keep_count: Number of most recent checkpoints to keep per run_id
            
        Returns:
            Number of checkpoints deleted
        """
        # Group checkpoints by run_id
        checkpoints_by_run = {}
        for json_file in self.checkpoint_dir.glob("*_phase_*.json"):
            try:
                with open(json_file, 'r') as f:
                    metadata = json.load(f)
                    run_id = metadata.get('run_id')
                    if run_id not in checkpoints_by_run:
                        checkpoints_by_run[run_id] = []
                    checkpoints_by_run[run_id].append(json_file)
            except (json.JSONDecodeError, IOError):
                continue
        
        deleted_count = 0
        
        # Delete old checkpoints, keeping only keep_count most recent
        for run_id, json_files in checkpoints_by_run.items():
            # Sort by modification time, newest first
            json_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            # Delete older ones
            for json_file in json_files[keep_count:]:
                pkl_file = json_file.parent / json_file.name.replace('.json', '.pkl')
                try:
                    json_file.unlink()
                    if pkl_file.exists():
                        pkl_file.unlink()
                    deleted_count += 1
                except OSError:
                    pass
        
        return deleted_count
