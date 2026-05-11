"""UnifiedModelRegistry: Manages artifacts from Phase 1, Phase 2, and NN."""

import json
from pathlib import Path
from typing import Dict, List, Optional


class UnifiedModelRegistry:
    """
    Manages artifacts from Phase 1, Phase 2, and NN in unified structure.
    """
    
    def __init__(self, base_dir: str = "models"):
        self.base_dir = base_dir
        self.registry = {
            "phase1": {"path": f"{base_dir}/phase1", "models": []},
            "phase2": {"path": f"{base_dir}/phase2", "models": []},
            "nn": {"path": f"{base_dir}/nn", "models": []},
        }
    
    def scan_models(self) -> Dict[str, List[Dict]]:
        """Scan all directories for best models."""
        models = {}
        
        # Phase 1
        p1_path = Path(self.registry["phase1"]["path"]) / "final"
        if p1_path.exists():
            p1_models = list(p1_path.glob("**/best_model.joblib"))
            models["phase1"] = [{"run_id": p.parent.name, "path": str(p)} for p in p1_models]
        
        # Phase 2
        p2_path = Path(self.registry["phase2"]["path"]) / "final"
        if p2_path.exists():
            p2_models = list(p2_path.glob("**/best_model.joblib"))
            models["phase2"] = [{"run_id": p.parent.name, "path": str(p)} for p in p2_models]
        
        # NN
        nn_path = Path(self.registry["nn"]["path"]) / "final"
        if nn_path.exists():
            nn_models = list(nn_path.glob("**/best_model.pt"))
            models["nn"] = [{"run_id": p.parent.name, "path": str(p)} for p in nn_models]
        
        return models
    
    def get_latest_models(self) -> Dict[str, Dict]:
        """Get most recent best model from each phase."""
        models = self.scan_models()
        latest = {}
        
        for phase, model_list in models.items():
            if model_list:
                # Sort by run_id (timestamp) and take latest
                latest[phase] = sorted(
                    model_list,
                    key=lambda x: x["run_id"],
                    reverse=True
                )[0]
        
        return latest
    
    def get_model_metadata(self, phase: str, run_id: str) -> dict:
        """Load metadata.json for a model."""
        metadata_path = Path(self.registry[phase]["path"]) / "final" / run_id / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        return {}
