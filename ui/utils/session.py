"""
Session management utilities
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import torch
from datetime import datetime


def load_calibration(calibration_dir: str) -> Dict[str, Any]:
    """
    Load calibration artifacts from a saved session.
    
    This loads the basis vectors, mode, layer masks, and other settings
    from a calibration session saved by AngularSteeringPipeline.
    
    Args:
        calibration_dir: Path to calibration directory
        
    Returns:
        Dictionary containing:
            - model_name: Model name from config
            - mode: Steering mode (standard, adaptive, selective)
            - threshold: Alignment threshold for adaptive mode
            - b1: First basis vector
            - b2: Second basis vector
            - layer_mask: Layer selection mask (for selective mode)
            - target_layers: List of target layer names (for selective mode)
            
    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If files are invalid
    """
    bundle_path = Path(calibration_dir)
    
    # Load plane data (contains basis vectors and layer mask)
    plane_path = bundle_path / "plane.pt"
    if not plane_path.exists():
        raise FileNotFoundError(f"plane.pt not found in {calibration_dir}")
    
    plane_data = torch.load(plane_path, weights_only=False)
    
    # Handle different plane.pt formats
    if "basis" in plane_data:
        # New format: basis is a tuple of (b1, b2)
        basis = plane_data["basis"]
        if isinstance(basis, tuple):
            b1, b2 = basis
        else:
            b1, b2 = basis["b1"], basis["b2"]
    elif "b1" in plane_data and "b2" in plane_data:
        # Old format: separate b1 and b2 keys
        b1 = plane_data["b1"]
        b2 = plane_data["b2"]
    else:
        raise ValueError(f"Invalid plane.pt format: keys={list(plane_data.keys())}")
    
    # Ensure basis vectors are 1D
    if b1.dim() > 1:
        b1 = b1.squeeze()
    if b2.dim() > 1:
        b2 = b2.squeeze()
    
    # Load config
    config_path = bundle_path / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Extract model name
    model_name = config.get("model", {}).get("name")
    
    # Get mode from config
    mode = config.get("steering", {}).get("mode", "standard")
    
    # Get threshold for adaptive mode
    threshold = config.get("steering", {}).get("threshold", 0.0)
    
    # Get layer mask from plane extra_info (for selective mode)
    extra_info = plane_data.get("extra_info", {})
    layer_mask = extra_info.get("layer_steering_mask", None)
    
    # Get target layers - prioritize from extra_info (new format)
    # Fallback to layer_mask keys for backward compatibility
    target_layers = extra_info.get("target_layers", None)
    if target_layers is None and layer_mask is not None:
        target_layers = list(layer_mask.keys())  # Backward compatibility
    
    return {
        "model_name": model_name,
        "mode": mode,
        "threshold": threshold,
        "b1": b1,
        "b2": b2,
        "layer_mask": layer_mask,
        "target_layers": target_layers,
        "config": config,
    }


class SessionManager:
    """Manage calibration sessions"""
    
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
    
    def list_sessions(self, model_name: Optional[str] = None) -> List[Dict]:
        """
        List available calibration sessions
        
        Args:
            model_name: Optional model name to filter
            
        Returns:
            List of session info dictionaries
        """
        if not self.artifacts_dir.exists():
            return []
        
        pattern = "calibration_*"
        if model_name:
            pattern = f"calibration_{model_name}_*"
        
        sessions = []
        for session_dir in self.artifacts_dir.glob(pattern):
            if session_dir.is_dir():
                info = self.get_session_info(session_dir)
                sessions.append(info)
        
        # Sort by creation time (newest first)
        sessions.sort(key=lambda x: x['modified'], reverse=True)
        
        return sessions
    
    def get_session_info(self, session_path: Path) -> Dict:
        """
        Get information about a session
        
        Args:
            session_path: Path to session directory
            
        Returns:
            Dictionary with session information
        """
        info = {
            'path': str(session_path),
            'name': session_path.name,
            'modified': session_path.stat().st_mtime,
            'modified_str': datetime.fromtimestamp(
                session_path.stat().st_mtime
            ).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Try to load config
        config_file = session_path / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                info['model'] = config.get('model', {}).get('name', 'unknown')
                info['mode'] = config.get('steering', {}).get('mode', 'standard')
            except:
                pass
        
        # Check for required files
        required_files = ['plane.pt', 'config.json']
        info['complete'] = all(
            (session_path / f).exists() for f in required_files
        )
        
        return info
    
    def validate_session(self, session_path: Path) -> bool:
        """
        Validate that a session has all required files
        
        Args:
            session_path: Path to session directory
            
        Returns:
            True if valid, False otherwise
        """
        required_files = ['plane.pt', 'config.json']
        return all((session_path / f).exists() for f in required_files)