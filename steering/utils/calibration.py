"""
Calibration utilities for loading saved steering calibrations.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import torch


def load_calibration(calibration_dir: str, mode: Optional[str] = None) -> Dict[str, Any]:
    """
    Load calibration artifacts from a saved session.
    
    This loads the basis vectors, mode, layer masks, and other settings
    from a calibration session saved by AngularSteeringPipeline.
    
    Args:
        calibration_dir: Path to calibration directory
        mode: Optional mode override (standard, adaptive, selective, addition, ablation)
        
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
    
    # Get mode from config or use override
    calibration_mode = config.get("steering", {}).get("mode", "standard")
    if mode is not None:
        calibration_mode = mode
    
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
        "mode": calibration_mode,
        "threshold": threshold,
        "b1": b1,
        "b2": b2,
        "layer_mask": layer_mask,
        "target_layers": target_layers,
        "config": config,
    }
