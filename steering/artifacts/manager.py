"""
Artifacts manager for saving calibration data
"""

import torch
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from ..utils.logger import setup_logger


class ArtifactsManager:
    """Manage saving and loading of calibration artifacts"""

    def __init__(self, output_dir: str = "artifacts", name: str = "default"):
        """Initialize artifacts manager for a specific calibration session"""
        self.base_dir = Path(output_dir)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_dir / f"calibration_{name}_{self.session_id}"

        # Setup logger
        self.logger = setup_logger(obj=self)
    
    def _save_activations(
        self,
        harmful_activations: Dict[str, torch.Tensor],
        harmless_activations: Dict[str, torch.Tensor]
    ) -> str:
        """Save extracted activations to session directory"""
        filepath = self.session_dir / "activations.pt"

        torch.save({
            'harmful': harmful_activations,
            'harmless': harmless_activations,
            'metadata': {
                'n_layers': len(harmful_activations),
                'layer_names': list(harmful_activations.keys()),
                'harmful_samples': list(harmful_activations.values())[0].shape[0],
                'harmless_samples': list(harmless_activations.values())[0].shape[0],
                'hidden_dim': list(harmful_activations.values())[0].shape[1],
                'session_id': self.session_id
            }
        }, filepath)

        self.logger.info(f"✓ Saved activations to {filepath}")
        return str(filepath)
    
    def _save_directions(
        self,
        candidates: Dict[str, torch.Tensor],
        selected_direction: torch.Tensor,
        selected_layer: str
    ) -> str:
        """Save candidate and selected feature directions to session directory"""
        filepath = self.session_dir / "directions.pt"

        torch.save({
            'candidates': candidates,
            'selected_direction': selected_direction,
            'selected_layer': selected_layer,
            'metadata': {
                'n_candidates': len(candidates),
                'layer_names': list(candidates.keys()),
                'hidden_dim': list(candidates.values())[0].shape[0],
                'session_id': self.session_id
            }
        }, filepath)

        self.logger.info(f"✓ Saved directions to {filepath}")
        return str(filepath)
    
    def _save_plane(
        self,
        basis: tuple,
        projections: Dict[str, tuple],
        plane_type: str = "standard",
        extra_info: Optional[Dict] = None
    ) -> str:
        """Save steering plane and projections to session directory"""
        filepath = self.session_dir / "plane.pt"

        torch.save({
            'basis': basis,
            'projections': projections,
            'plane_type': plane_type,
            'extra_info': extra_info or {},
            'metadata': {
                'n_projections': len(projections),
                'session_id': self.session_id
            }
        }, filepath)

        self.logger.info(f"✓ Saved steering plane to {filepath}")
        return str(filepath)
    
    def save_calibration_artifacts(
        self,
        harmful_activations: Dict[str, torch.Tensor],
        harmless_activations: Dict[str, torch.Tensor],
        candidates: Dict[str, torch.Tensor],
        selected_direction: torch.Tensor,
        selected_layer: str,
        basis: tuple,
        projections: Dict[str, tuple],
        config: Dict[str, Any],
        plane_type: str = "adaptive"
    ) -> str:
        """
        Universal function to save all calibration artifacts with consistent timestamping.

        This consolidates all artifact saving operations into a single method that:
        - Saves individual components (activations, directions, plane)
        - Creates a complete calibration bundle
        - Uses consistent session-based timestamping
        - Eliminates duplication from the pipeline

        Args:
            harmful_activations: Harmful activation tensors by layer
            harmless_activations: Harmless activation tensors by layer
            candidates: Candidate direction tensors by layer
            selected_direction: The selected steering direction
            selected_layer: The layer selected for steering
            basis: Steering plane basis vectors (b1, b2)
            projections: Layer projections onto the steering plane
            config: Configuration dictionary
            plane_type: Type of steering plane (e.g., 'adaptive', 'angular')

        Returns:
            Path to the session directory containing all artifacts
        """
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Artifacts session created at: {self.session_dir}")
        
        self.logger.info("[Saving Calibration Artifacts]")

        # Save individual components using private methods
        self._save_activations(harmful_activations, harmless_activations)
        self._save_directions(candidates, selected_direction, selected_layer)
        self._save_plane(basis, projections, plane_type)

        # Save configuration
        config_path = self.session_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        self.logger.info(f"✓ Saved configuration to {config_path}")

        # Create session metadata
        metadata = {
            'session_id': self.session_id,
            'selected_layer': selected_layer,
            'plane_type': plane_type,
            'n_layers': len(candidates),
            'harmful_samples': list(harmful_activations.values())[0].shape[0],
            'harmless_samples': list(harmless_activations.values())[0].shape[0],
            'hidden_dim': list(harmful_activations.values())[0].shape[1]
        }

        metadata_path = self.session_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"✓ Saved metadata to {metadata_path}")

        self.logger.info(f"✓ Complete calibration session saved to {self.session_dir}")
        return str(self.session_dir)

    # Backward compatibility methods (deprecated but maintained for legacy code)
    def save_activations(self, harmful_activations: Dict[str, torch.Tensor],
                        harmless_activations: Dict[str, torch.Tensor], name: str = "default") -> str:
        """Legacy method - use save_calibration_artifacts instead"""
        self.logger.warning("save_activations is deprecated, use save_calibration_artifacts instead")
        return self._save_activations(harmful_activations, harmless_activations)

    def save_directions(self, candidates: Dict[str, torch.Tensor], selected_direction: torch.Tensor,
                       selected_layer: str, name: str = "default") -> str:
        """Legacy method - use save_calibration_artifacts instead"""
        self.logger.warning("save_directions is deprecated, use save_calibration_artifacts instead")
        return self._save_directions(candidates, selected_direction, selected_layer)

    def save_plane(self, basis: tuple, projections: Dict[str, tuple], plane_type: str = "standard",
                   name: str = "default", extra_info: Optional[Dict] = None) -> str:
        """Legacy method - use save_calibration_artifacts instead"""
        self.logger.warning("save_plane is deprecated, use save_calibration_artifacts instead")
        return self._save_plane(basis, projections, plane_type, extra_info)

    def save_calibration(self, harmful_activations: Dict[str, torch.Tensor],
                        harmless_activations: Dict[str, torch.Tensor], candidates: Dict[str, torch.Tensor],
                        selected_direction: torch.Tensor, selected_layer: str, basis: tuple,
                        projections: Dict[str, tuple], config: Dict[str, Any], name: str = "default") -> str:
        """Legacy method - use save_calibration_artifacts instead"""
        self.logger.warning("save_calibration is deprecated, use save_calibration_artifacts instead")
        return self.save_calibration_artifacts(
            harmful_activations, harmless_activations, candidates, selected_direction,
            selected_layer, basis, projections, config, "adaptive"
        )

    def load_calibration(self, bundle_dir: str) -> Dict[str, Any]:
        """Load complete calibration bundle"""
        bundle_path = Path(bundle_dir)
        
        # Because activations are large, we can ignore
        try:
            activations = torch.load(bundle_path / "activations.pt")
        except FileNotFoundError as e:
            self.logger.warning("Activations are not loaded!")
            activations = {}
        directions = torch.load(bundle_path / "directions.pt")
        plane = torch.load(bundle_path / "plane.pt")
        
        with open(bundle_path / "config.json", 'r') as f:
            config = json.load(f)
        
        with open(bundle_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.logger.info(f"✓ Loaded calibration from {bundle_dir}")

        return {
            'activations': activations,
            'directions': directions,
            'plane': plane,
            'config': config,
            'metadata': metadata
        }