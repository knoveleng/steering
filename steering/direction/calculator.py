"""
Feature direction calculation implementation
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .base import BaseFeatureDirectionCalculator


class FeatureDirectionCalculator(BaseFeatureDirectionCalculator):
    """
    Calculate feature directions using difference-in-means
    """
    
    def __init__(self, method: str = 'diff_in_means'):
        """
        Initialize calculator
        
        Args:
            method: Method for computing directions
                    ('diff_in_means', 'pca', 'ica')
        """
        self.method = method
    
    def compute_candidate_directions(
        self,
        positive_activations: Dict[str, torch.Tensor],
        negative_activations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute candidate directions using difference in means
        
        Args:
            positive_activations: Dict[layer, Tensor(n_samples, hidden_dim)]
            negative_activations: Dict[layer, Tensor(n_samples, hidden_dim)]
            
        Returns:
            Dict[layer, Tensor(hidden_dim)]
        """
        if self.method == 'diff_in_means':
            return self._diff_in_means(positive_activations, negative_activations)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _diff_in_means(
        self,
        positive_activations: Dict[str, torch.Tensor],
        negative_activations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Difference-in-means method"""
        candidates = {}
        
        for layer_name in positive_activations.keys():
            if layer_name not in negative_activations:
                continue
            
            # Compute means
            mean_positive = positive_activations[layer_name].mean(dim=0)
            mean_negative = negative_activations[layer_name].mean(dim=0)
            
            # Compute direction
            direction = mean_positive - mean_negative
            
            # Normalize to unit vector
            direction = direction / (direction.norm() + 1e-8)
            
            candidates[layer_name] = direction
        
        return candidates
    
    def select_best_direction(
        self,
        candidates: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, str]:
        """
        Select direction with highest average cosine similarity
        
        Args:
            candidates: Dict[layer_name, direction_vector]
            
        Returns:
            (best_direction, best_layer_name)
        """
        layer_names = list(candidates.keys())
        n_layers = len(layer_names)
        
        if n_layers == 0:
            raise ValueError("No candidate directions provided")
        
        if n_layers == 1:
            layer_name = layer_names[0]
            return candidates[layer_name], layer_name
        
        # Compute similarity matrix
        similarities = torch.zeros(n_layers, n_layers)
        
        for i, layer_i in enumerate(layer_names):
            for j, layer_j in enumerate(layer_names):
                # Ensure both tensors are float32 for compatibility
                vec_i = candidates[layer_i].float()
                vec_j = candidates[layer_j].float()
                
                sim = F.cosine_similarity(
                    vec_i.unsqueeze(0),
                    vec_j.unsqueeze(0)
                )
                similarities[i, j] = sim.item()
        
        # Find layer with maximum average similarity
        avg_similarities = similarities.mean(dim=1)
        best_idx = avg_similarities.argmax().item()
        best_layer = layer_names[best_idx]
        
        return candidates[best_layer], best_layer
    
    def compute_similarity_matrix(
        self,
        candidates: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute pairwise cosine similarity matrix
        
        Args:
            candidates: Candidate directions
            
        Returns:
            Similarity matrix of shape (n_layers, n_layers)
        """
        layer_names = list(candidates.keys())
        n_layers = len(layer_names)
        
        similarities = torch.zeros(n_layers, n_layers)
        
        for i, layer_i in enumerate(layer_names):
            for j, layer_j in enumerate(layer_names):
                # Convert to float32 for compatibility
                vec_i = candidates[layer_i].float()
                vec_j = candidates[layer_j].float()
                
                sim = F.cosine_similarity(
                    vec_i.unsqueeze(0),
                    vec_j.unsqueeze(0)
                )
                similarities[i, j] = sim.item()
        
        return similarities