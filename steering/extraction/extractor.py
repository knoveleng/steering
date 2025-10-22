"""
Activation extraction implementation
"""

import torch
from typing import Dict, List, Optional, Any
from collections import defaultdict

from .base import BaseActivationExtractor


class ActivationExtractor(BaseActivationExtractor):
    """
    Extract activations from transformer models
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: str = 'cuda'
    ):
        """
        Initialize extractor
        
        Args:
            model: The model to extract from
            tokenizer: Tokenizer for the model
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        self.hooks = []
        self.activations = defaultdict(list)
        self.target_layers = []
    
    def register_hooks(self, layer_names: List[str]) -> None:
        """Register forward hooks on layers"""
        self.remove_hooks()
        self.target_layers = layer_names
        
        for name, module in self.model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(
                    self._create_hook(name)
                )
                self.hooks.append(hook)
    
    def _create_hook(self, layer_name: str):
        """Create a forward hook for a layer"""
        def hook(module, input, output):
            # Store the output activation
            # Take the last token's activation
            if isinstance(output, tuple):
                output = output[0]
            
            # Extract last token activation
            last_token_act = output[:, -1, :].detach()
            self.activations[layer_name].append(last_token_act)
        
        return hook
    
    def extract_activations(
        self,
        prompts: List[str],
        normalize: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Extract activations for prompts
        
        Args:
            prompts: Input prompts
            normalize: Whether to normalize activations
            **kwargs: Additional generation arguments
            
        Returns:
            Dict mapping layer names to activation tensors
        """
        # Clear previous activations
        self.activations.clear()
        
        self.model.eval()
        with torch.no_grad():
            for prompt in prompts:
                # Tokenize
                inputs = self.tokenizer(
                    prompt,
                    return_tensors='pt',
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                # Forward pass (hooks will capture activations)
                _ = self.model(**inputs)
        
        # Concatenate activations
        result = {}
        for layer_name in self.target_layers:
            acts = torch.cat(self.activations[layer_name], dim=0)
            
            # if normalize:
            #     # Normalize to unit sphere (per activation)
            #     # Preserve dtype during normalization
            #     original_dtype = acts.dtype
            #     norms = torch.linalg.norm(acts, dim=-1, keepdim=True)
            #     # Add epsilon to avoid division by zero
            #     acts = acts / (norms + 1e-8)
            #     # Ensure dtype is preserved
            #     acts = acts.to(dtype=original_dtype)
            
            result[layer_name] = acts
        
        return result
    
    def remove_hooks(self) -> None:
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_layer_names(self) -> List[str]:
        """Get all layer names in model"""
        return [name for name, _ in self.model.named_modules()]
    
    def get_normalization_layers(self) -> List[str]:
        """
        Get names of normalization layers
        
        Returns:
            List of layer names that are normalization layers
        """
        norm_layers = []
        
        for name, module in self.model.named_modules():
            # Check for common normalization layer types
            # if any(norm_type in module.__class__.__name__.lower() 
            #        for norm_type in ['norm', 'layernorm', 'rmsnorm']):
            # if any(norm_type in module.__class__.__name__.lower() 
            #        for norm_type in ['layernorm', 'rmsnorm']):
            #     norm_layers.append(name)
            if any(norm_type in name.lower() 
                   for norm_type in ['layernorm', 'rmsnorm']):
                norm_layers.append(name)
        
        return norm_layers