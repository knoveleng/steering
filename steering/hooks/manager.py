"""
Hook manager implementation
"""

import torch
from typing import List, Optional, Dict, Any, Callable

from .base import BaseHookManager
from ..steering import BaseSteeringOperator


class ModelHookManager(BaseHookManager):
    """
    Manage hooks for steering during model inference
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        steering_operator: BaseSteeringOperator
    ):
        """
        Initialize hook manager
        
        Args:
            model: The model to hook
            steering_operator: Steering operator to apply
        """
        self.model = model
        self.steering_operator = steering_operator
        
        self.hooks = []
        self.steering_enabled = True
        self.current_theta = None
    
    def register_hooks(
        self,
        layer_names: List[str],
        steering_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register steering hooks on layers
        
        Args:
            layer_names: Layer names to hook
            steering_params: Dict with 'theta' key for angle
        """
        # Remove existing hooks
        self.remove_hooks()
        
        # Extract theta from params
        if steering_params is not None:
            self.current_theta = steering_params.get('theta', 0.0)
        
        # Register hooks on target layers
        for name, module in self.model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(
                    self._create_steering_hook()
                )
                self.hooks.append((name, hook))
    
    def _create_steering_hook(self) -> Callable:
        """Create a forward hook that applies steering"""
        def hook(module, input, output):
            if not self.steering_enabled:
                return output
            
            # Handle tuple outputs (some models return (hidden_state, ...))
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None
            
            # Apply steering
            steered = self.steering_operator.steer_activation(
                hidden_states,
                self.current_theta
            )
            
            # Return in same format as input
            if rest is not None:
                return (steered,) + rest
            else:
                return steered
        
        return hook
    
    def remove_hooks(self) -> None:
        """Remove all hooks"""
        for name, hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def enable_steering(self) -> None:
        """Enable steering"""
        self.steering_enabled = True
    
    def disable_steering(self) -> None:
        """Disable steering (hooks stay registered)"""
        self.steering_enabled = False
    
    def set_theta(self, theta: float) -> None:
        """
        Update steering angle
        
        Args:
            theta: New angle in degrees
        """
        self.current_theta = theta
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up hooks"""
        self.remove_hooks()