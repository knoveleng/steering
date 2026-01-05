"""
vLLM Steering Hooks

Contains hook creation and management functions for steering in vLLM models.
Uses operator classes for steering logic.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Callable

from .operators import create_operator, BaseSteeringOperator, SteeringMode
from ..utils.logger import setup_logger

logger = setup_logger(name="steering.vllm_steering.hooks")


def create_steering_hook(
    operator: BaseSteeringOperator,
    state: Dict,
    layer_name: str,
) -> Callable:
    """
    Create a steering hook with mutable theta state using a shared operator.
    
    Args:
        operator: Shared steering operator instance (created once per model)
        state: Mutable dict containing 'theta', 'enabled', and 'last_theta' keys
        layer_name: Name of the layer this hook is attached to
        
    Returns:
        Hook function
    """
    _layer_name = layer_name
    # Store initial operator reference for fallback
    _initial_operator = operator
    
    def hook(module, input, output):
        import builtins
        
        # Read mutable state
        theta = state.get('theta', 0.0)
        enabled = state.get('enabled', True)
        
        if not enabled:
            return output
        
        # Get current operator - check builtins first for dynamic updates
        current_operator = getattr(builtins, '_steering_operator', _initial_operator)
        
        # Clear rotation cache when theta changes to prevent OOM
        last_theta = state.get('last_theta', None)
        if last_theta is not None and last_theta != theta:
            current_operator.clear_rotation_cache()
        state['last_theta'] = theta
        
        # Handle tuple outputs
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None
        
        # Apply steering using current operator
        steered = current_operator.steer(hidden_states, theta, _layer_name)
        
        if rest is not None:
            return (steered,) + rest
        return steered
    
    return hook


def clear_hooks(model: nn.Module) -> int:
    """
    Clear all forward hooks from a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of hooks cleared
    """
    count = 0
    for module in model.modules():
        if hasattr(module, '_forward_hooks') and module._forward_hooks:
            count += len(module._forward_hooks)
            module._forward_hooks.clear()
    return count


def get_target_layer_names(model: nn.Module, pattern: str = "input_layernorm") -> list:
    """
    Get names of layers matching a pattern.
    
    Args:
        model: PyTorch model
        pattern: Pattern to match in layer names
        
    Returns:
        List of matching layer names
    """
    return [name for name, _ in model.named_modules() if pattern in name]
