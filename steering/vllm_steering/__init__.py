"""
vLLM Steering Module

Provides vLLM integration for activation steering with high-performance inference.

Supports steering modes:
- standard: Angular rotation on all target layers
- adaptive: Conditional steering based on alignment threshold  
- selective: Layer-specific steering based on projection analysis
- addition: Vector addition steering (special case of angular)
- ablation: Directional ablation (special case of angular)
"""

from .hooks import create_steering_hook, clear_hooks, get_target_layer_names
from .operators import (
    SteeringMode, 
    DEFAULT_MODE, 
    DEFAULT_THRESHOLD,
    BaseSteeringOperator,
    StandardSteeringOperator,
    AdaptiveSteeringOperator,
    SelectiveSteeringOperator,
    AdditionSteeringOperator,
    AblationSteeringOperator,
    create_operator,
)
from .llm import SteeringLLM

__all__ = [
    # Main class
    "SteeringLLM",
    # Operators
    "SteeringMode",
    "DEFAULT_MODE",
    "DEFAULT_THRESHOLD",
    "BaseSteeringOperator",
    "StandardSteeringOperator",
    "AdaptiveSteeringOperator",
    "SelectiveSteeringOperator",
    "AdditionSteeringOperator",
    "AblationSteeringOperator",
    "create_operator",
    # Hooks
    "create_steering_hook",
    "clear_hooks",
    "get_target_layer_names",
]
