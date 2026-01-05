"""
SteeringLLM - vLLM LLM wrapper with activation steering support.

Provides a high-level interface for running vLLM inference with steering enabled.
Supports standard, adaptive, and selective steering modes.
"""

import os
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any

# Set insecure serialization for vLLM v0.12+
os.environ.setdefault('VLLM_ALLOW_INSECURE_SERIALIZATION', '1')

from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

from .hooks import create_steering_hook, clear_hooks
from .operators import SteeringMode, DEFAULT_MODE, DEFAULT_THRESHOLD
from ..utils.logger import setup_logger

logger = setup_logger(name="steering.vllm_steering.llm")


class SteeringLLM:
    """
    vLLM LLM wrapper with activation steering support.
    
    Supports three steering modes:
    - standard: Angular rotation on all target layers
    - adaptive: Conditional steering based on alignment threshold
    - selective: Layer-specific steering based on projection analysis
    
    Key features:
    - Accept pre-loaded calibration data from ui.utils.load_calibration
    - Hooks are registered once and theta/enabled state can be updated fast
    """
    
    def __init__(
        self,
        model: str,
        b1: torch.Tensor,
        b2: torch.Tensor,
        mode: str = "standard",
        threshold: float = 0.0,
        target_layers: Optional[List[str]] = None,
        layer_mask: Optional[Dict[str, bool]] = None,
        **vllm_kwargs,
    ):
        """
        Initialize SteeringLLM with calibration data.
        
        Args:
            model: Model name or path
            b1: First basis vector (from calibration)
            b2: Second basis vector (from calibration)
            mode: Steering mode - "standard", "adaptive", or "selective"
            threshold: Alignment threshold for adaptive mode
            target_layers: Optional list of layer names to hook
            layer_mask: Layer selection mask for selective mode
            **vllm_kwargs: Additional arguments for vLLM
        """
        self._model_name = model
        self._b1 = b1
        self._b2 = b2
        self._mode = mode
        self._threshold = threshold
        self._target_layers = target_layers
        self._layer_mask = layer_mask
        
        # State - no default steering until set_theta called
        self._theta: Optional[float] = None
        self._steering_enabled = False
        
        # Initialize vLLM
        logger.info(f"Initializing vLLM with model: {model}")
        self.llm = LLM(model, **vllm_kwargs)
        
        # Register hooks
        self._hooks_registered = False
        self._register_hooks()
        
        # Log summary
        n_hooked = len(self._target_layers) if self._target_layers else "auto"
        if self._mode == "selective" and self._layer_mask:
            n_selected = sum(1 for v in self._layer_mask.values() if v)
            logger.info(f"SteeringLLM initialized: mode={self._mode}, layers={n_selected}/{n_hooked} selected")
        else:
            logger.info(f"SteeringLLM initialized: mode={self._mode}, layers={n_hooked}")
    
    @classmethod
    def from_calibration(
        cls,
        calibration_data: Dict[str, Any],
        **vllm_kwargs,
    ) -> "SteeringLLM":
        """
        Create SteeringLLM from calibration data dict.
        
        Args:
            calibration_data: Dict from ui.utils.load_calibration()
            **vllm_kwargs: Additional arguments for vLLM
            
        Returns:
            SteeringLLM instance
        """
        model_name = calibration_data.get("model_name")
        if not model_name:
            raise ValueError("Calibration data missing model_name")
        
        return cls(
            model=model_name,
            b1=calibration_data["b1"],
            b2=calibration_data["b2"],
            mode=calibration_data.get("mode", "standard"),
            threshold=calibration_data.get("threshold", 0.0),
            target_layers=calibration_data.get("target_layers"),
            layer_mask=calibration_data.get("layer_mask"),
            **vllm_kwargs,
        )
    
    def _register_hooks(self):
        """Register steering hooks on the model (one-time setup)."""
        from .operators import create_operator
        
        b1 = self._b1
        b2 = self._b2
        target_layers = self._target_layers
        layer_mask = self._layer_mask
        mode = self._mode
        threshold = self._threshold
        initial_theta = self._theta or 0.0
        initial_enabled = self._steering_enabled
        
        # Create a single shared operator for all hooks
        shared_operator = create_operator(
            mode=mode,
            b1=b1,
            b2=b2,
            threshold=threshold,
            layer_mask=layer_mask,
        )
        
        def register_hooks_fn(model: nn.Module):
            """Register hooks on target layers in worker process."""
            # Create shared mutable state in worker process
            import builtins
            if not hasattr(builtins, '_steering_state'):
                builtins._steering_state = {}
            builtins._steering_state['theta'] = initial_theta
            builtins._steering_state['enabled'] = initial_enabled
            builtins._steering_state['last_theta'] = None  # Track last theta for cache clearing
            
            # Store operator reference for cache clearing
            builtins._steering_operator = shared_operator
            
            # Remove existing hooks
            clear_hooks(model)
            
            count = 0
            hooked_layers = []
            
            for name, module in model.named_modules():
                should_hook = False
                
                if target_layers is not None:
                    # Use specified target layers
                    should_hook = name in target_layers
                else:
                    # Auto-detect: hook normalization layers (layernorm, rmsnorm)
                    should_hook = any(norm_type in name.lower() 
                                      for norm_type in ['layernorm', 'rmsnorm'])
                
                if should_hook:
                    hook = create_steering_hook(
                        operator=shared_operator,
                        state=builtins._steering_state,
                        layer_name=name,
                    )
                    module.register_forward_hook(hook)
                    count += 1
                    hooked_layers.append(name)
            
            return count
        
        # Apply to model in worker process
        results = self.llm.apply_model(register_hooks_fn)
        self._hooks_registered = True
        logger.info(f"Registered steering hooks: {results} (mode={mode})")
    
    def _update_state(self, theta: Optional[float] = None, enabled: Optional[bool] = None):
        """Update the steering state in the worker process (fast, no hook re-registration)."""
        new_theta = theta if theta is not None else (self._theta or 0.0)
        new_enabled = enabled if enabled is not None else self._steering_enabled
        
        def update_state_fn(model: nn.Module):
            import builtins
            if hasattr(builtins, '_steering_state'):
                builtins._steering_state['theta'] = new_theta
                builtins._steering_state['enabled'] = new_enabled
            return (new_theta, new_enabled)
        
        self.llm.apply_model(update_state_fn)
        
        if theta is not None:
            self._theta = theta
        if enabled is not None:
            self._steering_enabled = enabled
        
        logger.info(f"Updated steering state: theta={new_theta}, enabled={new_enabled}")
    
    def set_theta(self, theta: float):
        """Set the steering angle in degrees."""
        self._steering_enabled = True
        self._update_state(theta=theta, enabled=True)
    
    def enable_steering(self):
        """Enable steering."""
        self._update_state(enabled=True)
    
    def disable_steering(self):
        """Disable steering."""
        self._update_state(enabled=False)
    
    def set_operator(self, operator):
        """
        Change the steering operator without reloading the model.
        
        This allows switching between different steering methods (standard, 
        adaptive, addition, ablation) at runtime.
        
        Args:
            operator: New steering operator instance
        """
        new_operator = operator
        
        def update_operator_fn(model: nn.Module):
            """Update the operator reference in worker process."""
            import builtins
            if hasattr(builtins, '_steering_operator'):
                builtins._steering_operator = new_operator
            return True
        
        self.llm.apply_model(update_operator_fn)
        logger.info(f"Changed operator to: {type(operator).__name__}")
    
    def generate(
        self,
        prompts: List[str],
        theta: Optional[float] = None,
        sampling_params: Optional[SamplingParams] = None,
        **kwargs,
    ) -> List[RequestOutput]:
        """
        Generate text with optional steering.
        
        Args:
            prompts: List of prompts
            theta: Steering angle (uses current if not specified)
            sampling_params: vLLM sampling parameters
            **kwargs: Additional arguments for vLLM generate
            
        Returns:
            List of RequestOutput
        """
        if theta is not None and theta != self._theta:
            self.set_theta(theta)
        
        if sampling_params is None:
            sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
        
        return self.llm.generate(prompts, sampling_params, **kwargs)
    
    @property
    def theta(self) -> Optional[float]:
        """Current steering angle."""
        return self._theta
    
    @property
    def steering_enabled(self) -> bool:
        """Whether steering is enabled."""
        return self._steering_enabled
    
    @property
    def mode(self) -> str:
        """Steering mode."""
        return self._mode
