"""
vLLM integration for fast inference with steering
"""

import torch
import logging
from typing import List, Dict, Any, Optional
from vllm import LLM, SamplingParams
from vllm.model_executor.layers.layernorm import RMSNorm
import gc

from .base import BaseModelServer
from ..steering import BaseSteeringOperator
from ..utils.logger import setup_logger


class VLLMSteeringServer(BaseModelServer):
    """
    vLLM server with steering support
    """
    
    def __init__(
        self,
        model_name: str,
        steering_operator: Optional[BaseSteeringOperator] = None,
        target_layers: Optional[List[str]] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "bfloat16",
        **vllm_kwargs
    ):
        """
        Initialize vLLM server with steering
        
        Args:
            model_name: Model name or path
            steering_operator: Steering operator to use
            target_layers: Layer names to apply steering (None = auto-detect)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization (0-1)
            dtype: Model dtype
            **vllm_kwargs: Additional vLLM arguments
        """
        self.model_name = model_name
        self.steering_operator = steering_operator
        self.target_layers = target_layers
        self.steering_enabled = False
        self.current_theta = 0.0

        # Setup logger
        self.logger = setup_logger(obj=self)

        # Initialize vLLM
        self.logger.info(f"Initializing vLLM with {model_name}...")
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            trust_remote_code=True,
            **vllm_kwargs
        )

        self.hooks = []

        # Auto-detect target layers if not provided
        if self.target_layers is None and self.steering_operator is not None:
            self.target_layers = self._get_normalization_layers()
            self.logger.info(f"Auto-detected {len(self.target_layers)} normalization layers")

        # Register hooks if steering operator is provided
        if self.steering_operator is not None and self.target_layers:
            self._register_steering_hooks()
    
    def _get_normalization_layers(self) -> List[str]:
        """Get normalization layer names from model"""
        norm_layers = []
        model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.LayerNorm, RMSNorm)) or \
               'norm' in module.__class__.__name__.lower():
                norm_layers.append(name)
        
        return norm_layers
    
    def _create_steering_hook(self):
        """Create forward hook for steering"""
        def hook(module, input, output):
            if not self.steering_enabled or self.steering_operator is None:
                return output
            
            # Handle tuple outputs
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
            
            # Return in same format
            if rest is not None:
                return (steered,) + rest
            else:
                return steered
        
        return hook
    
    def _register_steering_hooks(self):
        """Register steering hooks on target layers"""
        model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        
        for name, module in model.named_modules():
            if name in self.target_layers:
                hook = module.register_forward_hook(self._create_steering_hook())
                self.hooks.append((name, hook))
        
        self.logger.info(f"Registered steering hooks on {len(self.hooks)} layers")
    
    def _remove_hooks(self):
        """Remove all steering hooks"""
        for name, hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = -1,
        **kwargs
    ) -> List[str]:
        """
        Generate text using vLLM
        
        Args:
            prompts: Input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            **kwargs: Additional sampling parameters
            
        Returns:
            List of generated texts
        """
        # Create sampling params
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Generate
        outputs = self.llm.generate(prompts, sampling_params)
        
        # Extract text
        generated_texts = [output.outputs[0].text for output in outputs]
        
        return generated_texts
    
    def set_steering(self, theta: float) -> None:
        """Set steering angle"""
        self.current_theta = theta
    
    def enable_steering(self) -> None:
        """Enable steering"""
        self.steering_enabled = True
    
    def disable_steering(self) -> None:
        """Disable steering"""
        self.steering_enabled = False
    
    def __del__(self):
        """Cleanup"""
        self._remove_hooks()
        if hasattr(self, 'llm'):
            del self.llm
        gc.collect()
        torch.cuda.empty_cache()