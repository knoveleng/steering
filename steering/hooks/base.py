"""
Base classes for model hook management
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any
import torch


class BaseHookManager(ABC):
    """Abstract base class for hook managers"""
    
    @abstractmethod
    def register_hooks(
        self,
        layer_names: List[str],
        steering_params: Optional[Any] = None
    ) -> None:
        """
        Register hooks on specified layers
        
        Args:
            layer_names: List of layer names to hook
            steering_params: Parameters for steering operation
        """
        pass
    
    @abstractmethod
    def remove_hooks(self) -> None:
        """Remove all registered hooks"""
        pass
    
    @abstractmethod
    def enable_steering(self) -> None:
        """Enable steering in hooks"""
        pass
    
    @abstractmethod
    def disable_steering(self) -> None:
        """Disable steering in hooks"""
        pass