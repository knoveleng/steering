"""
Base classes for model serving
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseModelServer(ABC):
    """Abstract base class for model servers"""
    
    @abstractmethod
    def generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """
        Generate text for prompts
        
        Args:
            prompts: Input prompts
            **kwargs: Generation parameters
            
        Returns:
            List of generated texts
        """
        pass
    
    @abstractmethod
    def set_steering(self, theta: float) -> None:
        """
        Set steering angle
        
        Args:
            theta: Steering angle in degrees
        """
        pass
    
    @abstractmethod
    def enable_steering(self) -> None:
        """Enable steering"""
        pass
    
    @abstractmethod
    def disable_steering(self) -> None:
        """Disable steering"""
        pass