"""
Base classes for evaluation
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple


class BaseEvaluator(ABC):
    """Abstract base class for evaluators"""
    
    @abstractmethod
    def evaluate(
        self,
        prompts: List[str],
        responses: List[str],
        sampling_params: Optional[Dict[str, Any]] = None,
        return_labels: bool = False
    ) -> Union[float, Tuple[float, List[int]]]:
        """
        Evaluate model responses
        
        Args:
            prompts: List of prompts
            responses: List of responses
            sampling_params: Optional sampling parameters for model-based evaluators
            return_labels: Whether to return labels
            
        Returns:
            If return_labels is False: float score
            If return_labels is True: tuple of (float score, List[int] labels)
        """
        pass