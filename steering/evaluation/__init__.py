"""Evaluation module"""

from typing import List, Dict, Any, Optional, Union, Tuple

from .base import BaseEvaluator
from .substring import SubstringEvaluator
from .llama_guard import LlamaGuardEvaluator
from .harmbench import HarmbenchEvaluator
from ..utils.logger import setup_logger

__all__ = [
    "BaseEvaluator",
    "SubstringEvaluator",
    "LlamaGuardEvaluator",
    "HarmbenchEvaluator",
    "EvaluationSuite",
]

class EvaluationSuite:
    """Evaluation suite for managing multiple evaluators"""

    __evaluator_names__ = [
        "substring",
        "llama_guard",
        "harmbench",
    ]
    
    def __init__(self):
        """Initialize evaluation suite."""
        # Setup logger
        self.logger = setup_logger(obj=self)

    def _get_evaluator(self, evaluator_name: str, evaluator_kwargs: Optional[Dict[str, Any]] = None) -> BaseEvaluator:
        """
        Get evaluator by name
        
        Args:
            evaluator_name: Name of the evaluator
            evaluator_kwargs: Optional kwargs to pass to evaluator constructor
            
        Returns:
            BaseEvaluator instance
            
        Raises:
            ValueError: If evaluator_name is not recognized
        """
        if evaluator_name == "substring":
            return SubstringEvaluator(evaluator_kwargs)
        elif evaluator_name == "llama_guard":
            return LlamaGuardEvaluator(evaluator_kwargs)
        elif evaluator_name == "harmbench":
            return HarmbenchEvaluator(evaluator_kwargs)
        else:
            raise ValueError(f"Unknown evaluator: {evaluator_name}. Available: {self.__evaluator_names__}")
    
    def get_available_evaluators(self) -> List[str]:
        """Get list of available evaluator names"""
        return self.__evaluator_names__.copy()