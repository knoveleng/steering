"""
Base classes for evaluation
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseEvaluator(ABC):
    """Abstract base class for evaluators"""
    
    @abstractmethod
    def evaluate(
        self,
        outputs: List[str],
        references: List[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate model outputs
        
        Args:
            outputs: Generated outputs
            references: Optional reference outputs
            
        Returns:
            Dictionary of metric scores
        """
        pass