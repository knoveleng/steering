"""
Base classes for data management
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any


class BaseDataLoader(ABC):
    """Abstract base class for data loaders"""
    
    @abstractmethod
    def load(self, path: str) -> List[Dict[str, Any]]:
        """
        Load data from path
        
        Args:
            path: Path to data file
            
        Returns:
            List of data samples
        """
        pass
    
    @abstractmethod
    def prepare_prompts(self, data: List[Dict[str, Any]]) -> List[str]:
        """
        Extract prompts from data
        
        Args:
            data: List of data samples
            
        Returns:
            List of prompt strings
        """
        pass


class BaseDataset(ABC):
    """Abstract base class for datasets"""
    
    @abstractmethod
    def __len__(self) -> int:
        """Return dataset size"""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index"""
        pass
    
    @abstractmethod
    def get_batch(self, indices: List[int]) -> Dict[str, Any]:
        """Get batch of items"""
        pass