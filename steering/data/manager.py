"""
Data management implementation
"""

import json
import random
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

from .base import BaseDataLoader, BaseDataset
from ..utils.logger import setup_logger


class JSONDataLoader(BaseDataLoader):
    """Load data from JSON files"""
    
    def load(self, path: str) -> List[Dict[str, Any]]:
        """Load JSON data"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, dict):
            if 'data' in data:
                return data['data']
            elif 'examples' in data:
                return data['examples']
        
        return data
    
    def prepare_prompts(self, data: List[Dict[str, Any]]) -> List[str]:
        """Extract prompts from data"""
        prompts = []
        
        for item in data:
            # Try common field names
            if 'prompt' in item:
                prompts.append(item['prompt'])
            elif 'instruction' in item:
                prompts.append(item['instruction'])
            elif 'text' in item:
                prompts.append(item['text'])
            elif 'input' in item:
                prompts.append(item['input'])
            elif 'goal' in item:
                prompts.append(item['goal'])
            else:
                # Use first string value found
                for value in item.values():
                    if isinstance(value, str):
                        prompts.append(value)
                        break
        
        return prompts


class DataManager:
    """
    Manage datasets for calibration and evaluation
    """
    
    def __init__(
        self, 
        loader: Optional[BaseDataLoader] = None,
        tokenizer: Optional[Any] = None,
        use_chat_template: bool = False,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize data manager
        
        Args:
            loader: Data loader instance (default: JSONDataLoader)
            tokenizer: Optional tokenizer for chat template
            use_chat_template: Whether to apply chat template to prompts
            system_prompt: System prompt for chat template
        """
        self.loader = loader or JSONDataLoader()
        self.tokenizer = tokenizer
        self.use_chat_template = use_chat_template
        self.system_prompt = system_prompt
        self.harmful_data = None
        self.harmless_data = None

        # Setup logger
        self.logger = setup_logger(obj=self)
    
    def format_prompt(
        self, 
        text: str,
        system_prompt: Optional[str] = None,
        add_generation_prompt: bool = True
    ) -> str:
        """
        Format prompt with chat template if enabled
        
        Args:
            text: Raw prompt text
            system_prompt: Optional system prompt override
            add_generation_prompt: Add assistant prefix for generation
            
        Returns:
            Formatted prompt
        """
        if not self.use_chat_template or self.tokenizer is None:
            return text
        
        # Check if tokenizer has chat template
        if not hasattr(self.tokenizer, 'chat_template') or not self.tokenizer.chat_template:
            return text
        
        # Use provided or default system prompt
        sys_prompt = system_prompt or self.system_prompt
        
        # Create message format
        messages = []
        
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        
        messages.append({"role": "user", "content": text})
        
        # Apply chat template
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt
            )
            return formatted
        except Exception as e:
            self.logger.warning(f"Failed to apply chat template: {e}")
            return text
    
    def load_datasets(
        self,
        harmful_path: str,
        harmless_path: str,
        harmful_samples: Optional[int] = None,
        harmless_samples: Optional[int] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Load harmful and harmless datasets
        
        Args:
            harmful_path: Path to harmful dataset
            harmless_path: Path to harmless dataset
            harmful_samples: Number of harmful samples (None = all)
            harmless_samples: Number of harmless samples (None = all)
            
        Returns:
            Tuple of (harmful_prompts, harmless_prompts)
        """
        # Load data
        harmful_raw = self.loader.load(harmful_path)
        harmless_raw = self.loader.load(harmless_path)
        
        # Sample if requested
        if harmful_samples and len(harmful_raw) > harmful_samples:
            harmful_raw = random.sample(harmful_raw, harmful_samples)
        
        if harmless_samples and len(harmless_raw) > harmless_samples:
            harmless_raw = random.sample(harmless_raw, harmless_samples)
        
        # Extract prompts
        harmful_prompts = self.loader.prepare_prompts(harmful_raw)
        harmless_prompts = self.loader.prepare_prompts(harmless_raw)
        
        # Apply chat template if enabled
        if self.use_chat_template:
            self.logger.info("  Applying chat template to prompts...")
            harmful_prompts = [self.format_prompt(p) for p in harmful_prompts]
            harmless_prompts = [self.format_prompt(p) for p in harmless_prompts]
        
        self.harmful_data = harmful_prompts
        self.harmless_data = harmless_prompts
        
        return harmful_prompts, harmless_prompts
    
    def split_data(
        self,
        data: List[str],
        train_ratio: float = 0.8
    ) -> Tuple[List[str], List[str]]:
        """
        Split data into train and validation sets
        
        Args:
            data: List of data samples
            train_ratio: Ratio of training data
            
        Returns:
            Tuple of (train_data, val_data)
        """
        n_train = int(len(data) * train_ratio)
        
        shuffled = data.copy()
        random.shuffle(shuffled)
        
        return shuffled[:n_train], shuffled[n_train:]


class SteeringDataset(BaseDataset):
    """Dataset for steering calibration"""
    
    def __init__(
        self, 
        prompts: List[str], 
        labels: Optional[List[int]] = None,
        tokenizer: Optional[Any] = None,
        use_chat_template: bool = False,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize dataset
        
        Args:
            prompts: List of prompt strings
            labels: Optional labels (0=harmless, 1=harmful)
            tokenizer: Optional tokenizer for chat template
            use_chat_template: Whether to apply chat template
            system_prompt: System prompt for chat template
        """
        self.tokenizer = tokenizer
        self.use_chat_template = use_chat_template
        self.system_prompt = system_prompt
        
        # Apply chat template if enabled
        if use_chat_template and tokenizer is not None:
            prompts = [self._format_prompt(p) for p in prompts]
        
        self.prompts = prompts
        self.labels = labels if labels is not None else [0] * len(prompts)
    
    def _format_prompt(self, text: str) -> str:
        """Format prompt with chat template"""
        if not hasattr(self.tokenizer, 'chat_template') or not self.tokenizer.chat_template:
            return text
        
        messages = []
        
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        messages.append({"role": "user", "content": text})
        
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            return text
    
    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            'prompt': self.prompts[idx],
            'label': self.labels[idx]
        }
    
    def get_batch(self, indices: List[int]) -> Dict[str, Any]:
        return {
            'prompts': [self.prompts[i] for i in indices],
            'labels': [self.labels[i] for i in indices]
        }