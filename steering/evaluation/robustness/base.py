"""
Base classes for robustness benchmarks.

This module provides abstract base classes for implementing robustness benchmarks
that can be used to evaluate model capabilities on various tasks.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseBenchmark(ABC):
    """Abstract base class for robustness benchmarks.
    
    All benchmark implementations should inherit from this class and implement
    the required abstract methods.
    
    Example:
        >>> @BenchmarkRegistry.register
        ... class MyBenchmark(BaseBenchmark):
        ...     @property
        ...     def name(self) -> str:
        ...         return "my_benchmark"
        ...     # ... implement other methods
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique benchmark name used for registration and identification.
        
        Returns:
            str: Benchmark name (e.g., 'tinyMMLU', 'tinyGSM8k')
        """
        pass
    
    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """HuggingFace dataset name.
        
        Returns:
            str: Full dataset name on HuggingFace (e.g., 'tinyBenchmarks/tinyMMLU')
        """
        pass
    
    @property
    @abstractmethod
    def default_split(self) -> str:
        """Default dataset split to use.
        
        Returns:
            str: Split name (e.g., 'test', 'validation')
        """
        pass
    
    @property
    @abstractmethod
    def answer_type(self) -> str:
        """Type of answer expected from the model.
        
        Returns:
            str: One of 'mcq' (multiple choice), 'numeric', or 'freeform'
        """
        pass
    
    @abstractmethod
    def load_data(
        self, 
        split: Optional[str] = None, 
        max_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Load benchmark data from HuggingFace.
        
        Args:
            split: Dataset split to load. If None, uses default_split.
            max_samples: Maximum number of samples to load. If None, loads all.
            
        Returns:
            List of sample dictionaries containing at minimum:
                - 'prompt': The formatted prompt for the model
                - 'ground_truth': The correct answer
        """
        pass
    
    @abstractmethod
    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """Format a raw sample into a prompt for the model.
        
        Args:
            sample: Raw sample from the dataset.
            
        Returns:
            Formatted prompt string.
        """
        pass
    
    @abstractmethod
    def extract_ground_truth(self, sample: Dict[str, Any]) -> str:
        """Extract the ground truth answer from a sample.
        
        Args:
            sample: Raw sample from the dataset.
            
        Returns:
            Ground truth answer string.
        """
        pass
    
    def get_system_prompt(self) -> str:
        """Get the system prompt to guide the LLM to generate answers directly.
        
        Returns:
            System prompt string for the model.
        """
        if self.answer_type == "mcq":
            return "You are a helpful assistant. Answer the multiple-choice question by providing your answer with a short explanation."
        elif self.answer_type == "numeric":
            return "You are a helpful assistant. Solve the math problem step by step and provide your final answer."
        else:  # freeform
            return "You are a helpful assistant. Answer the question directly."
    
    def get_judge_system_prompt(self) -> str:
        """Get the system prompt for the LLM judge.
        
        Can be overridden by subclasses for benchmark-specific prompts.
        
        Returns:
            System prompt string for the LLM judge.
        """
        if self.answer_type == "mcq":
            return """You are an answer checker. Your task is to determine if the model's answer matches the correct answer for a multiple-choice question.

The model's answer is CORRECT if it clearly selects the same option as the ground truth.
The model's answer is INCORRECT if it selects a different option or is unclear.

Respond with exactly one word: CORRECT or INCORRECT"""
        
        elif self.answer_type == "numeric":
            return """You are an answer checker. Your task is to determine if the model's numeric answer matches the correct answer.

The model's answer is CORRECT if the final numeric value matches the ground truth (ignoring formatting differences).
The model's answer is INCORRECT if the final numeric value is different.

Respond with exactly one word: CORRECT or INCORRECT"""
        
        else:  # freeform
            return """You are an answer checker. Your task is to determine if the model's answer is semantically equivalent to the correct answer.

The model's answer is CORRECT if it conveys the same meaning as the ground truth.
The model's answer is INCORRECT if it conveys a different meaning.

Respond with exactly one word: CORRECT or INCORRECT"""
    
    def format_judge_prompt(self, question: str, model_answer: str, ground_truth: str) -> str:
        """Format prompt for LLM judge to compare answers.
        
        Args:
            question: The original question/prompt.
            model_answer: The model's response.
            ground_truth: The correct answer.
            
        Returns:
            Formatted prompt for the LLM judge.
        """
        return f"""Question: {question}

Model's Answer: {model_answer}

Correct Answer: {ground_truth}

Is the model's answer correct?"""
