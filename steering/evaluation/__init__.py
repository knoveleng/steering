"""Evaluation module"""

from typing import List, Dict, Any, Optional, Union, Tuple

from .base import BaseEvaluator
from .substring import SubstringEvaluator
from .llama_guard import LlamaGuardEvaluator
from .harmbench import HarmbenchEvaluator
from .quality import (
    NgramRepetitionEvaluator,
    LanguageConsistencyEvaluator,
    CompressionRatioEvaluator,
)
from .qwen3guard import Qwen3GuardEvaluator
from .polyguard import PolyGuardEvaluator
from .llm_judge import LLMJudgeEvaluator
from .robustness import (
    BaseBenchmark,
    BenchmarkRegistry,
    RobustnessEvaluator,
)
from ..utils.logger import setup_logger

__all__ = [
    "BaseEvaluator",
    "SubstringEvaluator",
    "LlamaGuardEvaluator",
    "HarmbenchEvaluator",
    "NgramRepetitionEvaluator",
    "LanguageConsistencyEvaluator",
    "CompressionRatioEvaluator",
    "Qwen3GuardEvaluator",
    "PolyGuardEvaluator",
    "LLMJudgeEvaluator",
    "EvaluationSuite",
    # Robustness evaluation
    "BaseBenchmark",
    "BenchmarkRegistry",
    "RobustnessEvaluator",
]


class EvaluationSuite:
    """Evaluation suite for managing multiple evaluators"""

    __evaluator_names__ = [
        "substring",
        "llama_guard",
        "harmbench",
        "ngram_repetition",
        "language_consistency",
        "compression_ratio",
        "qwen3guard",
        "polyguard",
        "llm_judge",
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
        elif evaluator_name == "ngram_repetition":
            return NgramRepetitionEvaluator(evaluator_kwargs)
        elif evaluator_name == "language_consistency":
            return LanguageConsistencyEvaluator(evaluator_kwargs)
        elif evaluator_name == "compression_ratio":
            return CompressionRatioEvaluator(evaluator_kwargs)
        elif evaluator_name == "qwen3guard":
            return Qwen3GuardEvaluator(evaluator_kwargs)
        elif evaluator_name == "polyguard":
            return PolyGuardEvaluator(evaluator_kwargs)
        elif evaluator_name == "llm_judge":
            return LLMJudgeEvaluator(evaluator_kwargs)
        else:
            raise ValueError(f"Unknown evaluator: {evaluator_name}. Available: {self.__evaluator_names__}")
    
    def get_available_evaluators(self) -> List[str]:
        """Get list of available evaluator names"""
        return self.__evaluator_names__.copy()