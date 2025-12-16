"""
Quality metrics for detecting weird/collapsed responses

This module provides evaluators for:
- N-gram Repetition Rate: Measures percentage of n-grams that repeat
- Language Consistency: Detects foreign language contamination
- Compression Ratio: Pattern-agnostic collapse detection
"""

import zlib
import unicodedata
from collections import Counter
from typing import List, Dict, Optional, Any, Union, Tuple

from .base import BaseEvaluator
from ..utils.logger import setup_logger


class NgramRepetitionEvaluator(BaseEvaluator):
    """lib
    Evaluate response quality by measuring n-gram repetition rate.
    
    Higher repetition rate indicates weird/collapsed responses.
    Rep-n = (total n-grams - unique n-grams) / total n-grams
    """
    
    def __init__(self, evaluator_kwargs: Optional[Dict[str, Any]] = None):
        """
        Initialize n-gram repetition evaluator.
        
        Args:
            evaluator_kwargs: Optional kwargs with 'n' for n-gram size (default: 4)
        """
        self.logger = setup_logger(obj=self)
        
        kwargs = evaluator_kwargs or {}
        self.n = kwargs.get("n", 4)  # Default to 4-grams
    
    def _get_ngrams(self, text: str) -> List[Tuple[str, ...]]:
        """Extract n-grams from text."""
        words = text.split()
        if len(words) < self.n:
            return []
        return [tuple(words[i:i + self.n]) for i in range(len(words) - self.n + 1)]
    
    def _compute_repetition_rate(self, text: str) -> float:
        """
        Compute repetition rate for a single text.
        
        Returns:
            Float between 0.0 (no repetition) and 1.0 (all n-grams repeated)
        """
        ngrams = self._get_ngrams(text)
        if not ngrams:
            return 0.0
        
        total = len(ngrams)
        unique = len(set(ngrams))
        
        # Repetition rate = proportion of repeated n-grams
        return (total - unique) / total
    
    def evaluate(
        self,
        prompts: List[str],
        responses: List[str],
        sampling_params: Optional[Dict[str, Any]] = None,
        return_labels: bool = False
    ) -> Union[float, Tuple[float, List[float]]]:
        """
        Compute n-gram repetition rate for responses.
        
        Args:
            prompts: List of prompts (not used)
            responses: List of generated texts
            sampling_params: Not used
            return_labels: Whether to return individual scores
            
        Returns:
            Average repetition rate across all responses.
            If return_labels is True, also returns per-response rates.
        """
        rates = [self._compute_repetition_rate(response) for response in responses]
        avg_rate = sum(rates) / len(rates) if rates else 0.0
        
        if return_labels:
            return avg_rate, rates
        return avg_rate


class LanguageConsistencyEvaluator(BaseEvaluator):
    """
    Evaluate response quality by detecting foreign language contamination.
    
    Uses character-level Unicode script analysis to detect non-Latin
    character contamination (e.g., CJK, Arabic, Cyrillic characters
    appearing in English text).
    
    Score: 1.0 = fully consistent (all Latin/common), 0.0 = completely foreign
    """
    
    # Unicode script categories considered "expected" for English text
    EXPECTED_CATEGORIES = {
        'LATIN',
        'COMMON',  # Punctuation, digits, whitespace
        'INHERITED',  # Combining marks
    }
    
    def __init__(self, evaluator_kwargs: Optional[Dict[str, Any]] = None):
        """Initialize language consistency evaluator."""
        self.logger = setup_logger(obj=self)
    
    def _get_script(self, char: str) -> str:
        """Get Unicode script name for a character."""
        try:
            name = unicodedata.name(char, '')
            # Extract script from character name (first word is usually the script)
            if name:
                script = name.split()[0]
                return script
        except ValueError:
            pass
        return 'UNKNOWN'
    
    def _compute_consistency_score(self, text: str) -> float:
        """
        Compute language consistency score for a single text.
        
        Returns:
            Float between 0.0 (completely foreign) and 1.0 (fully consistent)
        """
        if not text:
            return 1.0
        
        # Count characters by script category
        total_chars = 0
        expected_chars = 0
        
        for char in text:
            if char.isspace():
                continue
            total_chars += 1
            
            script = self._get_script(char)
            if script in self.EXPECTED_CATEGORIES or char.isascii():
                expected_chars += 1
        
        if total_chars == 0:
            return 1.0
        
        return expected_chars / total_chars
    
    def evaluate(
        self,
        prompts: List[str],
        responses: List[str],
        sampling_params: Optional[Dict[str, Any]] = None,
        return_labels: bool = False
    ) -> Union[float, Tuple[float, List[float]]]:
        """
        Compute language consistency score for responses.
        
        Args:
            prompts: List of prompts (not used)
            responses: List of generated texts
            sampling_params: Not used
            return_labels: Whether to return individual scores
            
        Returns:
            Average consistency score across all responses.
            If return_labels is True, also returns per-response scores.
        """
        scores = [self._compute_consistency_score(response) for response in responses]
        avg_score = sum(scores) / len(scores) if scores else 1.0
        
        if return_labels:
            return avg_score, scores
        return avg_score


class CompressionRatioEvaluator(BaseEvaluator):
    """
    Evaluate response quality using compression ratio for pattern-agnostic collapse detection.
    
    Highly repetitive or patterned text compresses very well, resulting in
    a low compression ratio. This metric detects collapsed outputs without
    relying on specific patterns.
    
    Returns: Normalized compression ratio (compressed_size / original_size)
    Lower values indicate more repetitive/collapsed text.
    """
    
    def __init__(self, evaluator_kwargs: Optional[Dict[str, Any]] = None):
        """Initialize compression ratio evaluator."""
        self.logger = setup_logger(obj=self)
    
    def _compute_compression_ratio(self, text: str) -> float:
        """
        Compute compression ratio for a single text.
        
        Returns:
            Float representing compressed_size / original_size
            Lower values indicate more repetitive text.
        """
        if not text:
            return 1.0
        
        original = text.encode('utf-8')
        compressed = zlib.compress(original, level=9)
        
        original_size = len(original)
        compressed_size = len(compressed)
        
        if original_size == 0:
            return 1.0
        
        return compressed_size / original_size
    
    def evaluate(
        self,
        prompts: List[str],
        responses: List[str],
        sampling_params: Optional[Dict[str, Any]] = None,
        return_labels: bool = False
    ) -> Union[float, Tuple[float, List[float]]]:
        """
        Compute compression ratio for responses.
        
        Args:
            prompts: List of prompts (not used)
            responses: List of generated texts
            sampling_params: Not used
            return_labels: Whether to return individual ratios
            
        Returns:
            Average compression ratio across all responses.
            If return_labels is True, also returns per-response ratios.
        """
        ratios = [self._compute_compression_ratio(response) for response in responses]
        avg_ratio = sum(ratios) / len(ratios) if ratios else 1.0
        
        if return_labels:
            return avg_ratio, ratios
        return avg_ratio
