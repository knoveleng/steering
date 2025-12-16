"""
Robustness Evaluator using LLM Judge.

This module provides an evaluator that uses an LLM (default: Qwen2.5-14B-Instruct)
to compare model responses with ground truth answers for robustness benchmarks.
"""

import re
from typing import List, Dict, Any, Optional, Union, Tuple

from vllm import LLM, SamplingParams

from ..base import BaseEvaluator
from .base import BaseBenchmark
from .registry import BenchmarkRegistry
from ...utils.logger import setup_logger

logger = setup_logger(name="steering.evaluation.robustness.evaluator")


class RobustnessEvaluator(BaseEvaluator):
    """Evaluator for robustness benchmarks using LLM judge.
    
    This evaluator compares model responses with ground truth answers
    using an LLM as a judge. The judge determines if the model's answer
    is correct based on the benchmark's answer type.
    
    Example:
        >>> evaluator = RobustnessEvaluator({
        ...     "benchmark": "tinyMMLU",
        ...     "model": "Qwen/Qwen2.5-14B-Instruct"
        ... })
        >>> accuracy, labels = evaluator.evaluate(prompts, responses)
    """
    
    def __init__(self, evaluator_kwargs: Optional[Dict[str, Any]] = None):
        """Initialize the robustness evaluator.
        
        Args:
            evaluator_kwargs: Configuration dictionary with:
                - benchmark: Benchmark name (required)
                - split: Dataset split (default: benchmark's default)
                - model: Judge model (default: Qwen/Qwen2.5-14B-Instruct)
                - tensor_parallel_size: vLLM tensor parallel (default: 1)
                - gpu_memory_utilization: GPU memory (default: 0.9)
                - max_model_len: Max model length (default: 4096)
        """
        self.logger = setup_logger(obj=self)
        
        evaluator_kwargs = evaluator_kwargs or {}
        
        # Get benchmark
        benchmark_name = evaluator_kwargs.get("benchmark")
        if not benchmark_name:
            raise ValueError("'benchmark' is required in evaluator_kwargs")
        
        self.benchmark: BaseBenchmark = BenchmarkRegistry.get(benchmark_name)
        self.split = evaluator_kwargs.get("split", self.benchmark.default_split)
        
        # Load benchmark data for ground truth
        self._data_cache: Optional[List[Dict[str, Any]]] = None
        
        # Initialize LLM judge
        self.llm = LLM(**self._get_model_kwargs(evaluator_kwargs))
        
        self.logger.info(f"Initialized RobustnessEvaluator for {benchmark_name}")
    
    def _get_model_kwargs(self, evaluator_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Get model kwargs for vLLM."""
        default_kwargs = {
            "model": "Qwen/Qwen2.5-14B-Instruct",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 8192,
            "trust_remote_code": True,
        }
        
        # Extract model-specific kwargs
        model_keys = ["model", "tensor_parallel_size", "gpu_memory_utilization", 
                      "max_model_len", "trust_remote_code", "enforce_eager"]
        
        for key in model_keys:
            if key in evaluator_kwargs:
                default_kwargs[key] = evaluator_kwargs[key]
        
        self.logger.info(f"Model kwargs: {default_kwargs}")
        return default_kwargs
    
    def _get_sampling_params(self, sampling_params: Optional[Dict[str, Any]]) -> SamplingParams:
        """Get sampling params."""
        default_sampling_params = {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 16,  # Short response: just CORRECT or INCORRECT
        }
        
        if sampling_params:
            default_sampling_params.update(sampling_params)
        
        return SamplingParams(**default_sampling_params)
    
    def _load_data(self, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load benchmark data (cached)."""
        if self._data_cache is None:
            self._data_cache = self.benchmark.load_data(self.split, max_samples)
        return self._data_cache
    
    def _build_judge_messages(
        self, 
        prompt: str, 
        response: str, 
        ground_truth: str
    ) -> List[Dict[str, str]]:
        """Build chat messages for the LLM judge."""
        system_prompt = self.benchmark.get_judge_system_prompt()
        user_prompt = self.benchmark.format_judge_prompt(prompt, response, ground_truth)
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    
    def _parse_verdict(self, output: str) -> int:
        """Parse the verdict from LLM judge output.
        
        Args:
            output: Raw model output text
            
        Returns:
            1 if CORRECT, 0 if INCORRECT
        """
        output_upper = output.upper().strip()
        
        # Look for CORRECT/INCORRECT
        if "CORRECT" in output_upper:
            # Make sure it's not "INCORRECT"
            if "INCORRECT" in output_upper:
                return 0
            return 1
        
        return 0
    
    def evaluate(
        self,
        prompts: List[str],
        responses: List[str],
        sampling_params: Optional[Dict[str, Any]] = None,
        return_labels: bool = True,
        ground_truths: Optional[List[str]] = None,
    ) -> Union[float, Tuple[float, List[int]]]:
        """Evaluate model responses against ground truth.
        
        Args:
            prompts: List of prompts that were given to the model
            responses: List of model responses to evaluate
            sampling_params: Optional sampling parameters for the judge
            return_labels: Whether to return per-sample labels
            ground_truths: Optional list of ground truth answers. If not provided,
                          will be loaded from the benchmark data.
            
        Returns:
            If return_labels is False: accuracy score (float)
            If return_labels is True: tuple of (accuracy, List[int] labels)
                where labels are 1 for correct, 0 for incorrect
        """
        if not prompts or not responses:
            score = 0.0
            labels = []
            return (score, labels) if return_labels else score
        
        # Get ground truths if not provided
        if ground_truths is None:
            data = self._load_data(max_samples=len(prompts))
            ground_truths = [d["ground_truth"] for d in data]
        
        if len(ground_truths) != len(responses):
            self.logger.warning(
                f"Ground truths ({len(ground_truths)}) != responses ({len(responses)}). "
                "Using min length."
            )
            min_len = min(len(ground_truths), len(responses), len(prompts))
            prompts = prompts[:min_len]
            responses = responses[:min_len]
            ground_truths = ground_truths[:min_len]
        
        # Build judge prompts
        all_messages = [
            self._build_judge_messages(prompt, response, gt)
            for prompt, response, gt in zip(prompts, responses, ground_truths)
        ]
        
        # Apply chat template
        tokenizer = self.llm.get_tokenizer()
        judge_prompts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            for messages in all_messages
        ]
        
        # Generate judgments
        self.logger.info(f"Judging {len(judge_prompts)} responses...")
        outputs = self.llm.generate(judge_prompts, self._get_sampling_params(sampling_params))
        generated_texts = [output.outputs[0].text for output in outputs]
        
        # Parse verdicts
        labels = [self._parse_verdict(text) for text in generated_texts]
        
        # Calculate accuracy
        accuracy = sum(labels) / len(labels) if labels else 0.0
        
        self.logger.info(f"Accuracy: {accuracy:.4f} ({sum(labels)}/{len(labels)})")
        
        if return_labels:
            return accuracy, labels
        else:
            return accuracy
