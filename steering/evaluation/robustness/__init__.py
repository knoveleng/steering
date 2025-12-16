"""
Robustness Evaluation Module

This module provides tools for evaluating model robustness on standard benchmarks:
- tinyMMLU: Massive Multitask Language Understanding
- tinyGSM8k: Grade School Math
- tinyAI2_arc: AI2 Reasoning Challenge
- tinyWinogrande: Commonsense Reasoning
- tinyTruthfulQA: Truthfulness Evaluation

Usage:
    from steering.evaluation.robustness import (
        BenchmarkRegistry,
        RobustnessEvaluator,
    )
    
    # List available benchmarks
    benchmarks = BenchmarkRegistry.list_benchmarks()
    
    # Get a benchmark
    benchmark = BenchmarkRegistry.get("tinyMMLU")
    
    # Load data
    data = benchmark.load_data(max_samples=100)
    
    # Evaluate with LLM judge
    evaluator = RobustnessEvaluator({
        "benchmark": "tinyMMLU",
        "model": "Qwen/Qwen2.5-14B-Instruct"
    })
    accuracy, labels = evaluator.evaluate(prompts, responses)
"""

from .base import BaseBenchmark
from .registry import BenchmarkRegistry
from .evaluator import RobustnessEvaluator

# Import benchmarks to register them
from .benchmarks import (
    TinyMMLU,
    TinyGSM8k,
    TinyAI2Arc,
    TinyWinogrande,
    TinyTruthfulQA,
)

__all__ = [
    # Base classes
    "BaseBenchmark",
    "BenchmarkRegistry",
    # Evaluator
    "RobustnessEvaluator",
    # Benchmarks
    "TinyMMLU",
    "TinyGSM8k",
    "TinyAI2Arc",
    "TinyWinogrande",
    "TinyTruthfulQA",
]
