"""
Benchmark implementations for tinyBenchmarks datasets.

This module contains implementations for the following benchmarks:
- tinyMMLU: Multiple-choice questions across various subjects
- tinyGSM8k: Grade school math problems
- tinyAI2_arc: AI2 Reasoning Challenge (science questions)
- tinyWinogrande: Commonsense reasoning (sentence completion)
- tinyTruthfulQA: Truthfulness evaluation
"""

import re
from typing import List, Dict, Any, Optional

from datasets import load_dataset

from .base import BaseBenchmark
from .registry import BenchmarkRegistry
from ...utils.logger import setup_logger

logger = setup_logger(name="steering.evaluation.robustness.benchmarks")


@BenchmarkRegistry.register
class TinyMMLU(BaseBenchmark):
    """MMLU (Massive Multitask Language Understanding) benchmark.
    
    Multiple-choice questions across 57 subjects including STEM, humanities,
    social sciences, and more.
    
    Answer type: MCQ (A/B/C/D)
    """
    
    @property
    def name(self) -> str:
        return "tinyMMLU"
    
    @property
    def dataset_name(self) -> str:
        return "tinyBenchmarks/tinyMMLU"
    
    @property
    def default_split(self) -> str:
        return "test"
    
    @property
    def answer_type(self) -> str:
        return "mcq"
    
    def load_data(
        self, 
        split: Optional[str] = None, 
        max_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        split = split or self.default_split
        logger.info(f"Loading {self.dataset_name} ({split})...")
        
        ds = load_dataset(self.dataset_name, split=split)
        
        samples = []
        for i, item in enumerate(ds):
            if max_samples is not None and i >= max_samples:
                break
            samples.append({
                "prompt": self.format_prompt(item),
                "ground_truth": self.extract_ground_truth(item),
                "raw": item,
            })
        
        logger.info(f"Loaded {len(samples)} samples from {self.name}")
        return samples
    
    def format_prompt(self, sample: Dict[str, Any]) -> str:
        question = sample["question"]
        choices = sample["choices"]
        
        # Build the question with choices
        prompt = f"{question}\n"
        for i, choice in enumerate(choices):
            label = chr(65 + i)  # A, B, C, D
            prompt += f"{label}. {choice}\n"
        prompt += "Answer:"
        
        return prompt
    
    def extract_ground_truth(self, sample: Dict[str, Any]) -> str:
        answer_idx = sample["answer"]
        return ['A', 'B', 'C', 'D'][answer_idx]


@BenchmarkRegistry.register
class TinyGSM8k(BaseBenchmark):
    """GSM8K (Grade School Math 8K) benchmark.
    
    Grade school math word problems requiring multi-step reasoning.
    
    Answer type: Numeric
    """
    
    @property
    def name(self) -> str:
        return "tinyGSM8k"
    
    @property
    def dataset_name(self) -> str:
        return "tinyBenchmarks/tinyGSM8k"
    
    @property
    def default_split(self) -> str:
        return "test"
    
    @property
    def answer_type(self) -> str:
        return "numeric"
    
    def load_data(
        self, 
        split: Optional[str] = None, 
        max_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        split = split or self.default_split
        logger.info(f"Loading {self.dataset_name} ({split})...")
        
        ds = load_dataset(self.dataset_name, split=split)
        
        samples = []
        for i, item in enumerate(ds):
            if max_samples is not None and i >= max_samples:
                break
            samples.append({
                "prompt": self.format_prompt(item),
                "ground_truth": self.extract_ground_truth(item),
                "raw": item,
            })
        
        logger.info(f"Loaded {len(samples)} samples from {self.name}")
        return samples
    
    def format_prompt(self, sample: Dict[str, Any]) -> str:
        return f"Question: {sample['question']}\nAnswer:"
    
    def extract_ground_truth(self, sample: Dict[str, Any]) -> str:
        # GSM8k answers end with "#### <number>"
        answer_text = sample["answer"]
        match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", answer_text)
        if match:
            return match.group(1)
        return answer_text.strip()


@BenchmarkRegistry.register
class TinyAI2Arc(BaseBenchmark):
    """AI2 Reasoning Challenge (ARC) benchmark.
    
    Science questions from 3rd to 9th grade exams.
    
    Answer type: MCQ (A/B/C/D)
    """
    
    @property
    def name(self) -> str:
        return "tinyAI2_arc"
    
    @property
    def dataset_name(self) -> str:
        return "tinyBenchmarks/tinyAI2_arc"
    
    @property
    def default_split(self) -> str:
        return "test"
    
    @property
    def answer_type(self) -> str:
        return "mcq"
    
    def load_data(
        self, 
        split: Optional[str] = None, 
        max_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        split = split or self.default_split
        logger.info(f"Loading {self.dataset_name} ({split})...")
        
        ds = load_dataset(self.dataset_name, split=split)
        
        samples = []
        for i, item in enumerate(ds):
            if max_samples is not None and i >= max_samples:
                break
            samples.append({
                "prompt": self.format_prompt(item),
                "ground_truth": self.extract_ground_truth(item),
                "raw": item,
            })
        
        logger.info(f"Loaded {len(samples)} samples from {self.name}")
        return samples
    
    def format_prompt(self, sample: Dict[str, Any]) -> str:
        question = sample["question"]
        choices = sample["choices"]
        
        prompt = f"Question: {question}\n"
        for label, text in zip(choices["label"], choices["text"]):
            prompt += f"{label}. {text}\n"
        prompt += "Answer:"
        
        return prompt
    
    def extract_ground_truth(self, sample: Dict[str, Any]) -> str:
        return sample["answerKey"]


@BenchmarkRegistry.register
class TinyWinogrande(BaseBenchmark):
    """Winogrande benchmark.
    
    Commonsense reasoning through sentence completion.
    
    Answer type: Sentence completion
    """
    
    @property
    def name(self) -> str:
        return "tinyWinogrande"
    
    @property
    def dataset_name(self) -> str:
        return "tinyBenchmarks/tinyWinogrande"
    
    @property
    def default_split(self) -> str:
        return "validation"
    
    @property
    def answer_type(self) -> str:
        return "mcq"
    
    def load_data(
        self, 
        split: Optional[str] = None, 
        max_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        split = split or self.default_split
        logger.info(f"Loading {self.dataset_name} ({split})...")
        
        ds = load_dataset(self.dataset_name, split=split)
        
        samples = []
        for i, item in enumerate(ds):
            if max_samples is not None and i >= max_samples:
                break
            # Skip samples without answers (test set)
            if not item.get("answer"):
                continue
            samples.append({
                "prompt": self.format_prompt(item),
                "ground_truth": self.extract_ground_truth(item),
                "raw": item,
            })
        
        logger.info(f"Loaded {len(samples)} samples from {self.name}")
        return samples
    
    def format_prompt(self, sample: Dict[str, Any]) -> str:
        sentence = sample["sentence"]
        option1 = sample["option1"]
        option2 = sample["option2"]
        
        return f"Complete the sentence by filling in the blank:\n{sentence}\nOptions: {option1} / {option2}\nAnswer:"
    
    def extract_ground_truth(self, sample: Dict[str, Any]) -> str:
        # Return the actual option text, not "1" or "2"
        answer = sample["answer"]
        if answer == "1":
            return sample["option1"]
        else:
            return sample["option2"]
    
    def get_system_prompt(self) -> str:
        return "You are a helpful assistant. Complete the sentence by choosing the correct option."
    
    def get_judge_system_prompt(self) -> str:
        return """You are an answer checker. Your task is to determine if the model's answer matches the correct answer for a sentence completion question.

The model's answer is CORRECT if it contains or selects the same word/phrase as the ground truth option.
The model's answer is INCORRECT if it selects a different option.

Respond with exactly one word: CORRECT or INCORRECT"""


@BenchmarkRegistry.register
class TinyTruthfulQA(BaseBenchmark):
    """TruthfulQA benchmark.
    
    Questions designed to test truthfulness and avoid imitative falsehoods.
    
    Answer type: MCQ
    """
    
    @property
    def name(self) -> str:
        return "tinyTruthfulQA"
    
    @property
    def dataset_name(self) -> str:
        return "tinyBenchmarks/tinyTruthfulQA"
    
    @property
    def default_split(self) -> str:
        return "validation"
    
    @property
    def answer_type(self) -> str:
        return "mcq"
    
    def load_data(
        self, 
        split: Optional[str] = None, 
        max_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        split = split or self.default_split
        logger.info(f"Loading {self.dataset_name} ({split})...")
        
        ds = load_dataset(self.dataset_name, split=split)
        
        samples = []
        for i, item in enumerate(ds):
            if max_samples is not None and i >= max_samples:
                break
            samples.append({
                "prompt": self.format_prompt(item),
                "ground_truth": self.extract_ground_truth(item),
                "raw": item,
            })
        
        logger.info(f"Loaded {len(samples)} samples from {self.name}")
        return samples
    
    def format_prompt(self, sample: Dict[str, Any]) -> str:
        question = sample["question"]
        mc1_targets = sample["mc1_targets"]
        choices = mc1_targets["choices"]
        
        # Build question with choices
        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65 + i)}. {choice}\n"
        prompt += "Answer:"
        
        return prompt
    
    def extract_ground_truth(self, sample: Dict[str, Any]) -> str:
        # mc1_targets has a 'labels' list where 1 indicates correct answer
        mc1_targets = sample["mc1_targets"]
        labels = mc1_targets["labels"]
        
        for i, label in enumerate(labels):
            if label == 1:
                return chr(65 + i)  # A, B, C, D, ...
        
        return "A"  # Default to A if no label found
