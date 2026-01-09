#!/usr/bin/env python3
"""
Step 1: Generate responses for robustness benchmarks.

This script generates model responses for robustness benchmark datasets.
It can be used with or without steering (calibration).

Usage:
    # With steering
    python examples/generate_robustness.py \
        --calibration artifacts/calibration_Qwen2.5-3B-Instruct_basic \
        --benchmark tinyMMLU \
        --output-dir logs/robustness/basic/Qwen2.5-3B-Instruct \
        --degree-start 0 --degree-end 360 --degree-step 10

    # Without steering (baseline)
    python examples/generate_robustness.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --benchmark tinyMMLU \
        --output-dir logs/robustness/baseline/Qwen2.5-3B-Instruct
    
    # With few-shot prompts
    python examples/generate_robustness.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --benchmark tinyMMLU \
        --use-fewshot \
        --output-dir logs/robustness/baseline/Qwen2.5-3B-Instruct
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional

from vllm import LLM, SamplingParams

from steering import SteeringLLM
from steering.evaluation.robustness import BenchmarkRegistry
from steering.utils.logger import setup_logger

# Enable insecure serialization for vLLM v0.12+
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'

logger = setup_logger("steering.example.generate_robustness")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate responses for robustness benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model options (mutually exclusive: calibration OR model)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--calibration",
        type=str,
        help="Path to calibration session for steering"
    )
    model_group.add_argument(
        "--model",
        type=str,
        help="Model name or path (for baseline without steering)"
    )
    
    # Benchmark options
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=BenchmarkRegistry.list_benchmarks(),
        help="Benchmark to evaluate"
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split (default: benchmark's default split)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to generate"
    )
    
    # Prompt options
    parser.add_argument(
        "--no-system-prompt",
        action="store_true",
        default=False,
        help="Disable system prompt (default: use benchmark's system prompt)"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results"
    )
    
    # Generation options
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature"
    )
    
    # Steering options (only used with --calibration)
    parser.add_argument(
        "--degree-start",
        type=int,
        default=0,
        help="Start degree for steering range"
    )
    parser.add_argument(
        "--degree-end",
        type=int,
        default=0,
        help="End degree for steering range (set same as start for single degree)"
    )
    parser.add_argument(
        "--degree-step",
        type=int,
        default=10,
        help="Step size for degree range"
    )
    
    # vLLM options
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="GPU memory utilization"
    )
    
    # Override options
    parser.add_argument(
        "--mode",
        type=str,
        choices=['standard', 'adaptive', 'selective', 'addition', 'ablation'],
        help="Override steering mode from calibration"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        help="Override model ID from calibration"
    )
    
    return parser.parse_args()


def format_prompts_with_chat_template(
    prompts: List[str],
    tokenizer,
    system_prompt: Optional[str] = None,
) -> List[str]:
    """Format prompts using chat template if available.
    
    If the model doesn't support chat template, prepends the system prompt
    as an instruction before the question.
    """
    # Check if tokenizer has chat template
    has_chat_template = (
        hasattr(tokenizer, 'apply_chat_template') and 
        tokenizer.chat_template is not None
    )
    
    if not has_chat_template:
        logger.warning("Tokenizer has no chat template. Prepending system prompt as instruction.")
        # Prepend system prompt as instruction
        if system_prompt:
            return [f"{system_prompt}\n\n{prompt}" for prompt in prompts]
        return prompts
    
    formatted_prompts = []
    for prompt in prompts:
        # First try with system role
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            formatted = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            formatted_prompts.append(formatted)
        except Exception as e:
            # If system role fails, try prepending system prompt to user message
            logger.warning(f"Chat template with system role failed: {e}. Trying without system role.")
            try:
                # Prepend system prompt to user message
                user_content = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                messages_no_system = [{"role": "user", "content": user_content}]
                formatted = tokenizer.apply_chat_template(
                    messages_no_system,
                    tokenize=False,
                    add_generation_prompt=True
                )
                formatted_prompts.append(formatted)
            except Exception as e2:
                logger.warning(f"Chat template failed completely: {e2}. Using raw prompt with instruction.")
                # Fallback: prepend system prompt as instruction
                if system_prompt:
                    formatted_prompts.append(f"{system_prompt}\n\n{prompt}")
                else:
                    formatted_prompts.append(prompt)
    
    return formatted_prompts


def generate_with_steering(args: argparse.Namespace) -> None:
    """Generate responses using SteeringLLM."""
    from steering.utils import load_calibration
    
    logger.info("=" * 60)
    logger.info("Generate Robustness with Steering")
    logger.info("=" * 60)
    
    # Load calibration
    logger.info(f"Loading calibration from {args.calibration}...")
    calibration = load_calibration(args.calibration)
    
    # Apply overrides
    if args.model_id:
        calibration['model_name'] = args.model_id
        logger.info(f"  Model override: {args.model_id}")
    if args.mode:
        calibration['mode'] = args.mode
        logger.info(f"  Mode override: {args.mode}")
    
    model_name = calibration['model_name']
    mode = calibration['mode']
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Mode: {mode}")
    
    # Initialize SteeringLLM
    logger.info("Initializing model...")
    start_load = time.time()
    llm = SteeringLLM.from_calibration(
        calibration,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        # enforce_eager=True, # Must be True not to bypass PyTorch forward hooks
        max_model_len=8192,
    )
    load_time = time.time() - start_load
    logger.info(f"✓ Model loaded in {load_time:.1f}s")
    
    # Get tokenizer
    tokenizer = llm.llm.get_tokenizer()
    
    # Load benchmark data
    benchmark = BenchmarkRegistry.get(args.benchmark)
    split = args.split or benchmark.default_split
    logger.info(f"Loading benchmark: {args.benchmark} (split: {split})")
    data = benchmark.load_data(split=split, max_samples=args.max_samples)
    logger.info(f"Loaded {len(data)} samples")
    
    # Get prompts and ground truths
    raw_prompts = [d["prompt"] for d in data]
    ground_truths = [d["ground_truth"] for d in data]
    
    # Get system prompt (unless disabled)
    system_prompt = None if args.no_system_prompt else benchmark.get_system_prompt()
    if system_prompt:
        logger.info(f"Using system prompt: {system_prompt[:80]}...")
    
    # Format prompts with chat template
    formatted_prompts = format_prompts_with_chat_template(raw_prompts, tokenizer, system_prompt)
    
    # Sampling params
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate for each degree
    degrees = list(range(args.degree_start, args.degree_end + 1, args.degree_step))
    if not degrees:
        degrees = [args.degree_start]
    
    logger.info(f"Testing degrees: {degrees}")
    
    for degree in degrees:
        logger.info(f"--- θ = {degree}° ---")
        start = time.time()
        vllm_outputs = llm.generate(formatted_prompts, theta=degree, sampling_params=sampling_params)
        elapsed = time.time() - start
        logger.info(f"Generation time: {elapsed:.2f}s")
        
        # Build results
        results = []
        for i, (prompt, vllm_out, gt) in enumerate(zip(raw_prompts, vllm_outputs, ground_truths)):
            results.append({
                "prompt": prompt,
                "response": vllm_out.outputs[0].text.strip(),
                "ground_truth": gt,
            })
        
        # Build output
        output = {
            "degree": degree,
            "metadata": {
                "benchmark": args.benchmark,
                "split": split,
                "num_samples": len(results),
                "generation_time": elapsed,
                "model": model_name,
                "mode": calibration['mode'],
                "use_system_prompt": not args.no_system_prompt,
            },
            "results": results
        }
        
        # Save
        output_file = output_dir / f"results_degree_{degree}.json"
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved results to {output_file}")
    
    logger.info("=" * 60)
    logger.info("Generation completed!")
    logger.info("=" * 60)


def generate_baseline(args: argparse.Namespace) -> None:
    """Generate responses using vanilla vLLM (no steering)."""
    logger.info("=" * 60)
    logger.info("Generate Robustness Baseline (No Steering)")
    logger.info("=" * 60)
    
    # Initialize vLLM
    logger.info(f"Initializing model: {args.model}...")
    start_load = time.time()
    llm = LLM(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=8192,
    )
    load_time = time.time() - start_load
    logger.info(f"✓ Model loaded in {load_time:.1f}s")
    
    # Get tokenizer
    tokenizer = llm.get_tokenizer()
    
    # Load benchmark data
    benchmark = BenchmarkRegistry.get(args.benchmark)
    split = args.split or benchmark.default_split
    logger.info(f"Loading benchmark: {args.benchmark} (split: {split})")
    data = benchmark.load_data(split=split, max_samples=args.max_samples)
    logger.info(f"Loaded {len(data)} samples")
    
    # Get prompts and ground truths
    raw_prompts = [d["prompt"] for d in data]
    ground_truths = [d["ground_truth"] for d in data]
    
    # Get system prompt (unless disabled)
    system_prompt = None if args.no_system_prompt else benchmark.get_system_prompt()
    if system_prompt:
        logger.info(f"Using system prompt: {system_prompt[:80]}...")
    
    # Format prompts with chat template
    formatted_prompts = format_prompts_with_chat_template(raw_prompts, tokenizer, system_prompt)
    
    # Sampling params
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate
    logger.info("Generating responses...")
    start = time.time()
    vllm_outputs = llm.generate(formatted_prompts, sampling_params)
    elapsed = time.time() - start
    logger.info(f"Generation time: {elapsed:.2f}s")
    
    # Build results
    results = []
    for i, (prompt, vllm_out, gt) in enumerate(zip(raw_prompts, vllm_outputs, ground_truths)):
        results.append({
            "prompt": prompt,
            "response": vllm_out.outputs[0].text.strip(),
            "ground_truth": gt,
        })
    
    # Build output (degree=None for baseline)
    output = {
        "degree": None,
        "metadata": {
            "benchmark": args.benchmark,
            "split": split,
            "num_samples": len(results),
            "generation_time": elapsed,
            "model": args.model,
            "mode": "baseline",
            "use_system_prompt": not args.no_system_prompt,
        },
        "results": results
    }
    
    # Save
    output_file = output_dir / "results_baseline.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved results to {output_file}")
    
    logger.info("=" * 60)
    logger.info("Generation completed!")
    logger.info("=" * 60)


def main() -> None:
    args = parse_args()
    
    if args.calibration:
        generate_with_steering(args)
    else:
        generate_baseline(args)


if __name__ == "__main__":
    main()
