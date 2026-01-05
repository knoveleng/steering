#!/usr/bin/env python3
"""
vLLM Load and Steer Example

Example script to demonstrate loading calibration artifacts
and using them for steering with vLLM inference engine.

Usage:
    python load_and_steer_vllm.py --calibration artifacts/calibration_Qwen2.5-3B-Instruct_basic
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Union

from vllm import SamplingParams
from steering import SteeringLLM
from ui.utils import load_calibration
from steering.utils.logger import setup_logger

# Enable insecure serialization for vLLM v0.12+
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'

logger = setup_logger("steering.load_and_steer")


def parse_args():
    parser = argparse.ArgumentParser(
        description="vLLM Load and Steer Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python load_and_steer_vllm.py --calibration artifacts/calibration_Qwen2.5-3B-Instruct_basic
        """
    )
    parser.add_argument('--calibration', type=str, required=True, help='Path to calibration session')
    parser.add_argument('--data', type=str, default='data/advbench_test.json', help='Path to input data JSON file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results (default: eval/{model_name})')
    parser.add_argument('--mode', type=str, choices=['standard', 'adaptive', 'selective', 'addition', 'ablation'],
                        help='Override steering mode from calibration')
    parser.add_argument('--model-id', type=str, help='Override model ID from calibration')
    parser.add_argument('--max-tokens', type=int, default=512, help='Max tokens to generate')
    parser.add_argument('--tensor-parallel-size', type=int, default=1, help='Tensor parallel size')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.8, help='GPU memory utilization')
    parser.add_argument('--system-prompt', type=str, default=None, help='Optional system prompt')
    parser.add_argument('--no-chat-template', action='store_true', help='Disable chat template formatting')
    parser.add_argument('--degree-start', type=int, default=0, help='Start degree for range')
    parser.add_argument('--degree-end', type=int, default=360, help='End degree for range')
    parser.add_argument('--degree-step', type=int, default=60, help='Step size for degree range')
    return parser.parse_args()


def setup_tokenizer(tokenizer) -> None:
    """
    Ensure tokenizer has required special tokens.
    Following the pattern from steering_pipeline.py.
    """
    if tokenizer.pad_token is None:
        logger.warning("Tokenizer has no pad_token. Using eos_token as pad_token.")
        if tokenizer.eos_token is None:
            logger.warning("Tokenizer also has no eos_token. Adding '[PAD]' as pad token.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        else:
            tokenizer.pad_token = tokenizer.eos_token


def format_prompts(
    prompts: Union[str, List[str]],
    tokenizer,
    use_chat_template: bool = True,
    system_prompt: Optional[str] = None,
    add_generation_prompt: bool = True,
) -> List[str]:
    """
    Format prompts using chat template if available.
    
    Args:
        prompts: Single prompt string or list of prompts
        tokenizer: Tokenizer with chat template
        use_chat_template: Whether to apply chat template
        system_prompt: Optional system prompt to prepend
        add_generation_prompt: Whether to add generation prompt marker
        
    Returns:
        List of formatted prompts
    """
    # Normalize to list
    if isinstance(prompts, str):
        prompts = [prompts]
    
    # Return as-is if chat template disabled
    if not use_chat_template:
        return prompts
    
    # Check if tokenizer has chat template
    if not hasattr(tokenizer, 'apply_chat_template') or tokenizer.chat_template is None:
        logger.warning("Tokenizer has no chat template. Using prompts as-is.")
        return prompts
    
    formatted_prompts = []
    for prompt in prompts:
        # Build messages list
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Apply chat template
        try:
            formatted = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=add_generation_prompt
            )
            formatted_prompts.append(formatted)
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}. Using raw prompt.")
            formatted_prompts.append(prompt)
    
    return formatted_prompts


def load_data(data_file: str) -> List[str]:
    """Load prompts from JSON data file."""
    with open(data_file, "r") as f:
        data = json.load(f)
    return [point["prompt"] for point in data]


def main():
    args = parse_args()

    logger.info("="*60)
    logger.info("Load and Steer Example")
    logger.info("="*60)
    
    calibration_path = args.calibration
    logger.info(f"Using calibration: {calibration_path}")
    
    # Load calibration data
    logger.info("\nLoading calibration...")
    calibration = load_calibration(calibration_path)
    
    # Override mode if specified
    if args.mode:
        calibration['mode'] = args.mode
        logger.info(f"  Mode override: {args.mode}")
    
    # Override model if specified
    if args.model_id:
        calibration['model_name'] = args.model_id
        logger.info(f"  Model override: {args.model_id}")
    
    logger.info(f"  Model: {calibration['model_name']}")
    logger.info(f"  Mode: {calibration['mode']}")
    if calibration['target_layers']:
        logger.info(f"  Target layers: {len(calibration['target_layers'])}")
    
    # Initialize SteeringLLM
    logger.info("\nInitializing model...")
    start_load = time.time()
    llm = SteeringLLM.from_calibration(
        calibration,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=True, # Must be True not to bypass PyTorch forward hooks
        max_model_len=4096,
    )
    load_time = time.time() - start_load
    logger.info(f"✓ Model loaded in {load_time:.1f}s")
    
    # Setup tokenizer
    tokenizer = llm.llm.get_tokenizer()
    setup_tokenizer(tokenizer)
    
    # Sampling params
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    
    # Load test prompts
    logger.info(f"\nLoading data from {args.data}")
    prompts = load_data(args.data)
    logger.info(f"Loaded {len(prompts)} prompts")
    
    # Format prompts with chat template
    use_chat_template = not args.no_chat_template
    formatted_prompts = format_prompts(
        prompts,
        tokenizer,
        use_chat_template=use_chat_template,
        system_prompt=args.system_prompt,
    )
    
    if use_chat_template:
        logger.info(f"Applied chat template to {len(formatted_prompts)} prompts")
    
    # Set output directory
    model_name = calibration['model_name']
    if args.output_dir:
        output_dir = Path(args.output_dir) / model_name.split('/')[-1]
    else:
        output_dir = Path("eval") / model_name.split('/')[-1]
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Test various steering angles
    degrees = list(range(args.degree_start, args.degree_end + 1, args.degree_step))
    logger.info(f"Testing degrees: {degrees}")
    
    for degree in degrees:
        logger.info(f"--- θ = {degree}° ---")
        start = time.time()
        vllm_outputs = llm.generate(formatted_prompts, theta=degree, sampling_params=sampling_params)
        elapsed = time.time() - start
        logger.info(f"Time: {elapsed:.2f}s")

        # Build outputs list
        results = []
        for i, (prompt, vllm_out) in enumerate(zip(prompts, vllm_outputs)):
            results.append({
                "prompt": prompt,
                "response": vllm_out.outputs[0].text.strip(),
            })
        
        # Calculate metadata
        metadata = {
            "num_samples": len(results),
            "generation_time": elapsed,
            "degree": degree,
        }
        
        # Save results in same format as eval_perplexity_vllm.py
        result = {
            "degree": degree,
            "metadata": metadata,
            "results": results
        }
        
        output_file = output_dir / f"results_degree_{degree}.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved results to {output_file}")
    
    logger.info(f"\n{'='*60}")
    logger.info("All experiments completed!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
