#!/usr/bin/env python3
"""
vLLM Steering Evaluation with Perplexity

Runs steering experiments with vLLM and calculates perplexity for outputs.
Output format follows eval_perplexity.py conventions.

Usage:
    python eval_perplexity_vllm.py --calibration artifacts/calibration_gemma-2-9b-it_selective/
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from vllm import SamplingParams
from steering import SteeringLLM
from ui.utils import load_calibration
from steering.utils.logger import setup_logger

# Enable insecure serialization for vLLM v0.12+
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'

logger = setup_logger("steering.eval_perplexity_vllm")


def parse_args():
    parser = argparse.ArgumentParser(
        description="vLLM Steering Evaluation with Perplexity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--calibration', type=str, required=True, 
                        help='Path to calibration session')
    parser.add_argument('--data', type=str, default='./data/advbench_test.json',
                        help='Path to input data JSON file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results (default: eval/{model_name})')
    parser.add_argument('--degrees', type=int, nargs='+', default=None,
                        help='Specific degrees to test (e.g., --degrees 0 45 90)')
    parser.add_argument('--degree-start', type=int, default=0,
                        help='Start degree for range (default: 0)')
    parser.add_argument('--degree-end', type=int, default=360,
                        help='End degree for range (default: 360)')
    parser.add_argument('--degree-step', type=int, default=60,
                        help='Step size for degree range (default: 60)')
    parser.add_argument('--max-tokens', type=int, default=512,
                        help='Max tokens to generate (default: 512)')
    parser.add_argument('--tensor-parallel-size', type=int, default=1,
                        help='Tensor parallel size')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.8,
                        help='GPU memory utilization for vLLM')
    parser.add_argument('--no-perplexity', action='store_true',
                        help='Disable perplexity calculation')
    parser.add_argument('--system-prompt', type=str, default=None,
                        help='Optional system prompt')
    parser.add_argument('--no-chat-template', action='store_true',
                        help='Disable chat template formatting')
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


def calculate_perplexity_from_logprobs(logprobs: List[Dict]) -> float:
    """
    Calculate perplexity from vLLM's logprobs output.
    
    Perplexity = exp(-1/N * sum(log_probs))
    where N is the number of tokens.
    
    Args:
        logprobs: List of logprob dicts from vLLM output
        
    Returns:
        Perplexity value
    """
    if not logprobs:
        return float('inf')
    
    total_log_prob = 0.0
    num_tokens = 0
    
    for token_logprob in logprobs:
        if token_logprob is not None:
            # Get the log probability of the chosen token
            # In vLLM, each entry contains the logprob of the sampled token
            for token_id, logprob_obj in token_logprob.items():
                total_log_prob += logprob_obj.logprob
                num_tokens += 1
                break  # Only take the first (sampled) token
    
    if num_tokens == 0:
        return float('inf')
    
    # Perplexity = exp(-mean_log_prob)
    avg_neg_log_prob = -total_log_prob / num_tokens
    perplexity = math.exp(avg_neg_log_prob)
    
    return perplexity


def calculate_metadata(outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate summary statistics from outputs."""
    metadata = {}
    
    if outputs and "perplexity" in outputs[0]:
        perplexities = [out["perplexity"] for out in outputs if out.get("perplexity") is not None]
        if perplexities:
            metadata["average_perplexity"] = sum(perplexities) / len(perplexities)
            metadata["min_perplexity"] = min(perplexities)
            metadata["max_perplexity"] = max(perplexities)
    
    metadata["num_samples"] = len(outputs)
    
    return metadata


def save_results(output_dir: Path, degree: int, outputs: List[Dict[str, Any]], metadata: Dict[str, Any]):
    """Save results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result = {
        "degree": degree,
        "metadata": metadata,
        "results": outputs
    }
    
    output_file = output_dir / f"results_degree_{degree}.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Saved results to {output_file}")


def main():
    args = parse_args()
    
    logger.info("="*60)
    logger.info("vLLM Steering Evaluation with Perplexity")
    logger.info("="*60)
    
    # Load calibration
    logger.info(f"Loading calibration from {args.calibration}")
    calibration = load_calibration(args.calibration)
    model_name = calibration['model_name']
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Mode: {calibration['mode']}")
    
    # Initialize SteeringLLM
    logger.info("Initializing vLLM model...")
    start_load = time.time()
    llm = SteeringLLM.from_calibration(
        calibration,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=4096,
    )
    load_time = time.time() - start_load
    logger.info(f"✓ Model loaded in {load_time:.1f}s")
    
    # Setup tokenizer
    tokenizer = llm.llm.get_tokenizer()
    setup_tokenizer(tokenizer)
    
    # Sampling params - enable logprobs for perplexity calculation
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=args.max_tokens,
        logprobs=1 if not args.no_perplexity else None,
    )
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    prompts = load_data(args.data)
    logger.info(f"Loaded {len(prompts)} prompts")

    # For gemma models, we should add instruction to the prompt
    if "gemma" in model_name.lower():
        logger.info("Adding instruction to prompts for gemma models")
        prompts = [f"Please answer the following request: {prompt}." for prompt in prompts]
    
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
    
    # Determine degrees to test
    if args.degrees:
        degrees = args.degrees
    else:
        degrees = list(range(args.degree_start, args.degree_end + 1, args.degree_step))
    
    logger.info(f"Testing degrees: {degrees}")
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir) / model_name.split('/')[-1]
    else:
        output_dir = Path("eval") / model_name.split('/')[-1]
    
    logger.info(f"Output directory: {output_dir}")
    
    # Run experiments
    for degree in degrees:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing degree: {degree}")
        logger.info(f"{'='*60}")
        
        # Generate with steering
        start = time.time()
        vllm_outputs = llm.generate(formatted_prompts, theta=degree, sampling_params=sampling_params)
        gen_time = time.time() - start
        logger.info(f"Generation time: {gen_time:.2f}s")
        
        # Extract responses
        responses = [out.outputs[0].text.strip() for out in vllm_outputs]
        
        # Build outputs list with perplexity from logprobs
        outputs = []
        for i, (prompt, response, vllm_out) in enumerate(zip(prompts, responses, vllm_outputs)):
            output_entry = {
                "prompt": prompt,
                "response": response,
            }
            
            # Calculate perplexity from logprobs if available
            if not args.no_perplexity and vllm_out.outputs[0].logprobs:
                ppl = calculate_perplexity_from_logprobs(vllm_out.outputs[0].logprobs)
                output_entry["perplexity"] = ppl
            
            outputs.append(output_entry)
        
        # Calculate metadata
        metadata = calculate_metadata(outputs)
        metadata["generation_time"] = gen_time
        metadata["degree"] = degree
        
        if "average_perplexity" in metadata:
            logger.info(f"Average perplexity: {metadata['average_perplexity']:.2f}")
        
        # Save results
        save_results(output_dir, degree, outputs, metadata)
    
    logger.info(f"\n{'='*60}")
    logger.info("All experiments completed!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()

