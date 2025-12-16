#!/usr/bin/env python3
"""
Evaluation script for harm detection using multiple evaluators.

This script evaluates model responses for harm detection using configurable
evaluators (substring, llama_guard, harmbench).
"""

import argparse
import os
import json
from pathlib import Path
from typing import List, Optional

from steering.evaluation import EvaluationSuite
from steering.utils.logger import setup_logger

# Setup logger
logger = setup_logger("steering.example.eval_harm")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate model responses for harm detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing input JSON files with prompts and responses"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Root output directory for evaluation results"
    )
    
    parser.add_argument(
        "--evaluators",
        type=str,
        nargs="+",
        default=["substring", "llama_guard", "harmbench"],
        choices=["substring", "llama_guard", "harmbench", "ngram_repetition", "language_consistency", "compression_ratio", "qwen3guard", "polyguard", "llm_judge"],
        help="List of evaluators to use"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate per file (default: all samples)"
    )
    
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="*.json",
        help="Glob pattern to match input files (default: *.json)"
    )
    
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Input directory does not exist: {args.input_dir}")
    
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError(f"max-samples must be positive, got: {args.max_samples}")


def load_json_file(file_path: Path) -> dict:
    """Load and validate JSON file."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Validate required fields
        if "degree" not in data:
            logger.warning(f"Missing 'degree' field in {file_path}")
        if "metadata" not in data:
            logger.warning(f"Missing 'metadata' field in {file_path}, using empty dict")
            data["metadata"] = {}
        if "results" not in data:
            raise ValueError(f"Missing 'results' field in {file_path}")
        
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error reading {file_path}: {e}")


def evaluate_file_with_evaluator(
    file_path: Path,
    evaluator_name: str,
    evaluator,
    output_dir: Path,
    max_samples: Optional[int] = None
) -> None:
    """
    Evaluate a single file using the specified evaluator.
    
    Args:
        file_path: Path to input JSON file
        evaluator_name: Name of the evaluator (for logging and metadata)
        evaluator: Evaluator instance to use
        output_dir: Output directory for results
        max_samples: Maximum number of samples to evaluate (None for all)
    """
    logger.info(f"Evaluating {file_path.name} with {evaluator_name}...")
    
    # Load data
    data = load_json_file(file_path)
    degree = data.get("degree", None)
    metadata = data.get("metadata", {})
    results = data["results"]
    
    # Extract prompts and responses
    prompts = [r["prompt"] for r in results]
    responses = [r["response"] for r in results]
    
    # Limit samples if specified
    if max_samples is not None:
        prompts = prompts[:max_samples]
        responses = responses[:max_samples]
        results = results[:max_samples]
    
    if not prompts or not responses:
        logger.warning(f"No valid prompts/responses found in {file_path}")
        return
    
    # Evaluate
    score, labels = evaluator.evaluate(prompts, responses, return_labels=True)
    logger.info(f"Score: {score:.4f} - Degree: {degree}")
    
    # Update metadata with score
    metadata[evaluator_name] = score
    
    # Update labels in results
    for i, label in enumerate(labels):
        results[i][evaluator_name] = label
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    output_file = output_dir / f"{file_path.stem}_{evaluator_name}.json"
    
    # Save results
    with open(output_file, "w") as f:
        json.dump({
            "degree": degree,
            "metadata": metadata,
            "results": results
        }, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")


def main() -> None:
    """Main evaluation function."""
    args = parse_args()
    validate_args(args)
    
    # Convert to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Initialize evaluation suite
    evaluator_suite = EvaluationSuite()
    
    # Get available evaluators
    available_evaluators = evaluator_suite.get_available_evaluators()
    
    # Validate evaluator names
    for evaluator_name in args.evaluators:
        if evaluator_name not in available_evaluators:
            raise ValueError(
                f"Unknown evaluator: {evaluator_name}. "
                f"Available: {available_evaluators}"
            )
    
    # Find input files
    input_files = sorted(input_dir.glob(args.file_pattern))
    
    if not input_files:
        logger.warning(f"No files found matching pattern '{args.file_pattern}' in {input_dir}")
        return
    
    logger.info(f"Found {len(input_files)} file(s) to evaluate")
    
    # Evaluate each evaluator
    for evaluator_name in args.evaluators:
        logger.info(f"Evaluating with metric: {evaluator_name}...")
        evaluator_output_dir = output_dir / evaluator_name
        
        # Create evaluator once per evaluator type (not per file)
        evaluator = None
        try:
            evaluator = evaluator_suite._get_evaluator(evaluator_name)
            
            # Process each file
            for file_path in input_files:
                try:
                    evaluate_file_with_evaluator(
                        file_path,
                        evaluator_name,
                        evaluator,
                        evaluator_output_dir,
                        args.max_samples
                    )
                except Exception as e:
                    logger.error(f"Error evaluating {file_path} with {evaluator_name}: {e}")
                    continue
        finally:
            # Free memory of evaluator to avoid OOM in GPU memory
            if evaluator is not None:
                del evaluator
            logger.info(f"Completed evaluation with {evaluator_name}")
    
    logger.info("Evaluation completed successfully")


if __name__ == "__main__":
    main()
