#!/usr/bin/env python3
"""
Extract Best Theta from Standard Logs

Finds the best theta from jailbreak logs based on a metric,
and copies ALL metric files at that theta for both jailbreak and robustness.

Usage:
    # Find best harmbench theta, copy all jailbreak and robustness metrics at that theta
    python examples/extract_best_theta.py \\
        --jailbreak-input logs/jailbreak/standard \\
        --jailbreak-output logs/jailbreak/addition \\
        --robustness-input logs/robustness-evaluation/standard \\
        --robustness-output logs/robustness-evaluation/addition \\
        --metric harmbench --mode max
        
    # Jailbreak only (without robustness)
    python examples/extract_best_theta.py \\
        --jailbreak-input logs/jailbreak/standard \\
        --jailbreak-output logs/jailbreak/addition \\
        --metric harmbench --mode max
"""

import argparse
import json
import shutil
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# All jailbreak metrics to copy
JAILBREAK_METRICS = [
    "harmbench", "llama_guard", "substring", "compression_ratio",
    "language_consistency", "ngram_repetition", "qwen3guard", "polyguard", "llm_judge"
]

# All robustness benchmarks to copy
ROBUSTNESS_BENCHMARKS = [
    "tinyAI2_arc", "tinyGSM8k", "tinyMMLU", "tinyTruthfulQA", "tinyWinogrande"
]


def find_result_files(input_dir: str, metric: str) -> Dict[str, List[str]]:
    """
    Find all result files organized by model for a specific metric.
    """
    results = {}
    input_path = Path(input_dir)
    
    if not input_path.exists():
        return results
    
    for model_dir in input_path.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        
        # Try metric subdirectory first (jailbreak format)
        metric_subdir = model_dir / metric
        if metric_subdir.exists() and metric_subdir.is_dir():
            search_dir = metric_subdir
        else:
            # Fall back to model dir (perplexity format)
            search_dir = model_dir
            
        result_files = list(search_dir.glob("results_degree_*.json"))
        
        if result_files:
            results[model_name] = [str(f) for f in sorted(result_files)]
    
    return results


def extract_degree_from_filename(filename: str) -> int:
    """Extract degree value from filename like results_degree_60.json"""
    match = re.search(r'results_degree_(\d+)', filename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract degree from {filename}")


def get_metric_value(data: dict, metric: str) -> Optional[float]:
    """Extract metric value from result data."""
    # Handle common metrics from metadata (jailbreak format)
    if "metadata" in data:
        metadata = data["metadata"]
        if metric in metadata and metadata[metric] is not None:
            return float(metadata[metric])
    
    # Handle perplexity
    if metric == "perplexity":
        if "results" in data and isinstance(data["results"], list):
            perplexities = [r.get("perplexity", 0) for r in data["results"] if "perplexity" in r]
            if perplexities:
                return sum(perplexities) / len(perplexities)
        if "average_perplexity" in data:
            return data["average_perplexity"]
    
    # Direct lookup
    if metric in data:
        val = data[metric]
        if isinstance(val, (int, float)):
            return float(val)
    
    return None


def find_best_theta(
    result_files: List[str],
    metric: str,
    mode: str = "min",
    max_degree: int = 360,
    step: int = 10
) -> Tuple[Optional[int], Optional[str], Optional[float]]:
    """Find the best theta based on metric."""
    best_degree = None
    best_file = None
    best_value = None
    
    for filepath in result_files:
        try:
            degree = extract_degree_from_filename(filepath)
            
            # Filter: only consider degrees from 0 to max_degree
            if degree > max_degree:
                continue
            
            # Filter: only consider degrees that match the step
            if step > 0 and degree % step != 0 and degree != 0:
                continue
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            value = get_metric_value(data, metric)
            if value is None:
                continue
            
            if best_value is None:
                best_degree, best_file, best_value = degree, filepath, value
            elif mode == "min" and value < best_value:
                best_degree, best_file, best_value = degree, filepath, value
            elif mode == "max" and value > best_value:
                best_degree, best_file, best_value = degree, filepath, value
                
        except (json.JSONDecodeError, IOError, ValueError) as e:
            print(f"Warning: Could not read {filepath}: {e}")
            continue
    
    return best_degree, best_file, best_value


def copy_jailbreak_metrics_at_degree(
    input_dir: Path,
    output_dir: Path,
    model_name: str,
    degree: int,
    logger
) -> int:
    """
    Copy all jailbreak metric files at the specified degree for a model.
    
    Returns number of metrics copied.
    """
    model_input_dir = input_dir / model_name
    copied = 0
    
    for metric in JAILBREAK_METRICS:
        metric_input_dir = model_input_dir / metric
        if not metric_input_dir.exists():
            continue
            
        # Find file matching this degree
        for f in metric_input_dir.glob(f"results_degree_{degree}_*.json"):
            # Create output directory
            metric_output_dir = output_dir / model_name / metric
            metric_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            output_file = metric_output_dir / f.name
            shutil.copy2(f, output_file)
            copied += 1
            break
    
    return copied


def find_closest_degree(target_degree: int, available_degrees: List[int], step: int = 30) -> Optional[int]:
    """
    Find the closest available degree to the target.
    """
    if not available_degrees:
        return None
    
    if target_degree in available_degrees:
        return target_degree
    
    # Find closest
    closest = min(available_degrees, key=lambda x: abs(x - target_degree))
    
    # Check if within acceptable range
    if abs(closest - target_degree) <= step:
        return closest
    
    return None


def get_available_degrees(benchmark_dir: Path) -> List[int]:
    """Get all available degrees in a benchmark directory."""
    degrees = set()
    
    if not benchmark_dir.exists():
        return []
    
    for json_file in benchmark_dir.glob("*.json"):
        match = re.search(r'results_degree_(\d+)', json_file.name)
        if match:
            degrees.add(int(match.group(1)))
    
    return sorted(degrees)


def copy_robustness_metrics_at_degree(
    input_dir: Path,
    output_dir: Path,
    model_name: str,
    target_degree: int,
    robustness_step: int,
    logger
) -> Tuple[int, Optional[int]]:
    """
    Copy all robustness benchmark files at the specified degree (or closest) for a model.
    
    Returns tuple of (number of benchmarks copied, actual degree used).
    """
    model_input_dir = input_dir / model_name
    copied = 0
    actual_degree = None
    
    if not model_input_dir.exists():
        return 0, None
    
    for benchmark in ROBUSTNESS_BENCHMARKS:
        benchmark_input_dir = model_input_dir / benchmark
        if not benchmark_input_dir.exists():
            continue
        
        # Find available degrees and get closest match
        available_degrees = get_available_degrees(benchmark_input_dir)
        degree_to_use = find_closest_degree(target_degree, available_degrees, robustness_step)
        
        if degree_to_use is None:
            continue
        
        actual_degree = degree_to_use
        
        # Find file matching this degree
        for f in benchmark_input_dir.glob(f"results_degree_{degree_to_use}_*.json"):
            # Create output directory
            benchmark_output_dir = output_dir / model_name / benchmark
            benchmark_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            output_file = benchmark_output_dir / f.name
            shutil.copy2(f, output_file)
            copied += 1
            break
    
    return copied, actual_degree


def main():
    parser = argparse.ArgumentParser(
        description="Extract best theta and copy jailbreak/robustness metrics at that theta",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Find best harmbench theta, copy ALL jailbreak and robustness metrics at that theta
    python examples/extract_best_theta.py \\
        --jailbreak-input logs/jailbreak/standard \\
        --jailbreak-output logs/jailbreak/addition \\
        --robustness-input logs/robustness-evaluation/standard \\
        --robustness-output logs/robustness-evaluation/addition \\
        --metric harmbench --mode max
        
    # Jailbreak only
    python examples/extract_best_theta.py \\
        --jailbreak-input logs/jailbreak/standard \\
        --jailbreak-output logs/jailbreak/addition \\
        --metric harmbench --mode max
        """
    )
    
    # Jailbreak arguments
    parser.add_argument('--jailbreak-input', type=str, required=True,
                        help='Input directory with jailbreak logs (e.g., logs/jailbreak/standard)')
    parser.add_argument('--jailbreak-output', type=str, required=True,
                        help='Output directory for jailbreak logs (e.g., logs/jailbreak/addition)')
    
    # Robustness arguments (optional)
    parser.add_argument('--robustness-input', type=str, default=None,
                        help='Input directory with robustness logs (optional)')
    parser.add_argument('--robustness-output', type=str, default=None,
                        help='Output directory for robustness logs (optional)')
    
    # Metric arguments
    parser.add_argument('--metric', type=str, required=True,
                        help='Primary metric to optimize (e.g., harmbench)')
    parser.add_argument('--mode', type=str, choices=['min', 'max'], default='max',
                        help='Optimization mode: min or max (default: max)')
    parser.add_argument('--max-degree', type=int, default=360,
                        help='Maximum degree to consider (default: 360)')
    parser.add_argument('--step', type=int, default=30,
                        help='Degree step for jailbreak (default: 30)')
    parser.add_argument('--robustness-step', type=int, default=30,
                        help='Degree step for robustness, used to find closest match (default: 30)')
    
    args = parser.parse_args()
    
    # Validate robustness arguments
    if args.robustness_input and not args.robustness_output:
        parser.error("--robustness-output is required when --robustness-input is specified")
    if args.robustness_output and not args.robustness_input:
        parser.error("--robustness-input is required when --robustness-output is specified")
    
    logger = logging.getLogger("steering.extract_best_theta")
    
    logger.info("=" * 60)
    logger.info("Extract Best Theta")
    logger.info("=" * 60)
    logger.info(f"Jailbreak input: {args.jailbreak_input}")
    logger.info(f"Jailbreak output: {args.jailbreak_output}")
    if args.robustness_input:
        logger.info(f"Robustness input: {args.robustness_input}")
        logger.info(f"Robustness output: {args.robustness_output}")
    logger.info(f"Primary metric: {args.metric}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Max degree: {args.max_degree}")
    logger.info(f"Step: {args.step}")
    if args.robustness_input:
        logger.info(f"Robustness step: {args.robustness_step}")
    
    # Find all result files for primary metric
    model_results = find_result_files(args.jailbreak_input, args.metric)
    
    if not model_results:
        logger.error(f"No result files found in {args.jailbreak_input}")
        return
    
    logger.info(f"Found {len(model_results)} models")
    
    jailbreak_input_path = Path(args.jailbreak_input)
    jailbreak_output_path = Path(args.jailbreak_output)
    robustness_input_path = Path(args.robustness_input) if args.robustness_input else None
    robustness_output_path = Path(args.robustness_output) if args.robustness_output else None
    
    # Process each model
    summary = []
    for model_name, result_files in sorted(model_results.items()):
        best_degree, best_file, best_value = find_best_theta(
            result_files, args.metric, args.mode, args.max_degree, args.step
        )
        
        if best_degree is not None:
            logger.info(f"  {model_name}: best θ = {best_degree}° ({args.metric} = {best_value:.4f})")
            
            # Copy jailbreak metrics at this degree
            jb_copied = copy_jailbreak_metrics_at_degree(
                jailbreak_input_path, jailbreak_output_path, model_name, best_degree, logger
            )
            logger.info(f"    → Copied {jb_copied} jailbreak metrics at θ = {best_degree}°")
            
            # Copy robustness metrics at this degree (or closest)
            rob_copied = 0
            rob_degree = None
            if robustness_input_path and robustness_output_path:
                rob_copied, rob_degree = copy_robustness_metrics_at_degree(
                    robustness_input_path, robustness_output_path, 
                    model_name, best_degree, args.robustness_step, logger
                )
                if rob_degree is not None:
                    if rob_degree != best_degree:
                        logger.info(f"    → Copied {rob_copied} robustness benchmarks at θ = {rob_degree}° (closest to {best_degree}°)")
                    else:
                        logger.info(f"    → Copied {rob_copied} robustness benchmarks at θ = {rob_degree}°")
                else:
                    logger.warning(f"    → No robustness data found for {model_name}")
            
            summary.append({
                "model": model_name,
                "best_degree": best_degree,
                "primary_metric": args.metric,
                "value": best_value,
                "jailbreak_metrics_copied": jb_copied,
                "robustness_benchmarks_copied": rob_copied,
                "robustness_degree_used": rob_degree
            })
        else:
            logger.warning(f"  {model_name}: No valid {args.metric} values found")
    
    # Save summary
    if summary:
        jailbreak_output_path.mkdir(parents=True, exist_ok=True)
        
        summary_file = jailbreak_output_path / f"best_theta_summary_{args.metric}.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "primary_metric": args.metric,
                "mode": args.mode,
                "max_degree": args.max_degree,
                "step": args.step,
                "jailbreak_input_dir": args.jailbreak_input,
                "robustness_input_dir": args.robustness_input,
                "results": summary
            }, f, indent=2)
        
        logger.info(f"\nSummary saved to: {summary_file}")
    
    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
