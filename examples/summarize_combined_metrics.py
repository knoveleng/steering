#!/usr/bin/env python3
"""
Summarize Combined Metrics: Find best theta from jailbreak ASR, then report robustness metrics.

This script:
1. Reads jailbreak evaluation logs to find the best theta based on a base metric (e.g., harmbench ASR)
2. Extracts the robustness metrics at that best theta
3. Outputs a combined summary table

The key insight is that we want to find the steering angle that achieves the best safety (lowest ASR),
and then report the robustness metrics at that angle to see the utility impact.

Usage:
    python examples/summarize_combined_metrics.py \\
        --jailbreak-dir logs/jailbreak \\
        --robustness-dir logs/robustness-evaluation \\
        --base-metric harmbench \\
        --output-file logs/combined_summary.txt
        
    # With specific methods and models
    python examples/summarize_combined_metrics.py \\
        --jailbreak-dir logs/jailbreak \\
        --robustness-dir logs/robustness-evaluation \\
        --base-metric harmbench \\
        --methods standard adaptive selective \\
        --max-degree 180
"""

import argparse
import json
import os
import csv
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import statistics


# Default jailbreak metrics and their optimization directions
JAILBREAK_METRICS = {
    "harmbench": {"agg": "max", "opt": "max", "display": "harmbench ↑"},
    "llama_guard": {"agg": "max", "opt": "max", "display": "llama_guard ↑"},
    "qwen3guard": {"agg": "max", "opt": "max", "display": "qwen3guard ↑"},
    "polyguard": {"agg": "max", "opt": "max", "display": "polyguard ↑"},
    "llm_judge": {"agg": "max", "opt": "max", "display": "llm_judge ↑"},
    "substring": {"agg": "min", "opt": "min", "display": "substring ↓"},
}

# Default robustness benchmarks
DEFAULT_BENCHMARKS = ["tinyAI2_arc", "tinyGSM8k", "tinyMMLU", "tinyTruthfulQA", "tinyWinogrande"]

# Default methods (subdirectories) - ordered as specified
# "baseline" is a special method that uses 0° data from any available source
DEFAULT_METHODS = ["baseline", "addition", "ablation", "standard", "adaptive", "selective"]

# Source methods to check for baseline data (in order of preference)
BASELINE_SOURCE_METHODS = ["selective", "standard", "adaptive", "ablation", "addition"]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Find best theta from jailbreak ASR, then report robustness metrics at that theta",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--jailbreak-dir",
        type=str,
        required=True,
        help="Path to jailbreak folder (e.g., logs/jailbreak)"
    )
    
    parser.add_argument(
        "--robustness-dir",
        type=str,
        required=True,
        help="Path to robustness-evaluation folder (e.g., logs/robustness-evaluation)"
    )
    
    parser.add_argument(
        "--base-metric",
        type=str,
        default="harmbench",
        choices=list(JAILBREAK_METRICS.keys()),
        help="Base jailbreak metric to find best theta (higher = safer)"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output TXT file path (default: combined_summary.txt in jailbreak dir parent)"
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Output CSV file path (optional)"
    )
    
    parser.add_argument(
        "--markdown",
        type=str,
        default=None,
        help="Output markdown file path (optional)"
    )
    
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=DEFAULT_METHODS,
        help="List of methods (subdirectories) to include"
    )
    
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=DEFAULT_BENCHMARKS,
        help="List of robustness benchmarks to include"
    )
    
    parser.add_argument(
        "--max-degree",
        type=int,
        default=360,
        help="Maximum degree to consider when finding best theta"
    )
    
    parser.add_argument(
        "--jailbreak-step",
        type=int,
        default=10,
        help="Degree step used in jailbreak evaluation (default 10)"
    )
    
    parser.add_argument(
        "--robustness-step",
        type=int,
        default=30,
        help="Degree step used in robustness evaluation (to find closest match)"
    )
    
    parser.add_argument(
        "--include-jailbreak-metrics",
        type=str,
        nargs="+",
        # default=["harmbench", "substring"],
        default=["harmbench"],
        help="Jailbreak metrics to include in the output (at best theta)"
    )
    
    return parser.parse_args()


def extract_degree_from_filename(filename: str) -> Optional[int]:
    """Extract degree value from filename like results_degree_60.json or results_degree_60_harmbench.json"""
    match = re.search(r'results_degree_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


def round_to_step(degree: int, step: int) -> int:
    """
    Round degree to nearest step value.
    
    Args:
        degree: Degree value to round
        step: Step size
        
    Returns:
        Nearest step value
    """
    return round(degree / step) * step


def load_jailbreak_scores_by_degree(
    metric_dir: Path, 
    metric_name: str,
    max_degree: int = 180
) -> Dict[int, List[float]]:
    """
    Load jailbreak scores organized by degree.
    
    Args:
        metric_dir: Path to the metric folder containing JSON files
        metric_name: Name of the metric to extract from metadata
        max_degree: Maximum degree to consider
        
    Returns:
        Dict mapping degree -> list of scores
    """
    scores_by_degree = defaultdict(list)
    
    if not metric_dir.exists():
        return scores_by_degree
    
    for json_file in sorted(metric_dir.glob("*.json")):
        try:
            degree = extract_degree_from_filename(json_file.name)
            if degree is None or degree > max_degree:
                continue
            
            with open(json_file, "r") as f:
                data = json.load(f)
            
            metadata = data.get("metadata", {})
            if metric_name in metadata:
                score = metadata[metric_name]
                if score is not None:
                    scores_by_degree[degree].append(float(score))
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Warning: Error reading {json_file}: {e}", file=sys.stderr)
            continue
    
    return scores_by_degree


def find_best_theta_for_model(
    model_dir: Path,
    metric_name: str,
    max_degree: int = 180,
    jailbreak_step: int = 10
) -> Tuple[Optional[int], Optional[float]]:
    """
    Find the best theta for a model based on the specified metric.
    
    For safety metrics (harmbench, llama_guard, etc.), higher is better (safer).
    For ASR metrics (substring), lower is better (safer).
    
    Args:
        model_dir: Path to model directory
        metric_name: Metric to optimize
        max_degree: Maximum degree to consider
        jailbreak_step: Only consider degrees that are multiples of this step
        
    Returns:
        Tuple of (best_degree, best_value)
    """
    metric_dir = model_dir / metric_name
    if not metric_dir.exists():
        return None, None
    
    scores_by_degree = load_jailbreak_scores_by_degree(metric_dir, metric_name, max_degree)
    
    if not scores_by_degree:
        return None, None
    
    # Determine optimization direction
    opt_direction = JAILBREAK_METRICS.get(metric_name, {}).get("opt", "max")
    
    best_degree = None
    best_value = None
    
    # Filter to only degrees that match the step
    valid_degrees = [d for d in scores_by_degree.keys() 
                     if d % jailbreak_step == 0 or d == 0]
    
    for degree in valid_degrees:
        scores = scores_by_degree[degree]
        
        # Use mean for aggregation when finding best
        avg_score = statistics.mean(scores)
        
        if best_value is None:
            best_degree, best_value = degree, avg_score
        elif opt_direction == "max" and avg_score > best_value:
            best_degree, best_value = degree, avg_score
        elif opt_direction == "min" and avg_score < best_value:
            best_degree, best_value = degree, avg_score
    
    return best_degree, best_value


def find_available_degrees(benchmark_dir: Path) -> List[int]:
    """
    Find all available degrees in the benchmark directory.
    
    Args:
        benchmark_dir: Path to the benchmark folder
        
    Returns:
        Sorted list of available degrees
    """
    degrees = set()
    
    if not benchmark_dir.exists():
        return []
    
    for json_file in benchmark_dir.glob("*.json"):
        degree = extract_degree_from_filename(json_file.name)
        if degree is not None:
            degrees.add(degree)
    
    return sorted(degrees)


def find_closest_degree(target_degree: int, available_degrees: List[int], step: int = 30) -> Optional[int]:
    """
    Find the closest available degree to the target.
    
    Args:
        target_degree: Target degree to find
        available_degrees: List of available degrees
        step: Maximum allowed difference (if difference > step, return None)
        
    Returns:
        Closest available degree, or None if no match within step
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


def load_robustness_scores_at_degree(
    benchmark_dir: Path,
    target_degree: int,
    robustness_step: int = 30
) -> Tuple[List[float], Optional[int]]:
    """
    Load robustness accuracy scores at a specific degree (or closest available).
    
    Args:
        benchmark_dir: Path to the benchmark folder
        target_degree: Target degree to find
        robustness_step: Maximum step to consider for finding closest match
        
    Returns:
        Tuple of (list of accuracy scores, actual degree used)
    """
    scores = []
    
    if not benchmark_dir.exists():
        return scores, None
    
    # Find available degrees and get closest match
    available_degrees = find_available_degrees(benchmark_dir)
    actual_degree = find_closest_degree(target_degree, available_degrees, robustness_step)
    
    if actual_degree is None:
        return scores, None
    
    for json_file in sorted(benchmark_dir.glob("*.json")):
        try:
            degree = extract_degree_from_filename(json_file.name)
            if degree != actual_degree:
                continue
            
            with open(json_file, "r") as f:
                data = json.load(f)
            
            metadata = data.get("metadata", {})
            if "accuracy" in metadata:
                score = metadata["accuracy"]
                if score is not None:
                    scores.append(float(score))
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Warning: Error reading {json_file}: {e}", file=sys.stderr)
            continue
    
    return scores, actual_degree


def load_jailbreak_scores_at_degree(
    metric_dir: Path,
    metric_name: str,
    target_degree: int
) -> List[float]:
    """
    Load jailbreak metric scores at a specific degree.
    
    Args:
        metric_dir: Path to the metric folder
        metric_name: Name of the metric
        target_degree: Target degree to find
        
    Returns:
        List of scores at that degree
    """
    scores = []
    
    if not metric_dir.exists():
        return scores
    
    for json_file in sorted(metric_dir.glob("*.json")):
        try:
            degree = extract_degree_from_filename(json_file.name)
            if degree != target_degree:
                continue
            
            with open(json_file, "r") as f:
                data = json.load(f)
            
            metadata = data.get("metadata", {})
            if metric_name in metadata:
                score = metadata[metric_name]
                if score is not None:
                    scores.append(float(score))
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Warning: Error reading {json_file}: {e}", file=sys.stderr)
            continue
    
    return scores


def aggregate_scores(scores: List[float], method: str = "mean_std") -> str:
    """Aggregate scores using the specified method."""
    if not scores:
        return "N/A"
    
    if method == "mean_std":
        mean = statistics.mean(scores)
        if len(scores) > 1:
            std = statistics.stdev(scores)
            return f"{mean:.4f} ± {std:.4f}"
        else:
            return f"{mean:.4f}"
    elif method == "max":
        return f"{max(scores):.4f}"
    elif method == "min":
        return f"{min(scores):.4f}"
    else:
        return f"{statistics.mean(scores):.4f}"


def collect_combined_metrics(
    jailbreak_dir: Path,
    robustness_dir: Path,
    methods: List[str],
    base_metric: str,
    benchmarks: List[str],
    include_jailbreak_metrics: List[str],
    max_degree: int = 180,
    jailbreak_step: int = 10,
    robustness_step: int = 30
) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[Tuple[str, str], int], Dict[Tuple[str, str], Dict[str, int]]]:
    """
    Collect combined metrics for all models and methods.
    
    Args:
        jailbreak_dir: Path to jailbreak folder
        robustness_dir: Path to robustness folder
        methods: List of method names
        base_metric: Base jailbreak metric for finding best theta
        benchmarks: List of robustness benchmarks
        include_jailbreak_metrics: Jailbreak metrics to include in output
        max_degree: Maximum degree to consider
        jailbreak_step: Step for finding best theta (only consider multiples)
        robustness_step: Step for finding closest robustness degree
        
    Returns:
        Tuple of (results dict, best_theta dict, robustness_theta dict)
    """
    results = {}
    best_thetas = {}
    robustness_thetas = {}  # Track actual robustness degrees used
    baseline_models_processed = set()  # Track which models have baseline data
    
    for method in methods:
        # Handle baseline specially - use 0° from any available source method
        if method == "baseline":
            # Find models from any available source method
            for source_method in BASELINE_SOURCE_METHODS:
                source_jailbreak_dir = jailbreak_dir / source_method
                source_robustness_dir = robustness_dir / source_method
                
                if not source_jailbreak_dir.exists():
                    continue
                
                model_dirs = [d for d in source_jailbreak_dir.iterdir() if d.is_dir()]
                
                for model_dir in sorted(model_dirs):
                    model_name = model_dir.name
                    
                    # Skip if already processed this model's baseline
                    if model_name in baseline_models_processed:
                        continue
                    
                    key = ("baseline", model_name)
                    results[key] = {}
                    best_thetas[key] = 0
                    robustness_thetas[key] = {}
                    
                    # Get jailbreak metrics at 0°
                    for jb_metric in include_jailbreak_metrics:
                        metric_dir = model_dir / jb_metric
                        scores = load_jailbreak_scores_at_degree(metric_dir, jb_metric, 0)
                        agg_method = JAILBREAK_METRICS.get(jb_metric, {}).get("agg", "max")
                        results[key][jb_metric] = aggregate_scores(scores, agg_method)
                    
                    # Get robustness metrics at 0°
                    robustness_model_dir = source_robustness_dir / model_name
                    if robustness_model_dir.exists():
                        for benchmark in benchmarks:
                            benchmark_dir = robustness_model_dir / benchmark
                            scores, actual_degree = load_robustness_scores_at_degree(
                                benchmark_dir, 0, robustness_step
                            )
                            robustness_thetas[key][benchmark] = actual_degree
                            results[key][benchmark] = aggregate_scores(scores, "mean_std")
                    else:
                        for benchmark in benchmarks:
                            results[key][benchmark] = "N/A"
                    
                    baseline_models_processed.add(model_name)
            continue
        
        jailbreak_method_dir = jailbreak_dir / method
        robustness_method_dir = robustness_dir / method
        
        if not jailbreak_method_dir.exists():
            print(f"Warning: Jailbreak directory not found: {jailbreak_method_dir}", file=sys.stderr)
            continue
        
        # Get all model directories from jailbreak
        model_dirs = [d for d in jailbreak_method_dir.iterdir() if d.is_dir()]
        
        for model_dir in sorted(model_dirs):
            model_name = model_dir.name
            key = (method, model_name)
            results[key] = {}
            
            # Step 1: Find best theta based on jailbreak metric
            best_degree, best_value = find_best_theta_for_model(
                model_dir, base_metric, max_degree, jailbreak_step
            )
            
            if best_degree is None:
                print(f"Warning: No valid {base_metric} data for {method}/{model_name}", file=sys.stderr)
                best_thetas[key] = None
                continue
            
            best_thetas[key] = best_degree
            results[key]["best_θ"] = f"{best_degree}°"
            
            # Step 2: Add jailbreak metrics at best theta
            for jb_metric in include_jailbreak_metrics:
                metric_dir = model_dir / jb_metric
                scores = load_jailbreak_scores_at_degree(metric_dir, jb_metric, best_degree)
                
                agg_method = JAILBREAK_METRICS.get(jb_metric, {}).get("agg", "max")
                results[key][jb_metric] = aggregate_scores(scores, agg_method)
            
            # Step 3: Add robustness metrics at best theta (or closest available)
            robustness_model_dir = robustness_method_dir / model_name
            robustness_thetas[key] = {}
            
            if not robustness_model_dir.exists():
                print(f"Warning: Robustness directory not found: {robustness_model_dir}", file=sys.stderr)
                for benchmark in benchmarks:
                    results[key][benchmark] = "N/A"
                continue
            
            for benchmark in benchmarks:
                benchmark_dir = robustness_model_dir / benchmark
                scores, actual_degree = load_robustness_scores_at_degree(
                    benchmark_dir, best_degree, robustness_step
                )
                robustness_thetas[key][benchmark] = actual_degree
                
                if actual_degree is not None and actual_degree != best_degree:
                    # Just use the score without annotation
                    results[key][benchmark] = aggregate_scores(scores, "mean_std")
                else:
                    results[key][benchmark] = aggregate_scores(scores, "mean_std")
    
    return results, best_thetas, robustness_thetas


def print_table(
    results: Dict[Tuple[str, str], Dict[str, str]], 
    methods: List[str], 
    columns: List[str],
    column_display_names: Dict[str, str]
) -> None:
    """Print results as a formatted table to console."""
    if not results:
        print("No results to display.")
        return
    
    # Prepare headers
    headers = ["Model", "Method"] + [column_display_names.get(c, c) for c in columns]
    
    # Calculate column widths
    rows = []
    for (method, model), metric_values in sorted(
        results.items(), 
        key=lambda x: (x[0][1], methods.index(x[0][0]) if x[0][0] in methods else 999)
    ):
        row = [model, method] + [str(metric_values.get(c, "N/A")) for c in columns]
        rows.append(row)
    
    # Calculate max width for each column
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    
    # Print header
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print("\n" + "=" * len(header_line))
    print(header_line)
    print("-" * len(header_line))
    
    # Print rows (grouped by model with separators)
    prev_model = None
    for row in rows:
        current_model = row[0]
        if prev_model is not None and current_model != prev_model:
            print("-" * len(header_line))
        print(" | ".join(str(cell).ljust(w) for cell, w in zip(row, widths)))
        prev_model = current_model
    
    print("=" * len(header_line) + "\n")


def save_txt(
    results: Dict[Tuple[str, str], Dict[str, str]], 
    output_file: Path, 
    methods: List[str], 
    columns: List[str],
    column_display_names: Dict[str, str],
    base_metric: str,
    max_degree: int
) -> None:
    """Save results to TXT file."""
    if not results:
        print("No results to save.")
        return
    
    headers = ["Model", "Method"] + [column_display_names.get(c, c) for c in columns]
    
    rows = []
    for (method, model), metric_values in sorted(
        results.items(), 
        key=lambda x: (x[0][1], methods.index(x[0][0]) if x[0][0] in methods else 999)
    ):
        row = [model, method] + [str(metric_values.get(c, "N/A")) for c in columns]
        rows.append(row)
    
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    
    with open(output_file, "w") as f:
        # Write header info
        f.write(f"# Combined Metrics Summary\n")
        f.write(f"# Base metric for theta selection: {base_metric}\n")
        f.write(f"# Max degree considered: {max_degree}\n")
        f.write("\n")
        
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
        f.write("=" * len(header_line) + "\n")
        f.write(header_line + "\n")
        f.write("-" * len(header_line) + "\n")
        
        prev_model = None
        for row in rows:
            current_model = row[0]
            if prev_model is not None and current_model != prev_model:
                f.write("-" * len(header_line) + "\n")
            f.write(" | ".join(str(cell).ljust(w) for cell, w in zip(row, widths)) + "\n")
            prev_model = current_model
        
        f.write("=" * len(header_line) + "\n")
    
    print(f"TXT saved to: {output_file}")


def save_csv(
    results: Dict[Tuple[str, str], Dict[str, str]], 
    output_file: Path, 
    methods: List[str], 
    columns: List[str],
    column_display_names: Dict[str, str]
) -> None:
    """Save results to CSV file."""
    if not results:
        print("No results to save.")
        return
    
    headers = ["Model", "Method"] + [column_display_names.get(c, c) for c in columns]
    
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for (method, model), metric_values in sorted(
            results.items(), 
            key=lambda x: (x[0][1], methods.index(x[0][0]) if x[0][0] in methods else 999)
        ):
            row = [model, method] + [str(metric_values.get(c, "N/A")) for c in columns]
            writer.writerow(row)
    
    print(f"CSV saved to: {output_file}")


def save_markdown(
    results: Dict[Tuple[str, str], Dict[str, str]], 
    output_file: Path, 
    methods: List[str], 
    columns: List[str],
    column_display_names: Dict[str, str],
    base_metric: str,
    max_degree: int
) -> None:
    """Save results to markdown file."""
    if not results:
        print("No results to save.")
        return
    
    headers = ["Model", "Method"] + [column_display_names.get(c, c) for c in columns]
    
    with open(output_file, "w") as f:
        f.write("# Combined Metrics Summary\n\n")
        f.write(f"**Base metric for theta selection:** {base_metric}\n\n")
        f.write(f"**Max degree considered:** {max_degree}\n\n")
        
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---" for _ in headers]) + "|\n")
        
        for (method, model), metric_values in sorted(
            results.items(), 
            key=lambda x: (x[0][1], methods.index(x[0][0]) if x[0][0] in methods else 999)
        ):
            row = [model, method] + [str(metric_values.get(c, "N/A")) for c in columns]
            f.write("| " + " | ".join(str(cell) for cell in row) + " |\n")
    
    print(f"Markdown saved to: {output_file}")


def main() -> None:
    """Main function."""
    args = parse_args()
    
    jailbreak_dir = Path(args.jailbreak_dir)
    robustness_dir = Path(args.robustness_dir)
    
    if not jailbreak_dir.exists():
        print(f"Error: Jailbreak directory does not exist: {jailbreak_dir}", file=sys.stderr)
        sys.exit(1)
    
    if not robustness_dir.exists():
        print(f"Error: Robustness directory does not exist: {robustness_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Set default output file
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = jailbreak_dir.parent / "combined_summary.txt"
    
    methods = args.methods
    benchmarks = args.benchmarks
    base_metric = args.base_metric
    include_jailbreak_metrics = args.include_jailbreak_metrics
    max_degree = args.max_degree
    jailbreak_step = args.jailbreak_step
    robustness_step = args.robustness_step
    
    print(f"Jailbreak directory: {jailbreak_dir}")
    print(f"Robustness directory: {robustness_dir}")
    print(f"Output file: {output_file}")
    print(f"Base metric for theta selection: {base_metric}")
    print(f"Max degree: {max_degree}")
    print(f"Jailbreak step: {jailbreak_step}")
    print(f"Robustness step: {robustness_step}")
    print(f"Methods: {methods}")
    print(f"Benchmarks: {benchmarks}")
    print(f"Jailbreak metrics to include: {include_jailbreak_metrics}")
    
    # Collect metrics
    results, best_thetas, robustness_thetas = collect_combined_metrics(
        jailbreak_dir,
        robustness_dir,
        methods,
        base_metric,
        benchmarks,
        include_jailbreak_metrics,
        max_degree,
        jailbreak_step,
        robustness_step
    )
    
    if not results:
        print("Error: No results found. Check input directory structure.", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nFound {len(results)} model-method combinations.")
    
    # Build column list: jailbreak metrics + robustness benchmarks (no best_θ column)
    columns = include_jailbreak_metrics + benchmarks
    
    # Build display names
    column_display_names = {}
    for jb_metric in include_jailbreak_metrics:
        column_display_names[jb_metric] = JAILBREAK_METRICS.get(jb_metric, {}).get("display", jb_metric)
    for benchmark in benchmarks:
        column_display_names[benchmark] = benchmark
    
    # Print table to console
    print_table(results, methods, columns, column_display_names)
    
    # Save to TXT
    save_txt(results, output_file, methods, columns, column_display_names, base_metric, max_degree)
    
    # Save to CSV if requested
    if args.csv:
        csv_file = Path(args.csv)
        save_csv(results, csv_file, methods, columns, column_display_names)
    
    # Save to markdown if requested
    if args.markdown:
        markdown_file = Path(args.markdown)
        save_markdown(results, markdown_file, methods, columns, column_display_names, base_metric, max_degree)
    
    # Print summary of best thetas
    print("\nBest theta summary:")
    for (method, model), theta in sorted(best_thetas.items()):
        if theta is not None:
            print(f"  {method}/{model}: θ = {theta}°")
        else:
            print(f"  {method}/{model}: N/A")


if __name__ == "__main__":
    main()
