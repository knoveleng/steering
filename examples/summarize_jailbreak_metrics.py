#!/usr/bin/env python3
"""
Summarize jailbreak evaluation metrics into a CSV table.

This script aggregates metrics from JSON result files across methods and models,
applying appropriate aggregation functions based on metric type.
"""

import argparse
import json
import os
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import statistics


# Default metrics and their aggregation methods
DEFAULT_METRICS = {
    "harmbench": "max",
    "llama_guard": "max",
    "substring": "min",
    "compression_ratio": "mean_std",
    "language_consistency": "mean_std",
    "ngram_repetition": "mean_std",
    "qwen3guard": "max",
    "polyguard": "max",
    "llm_judge": "max",
}

# Display names with arrows (↑ = higher is better, ↓ = lower is better)
METRIC_DISPLAY_NAMES = {
    "harmbench": "harmbench ↑",
    "llama_guard": "llama_guard ↑",
    "substring": "substring ↓",
    "compression_ratio": "compression_ratio ↑",
    "language_consistency": "language_consistency ↑",
    "ngram_repetition": "ngram_repetition ↓",
    "qwen3guard": "qwen3guard ↑",
    "polyguard": "polyguard ↑",
    "llm_judge": "llm_judge ↑",
}

# Default methods (subdirectories) - ordered as specified
DEFAULT_METHODS = ["basic", "adaptive", "selective"]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Summarize jailbreak evaluation metrics into a CSV table",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to jailbreak folder (e.g., logs/jailbreak)"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output TXT file path (default: {input_dir}_summary.txt)"
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Output CSV file path (optional)"
    )
    
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=DEFAULT_METHODS,
        help="List of methods (subdirectories) to include"
    )
    
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=list(DEFAULT_METRICS.keys()),
        help="List of metrics to include in summary"
    )
    
    parser.add_argument(
        "--markdown",
        type=str,
        default=None,
        help="Output markdown file path for summary report (optional)"
    )
    
    return parser.parse_args()


def load_metric_scores(metric_dir: Path, metric_name: str) -> List[float]:
    """
    Load all scores for a metric from JSON files in the metric directory.
    
    Args:
        metric_dir: Path to the metric folder containing JSON files
        metric_name: Name of the metric to extract from metadata
        
    Returns:
        List of scores from all JSON files
    """
    scores = []
    
    if not metric_dir.exists():
        return scores
    
    for json_file in sorted(metric_dir.glob("*.json")):
        try:
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


def aggregate_scores(scores: List[float], method: str) -> str:
    """
    Aggregate scores using the specified method.
    
    Args:
        scores: List of metric scores
        method: Aggregation method ('max', 'min', 'mean_std')
        
    Returns:
        Formatted string with aggregated value
    """
    if not scores:
        return "N/A"
    
    if method == "max":
        return f"{max(scores):.4f}"
    elif method == "min":
        return f"{min(scores):.4f}"
    elif method == "mean_std":
        mean = statistics.mean(scores)
        if len(scores) > 1:
            std = statistics.stdev(scores)
            return f"{mean:.4f} ± {std:.4f}"
        else:
            return f"{mean:.4f} ± 0.0000"
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def collect_metrics(input_dir: Path, methods: List[str], metrics: Dict[str, str]) -> Dict[Tuple[str, str], Dict[str, str]]:
    """
    Collect and aggregate metrics for all models and methods.
    
    Args:
        input_dir: Path to jailbreak folder
        methods: List of method names (subdirectories)
        metrics: Dictionary mapping metric name to aggregation method
        
    Returns:
        Dictionary mapping (method, model) -> {metric: aggregated_value}
    """
    results = {}
    
    for method in methods:
        method_dir = input_dir / method
        if not method_dir.exists():
            print(f"Warning: Method directory not found: {method_dir}", file=sys.stderr)
            continue
        
        # Get all model directories
        model_dirs = [d for d in method_dir.iterdir() if d.is_dir()]
        
        for model_dir in sorted(model_dirs):
            model_name = model_dir.name
            key = (method, model_name)
            results[key] = {}
            
            for metric_name, agg_method in metrics.items():
                metric_dir = model_dir / metric_name
                scores = load_metric_scores(metric_dir, metric_name)
                results[key][metric_name] = aggregate_scores(scores, agg_method)
    
    return results


def print_table(results: Dict[Tuple[str, str], Dict[str, str]], methods: List[str], metrics: List[str]) -> None:
    """Print results as a formatted table to console."""
    if not results:
        print("No results to display.")
        return
    
    # Prepare headers
    headers = ["Model", "Method"] + [METRIC_DISPLAY_NAMES.get(m, m) for m in metrics]
    
    # Calculate column widths
    rows = []
    for (method, model), metric_values in sorted(results.items(), key=lambda x: (x[0][1], methods.index(x[0][0]) if x[0][0] in methods else 999)):
        row = [model, method] + [metric_values.get(m, "N/A") for m in metrics]
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


def save_txt(results: Dict[Tuple[str, str], Dict[str, str]], output_file: Path, methods: List[str], metrics: List[str]) -> None:
    """Save results to TXT file (same format as console output)."""
    if not results:
        print("No results to save.")
        return
    
    # Prepare headers
    headers = ["Model", "Method"] + [METRIC_DISPLAY_NAMES.get(m, m) for m in metrics]
    
    # Calculate column widths
    rows = []
    for (method, model), metric_values in sorted(results.items(), key=lambda x: (x[0][1], methods.index(x[0][0]) if x[0][0] in methods else 999)):
        row = [model, method] + [metric_values.get(m, "N/A") for m in metrics]
        rows.append(row)
    
    # Calculate max width for each column
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    
    with open(output_file, "w") as f:
        # Write header
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
        f.write("=" * len(header_line) + "\n")
        f.write(header_line + "\n")
        f.write("-" * len(header_line) + "\n")
        
        # Write rows (grouped by model with separators)
        prev_model = None
        for row in rows:
            current_model = row[0]
            if prev_model is not None and current_model != prev_model:
                f.write("-" * len(header_line) + "\n")
            f.write(" | ".join(str(cell).ljust(w) for cell, w in zip(row, widths)) + "\n")
            prev_model = current_model
        
        f.write("=" * len(header_line) + "\n")
    
    print(f"TXT saved to: {output_file}")


def save_csv(results: Dict[Tuple[str, str], Dict[str, str]], output_file: Path, methods: List[str], metrics: List[str]) -> None:
    """Save results to CSV file."""
    if not results:
        print("No results to save.")
        return
    
    headers = ["Model", "Method"] + [METRIC_DISPLAY_NAMES.get(m, m) for m in metrics]
    
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for (method, model), metric_values in sorted(results.items(), key=lambda x: (x[0][1], methods.index(x[0][0]) if x[0][0] in methods else 999)):
            row = [model, method] + [metric_values.get(m, "N/A") for m in metrics]
            writer.writerow(row)
    
    print(f"CSV saved to: {output_file}")


def save_markdown(results: Dict[Tuple[str, str], Dict[str, str]], output_file: Path, methods: List[str], metrics: List[str]) -> None:
    """Save results to markdown file."""
    if not results:
        print("No results to save.")
        return
    
    headers = ["Model", "Method"] + [METRIC_DISPLAY_NAMES.get(m, m) for m in metrics]
    
    with open(output_file, "w") as f:
        # Write title
        f.write("# Jailbreak Evaluation Summary\n\n")
        
        # Write table header
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---" for _ in headers]) + "|\n")
        
        # Write rows
        for (method, model), metric_values in sorted(results.items(), key=lambda x: (x[0][1], methods.index(x[0][0]) if x[0][0] in methods else 999)):
            row = [model, method] + [metric_values.get(m, "N/A") for m in metrics]
            f.write("| " + " | ".join(str(cell) for cell in row) + " |\n")
    
    print(f"Markdown saved to: {output_file}")


def main() -> None:
    """Main function."""
    args = parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Set default output file (TXT)
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = input_dir.parent / f"{input_dir.name}_summary.txt"
    
    # Get methods and metrics from args
    methods = args.methods
    metric_names = args.metrics
    
    # Build metrics dict with aggregation methods
    metrics = {}
    for m in metric_names:
        if m in DEFAULT_METRICS:
            metrics[m] = DEFAULT_METRICS[m]
        else:
            # Default to 'max' for unknown metrics
            print(f"Warning: Unknown metric '{m}', using 'max' aggregation", file=sys.stderr)
            metrics[m] = "max"
    
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    print(f"Methods: {methods}")
    print(f"Metrics: {list(metrics.keys())}")
    
    # Collect metrics
    results = collect_metrics(input_dir, methods, metrics)
    
    if not results:
        print("Error: No results found. Check input directory structure.", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nFound {len(results)} model-method combinations.")
    
    # Print table to console
    print_table(results, methods, metric_names)
    
    # Save to TXT (default)
    save_txt(results, output_file, methods, metric_names)
    
    # Save to CSV if requested
    if args.csv:
        csv_file = Path(args.csv)
        save_csv(results, csv_file, methods, metric_names)
    
    # Save to markdown if requested
    if args.markdown:
        markdown_file = Path(args.markdown)
        save_markdown(results, markdown_file, methods, metric_names)


if __name__ == "__main__":
    main()

