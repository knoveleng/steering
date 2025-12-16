#!/usr/bin/env python3
"""
Summarize robustness evaluation metrics into a CSV table.

This script aggregates accuracy metrics from JSON result files across methods, models,
and benchmarks, computing mean and standard deviation for each combination.
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


# Default methods (subdirectories) - ordered as specified
DEFAULT_METHODS = ["basic", "adaptive", "selective"]

# Default benchmarks
DEFAULT_BENCHMARKS = ["tinyAI2_arc", "tinyGSM8k", "tinyMMLU", "tinyTruthfulQA", "tinyWinogrande"]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Summarize robustness evaluation metrics into a CSV table",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to robustness-evaluation folder (e.g., logs/robustness-evaluation)"
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
        "--benchmarks",
        type=str,
        nargs="+",
        default=DEFAULT_BENCHMARKS,
        help="List of benchmarks to include in summary"
    )
    
    parser.add_argument(
        "--markdown",
        type=str,
        default=None,
        help="Output markdown file path for summary report (optional)"
    )
    
    return parser.parse_args()


def load_accuracy_scores(benchmark_dir: Path) -> List[float]:
    """
    Load all accuracy scores from JSON files in the benchmark directory.
    
    Args:
        benchmark_dir: Path to the benchmark folder containing JSON files
        
    Returns:
        List of accuracy scores from all JSON files
    """
    scores = []
    
    if not benchmark_dir.exists():
        return scores
    
    for json_file in sorted(benchmark_dir.glob("*.json")):
        try:
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
    
    return scores


def aggregate_scores(scores: List[float]) -> str:
    """
    Aggregate scores using mean and standard deviation.
    
    Args:
        scores: List of accuracy scores
        
    Returns:
        Formatted string with mean ± std
    """
    if not scores:
        return "N/A"
    
    mean = statistics.mean(scores)
    if len(scores) > 1:
        std = statistics.stdev(scores)
        return f"{mean:.4f} ± {std:.4f}"
    else:
        return f"{mean:.4f} ± 0.0000"


def collect_metrics(input_dir: Path, methods: List[str], benchmarks: List[str]) -> Dict[Tuple[str, str], Dict[str, str]]:
    """
    Collect and aggregate metrics for all models and methods.
    
    Args:
        input_dir: Path to robustness-evaluation folder
        methods: List of method names (subdirectories)
        benchmarks: List of benchmark names
        
    Returns:
        Dictionary mapping (method, model) -> {benchmark: aggregated_value}
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
            
            for benchmark in benchmarks:
                benchmark_dir = model_dir / benchmark
                scores = load_accuracy_scores(benchmark_dir)
                results[key][benchmark] = aggregate_scores(scores)
    
    return results


def print_table(results: Dict[Tuple[str, str], Dict[str, str]], methods: List[str], benchmarks: List[str]) -> None:
    """Print results as a formatted table to console."""
    if not results:
        print("No results to display.")
        return
    
    # Prepare headers
    headers = ["Model", "Method"] + benchmarks
    
    # Calculate column widths
    rows = []
    for (method, model), benchmark_values in sorted(results.items(), key=lambda x: (x[0][1], methods.index(x[0][0]) if x[0][0] in methods else 999)):
        row = [model, method] + [benchmark_values.get(b, "N/A") for b in benchmarks]
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


def save_txt(results: Dict[Tuple[str, str], Dict[str, str]], output_file: Path, methods: List[str], benchmarks: List[str]) -> None:
    """Save results to TXT file (same format as console output)."""
    if not results:
        print("No results to save.")
        return
    
    # Prepare headers
    headers = ["Model", "Method"] + benchmarks
    
    # Calculate column widths
    rows = []
    for (method, model), benchmark_values in sorted(results.items(), key=lambda x: (x[0][1], methods.index(x[0][0]) if x[0][0] in methods else 999)):
        row = [model, method] + [benchmark_values.get(b, "N/A") for b in benchmarks]
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


def save_csv(results: Dict[Tuple[str, str], Dict[str, str]], output_file: Path, methods: List[str], benchmarks: List[str]) -> None:
    """Save results to CSV file."""
    if not results:
        print("No results to save.")
        return
    
    headers = ["Model", "Method"] + benchmarks
    
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for (method, model), benchmark_values in sorted(results.items(), key=lambda x: (x[0][1], methods.index(x[0][0]) if x[0][0] in methods else 999)):
            row = [model, method] + [benchmark_values.get(b, "N/A") for b in benchmarks]
            writer.writerow(row)
    
    print(f"CSV saved to: {output_file}")


def save_markdown(results: Dict[Tuple[str, str], Dict[str, str]], output_file: Path, methods: List[str], benchmarks: List[str]) -> None:
    """Save results to markdown file."""
    if not results:
        print("No results to save.")
        return
    
    headers = ["Model", "Method"] + benchmarks
    
    with open(output_file, "w") as f:
        # Write title
        f.write("# Robustness Evaluation Summary\n\n")
        
        # Write table header
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---" for _ in headers]) + "|\n")
        
        # Write rows
        for (method, model), benchmark_values in sorted(results.items(), key=lambda x: (x[0][1], methods.index(x[0][0]) if x[0][0] in methods else 999)):
            row = [model, method] + [benchmark_values.get(b, "N/A") for b in benchmarks]
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
    
    # Get methods and benchmarks from args
    methods = args.methods
    benchmarks = args.benchmarks
    
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    print(f"Methods: {methods}")
    print(f"Benchmarks: {benchmarks}")
    
    # Collect metrics
    results = collect_metrics(input_dir, methods, benchmarks)
    
    if not results:
        print("Error: No results found. Check input directory structure.", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nFound {len(results)} model-method combinations.")
    
    # Print table to console
    print_table(results, methods, benchmarks)
    
    # Save to TXT (default)
    save_txt(results, output_file, methods, benchmarks)
    
    # Save to CSV if requested
    if args.csv:
        csv_file = Path(args.csv)
        save_csv(results, csv_file, methods, benchmarks)
    
    # Save to markdown if requested
    if args.markdown:
        markdown_file = Path(args.markdown)
        save_markdown(results, markdown_file, methods, benchmarks)


if __name__ == "__main__":
    main()
