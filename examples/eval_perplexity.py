import json
import torch
import argparse
from pathlib import Path
from typing import List, Dict, Any
from steering.pipeline import AngularSteeringPipeline
from steering.utils import ConfigLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run angular steering experiments with configurable parameters"
    )
    
    parser.add_argument(
        "--calibration",
        type=str,
        default="./artifacts/calibration_basic",
        help="Path to pre-calibrated steering plane (default: ./artifacts/calibration_basic)"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="./data/advbench_test.json",
        help="Path to input data JSON file (default: ./data/advbench_test.json)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: eval/{model_name})"
    )
    
    parser.add_argument(
        "--degrees",
        type=int,
        nargs="+",
        default=None,
        help="Specific degrees to test (e.g., --degrees 0 45 90). If not specified, uses 0 to 360 by step 45"
    )
    
    parser.add_argument(
        "--degree-start",
        type=int,
        default=0,
        help="Start degree for range (default: 0)"
    )
    
    parser.add_argument(
        "--degree-end",
        type=int,
        default=360,
        help="End degree for range (default: 360)"
    )
    
    parser.add_argument(
        "--degree-step",
        type=int,
        default=60,
        help="Step size for degree range (default: 60)"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum generation length (default: 512)"
    )
    
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt to use for generation (default: None)"
    )
    
    parser.add_argument(
        "--no-perplexity",
        action="store_true",
        help="Disable perplexity calculation"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device map for model loading (default: auto)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["standard", "adaptive", "selective", "addition", "ablation"],
        help="Override steering mode from calibration"
    )
    
    parser.add_argument(
        "--model-id",
        type=str,
        help="Override model ID from config"
    )
    
    return parser.parse_args()


def load_data(data_file: str) -> List[str]:
    """Load prompts from JSON data file."""
    with open(data_file, "r") as f:
        data = json.load(f)
    return [point["prompt"] for point in data]


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
    
    print(f"Saved results to {output_file}")


def main():
    args = parse_args()
    
    # Load configuration from calibration directory. Config is stored in the calibration directory. It is a json file.
    print(f"Loading configuration from {args.calibration}")
    with open(Path(args.calibration) / "config.json", "r") as f:
        config = json.load(f)
    
    # Override model if specified
    if args.model_id:
        config['model']['name'] = args.model_id
        print(f"Model override: {args.model_id}")
    
    print(f"Loading model: {config['model']['name']}")
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'], 
        dtype=torch.bfloat16, 
        device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    # Initialize pipeline
    print("Initializing steering pipeline")
    pipeline = AngularSteeringPipeline(model, tokenizer, config)
    
    # Load pre-calibrated steering plane
    print(f"Loading calibration from {args.calibration}")
    pipeline.load_calibration(args.calibration, mode=args.mode)
    
    # Load dataset
    print(f"Loading data from {args.data}")
    prompts = load_data(args.data) # [:5]
    print(f"Loaded {len(prompts)} prompts")
    
    # Determine degrees to test
    if args.degrees:
        degrees = args.degrees
    else:
        degrees = range(args.degree_start, args.degree_end + 1, args.degree_step)
    
    print(f"Testing degrees: {list(degrees)}")
    
    # Set output directory
    if args.output_dir: 
        output_dir = Path(args.output_dir) / config['model']['name']
    else:
        output_dir = Path("eval") / config['model']['name']
    
    print(f"Output directory: {output_dir}")
    
    # Generate with perplexity
    for degree in degrees:
        print(f"\n{'='*60}")
        print(f"Processing degree: {degree}")
        print(f"{'='*60}")
        
        outputs = pipeline.steer_and_generate(
            prompts,
            theta=degree,  # Fixed: was 'theta' in original
            calculate_perplexity=not args.no_perplexity,
            **config["generation"]
        )
        
        # Calculate summary statistics
        metadata = calculate_metadata(outputs)
        if "average_perplexity" in metadata:
            print(f"Average perplexity: {metadata['average_perplexity']:.2f}")
        
        # Save results
        save_results(output_dir, degree, outputs, metadata)
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()