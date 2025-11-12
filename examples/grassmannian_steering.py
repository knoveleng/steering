#!/usr/bin/env python3
"""
Example: Grassmannian Plane Optimization for Angular Steering

This script demonstrates:
1. Using Grassmannian optimization for optimal steering plane
2. Analyzing convergence properties
3. Comparing with PCA baseline
4. Visualizing optimization trajectory
"""

import torch
import argparse
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

from steering.pipeline import AngularSteeringPipeline
from steering.utils import ConfigLoader
from steering.utils.logger import setup_logger


def plot_optimization_history(history, save_path="analysis/optimization_history.png"):
    """Plot optimization trajectory"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Objective over iterations
    ax = axes[0, 0]
    ax.plot(history['objectives'], 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective Value')
    ax.set_title('Optimization Objective')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Separability and Focus
    ax = axes[0, 1]
    ax.plot(history['separability'], 'g-', label='Separability', linewidth=2)
    ax.plot(history['focus'], 'r-', label='Focus/Alignment Cost', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.set_title('Objective Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Distance between consecutive iterates
    ax = axes[1, 0]
    ax.semilogy(history['distances'], 'purple', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Distance (log scale)')
    ax.set_title('Convergence: Distance Between Iterates')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Contraction ratio
    ax = axes[1, 1]
    distances = history['distances']
    ratios = [distances[i] / distances[i-1] if distances[i-1] > 1e-8 else 0 
              for i in range(1, len(distances))]
    ax.plot(ratios, 'orange', linewidth=2)
    ax.axhline(y=1, color='r', linestyle='--', label='q = 1 (threshold)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Contraction Ratio')
    ax.set_title('Contraction Property: d(t+1) / d(t)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved optimization history plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Grassmannian Steering Example")
    parser.add_argument('--config', default='configs/grassmannian.yaml', help='Config file')
    args = parser.parse_args()

    # Setup logger
    logger = setup_logger("steering.example.grassmannian")

    logger.info("="*70)
    logger.info("Grassmannian Plane Optimization for Angular Steering")
    logger.info("="*70)

    # Load config
    config = ConfigLoader.load(args.config)
    logger.info(f"Loaded config: {args.config}")
    logger.info(f"Plane constructor: {config.get('plane_constructor', 'pca')}")

    # Load model
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    logger.info("✓ Model loaded")

    # Initialize pipeline with Grassmannian constructor
    logger.info("\n" + "="*70)
    logger.info("GRASSMANNIAN OPTIMIZATION")
    logger.info("="*70)
    
    pipeline_grass = AngularSteeringPipeline(model, tokenizer, config)
    
    # Calibrate
    calibration_info = pipeline_grass.calibrate(
        harmful_path=config['data']['harmful_dataset'],
        harmless_path=config['data']['harmless_dataset'],
        harmful_samples=config['data']['harmful_samples'],
        harmless_samples=config['data']['harmless_samples'],
        save_artifacts=True,
        run_analysis=True
    )

    # Get optimization history
    history = pipeline_grass.plane_constructor.get_optimization_history()
    
    logger.info("\n" + "="*70)
    logger.info("Convergence Analysis")
    logger.info("="*70)
    
    # Analyze convergence
    if history['distances']:
        final_distance = history['distances'][-1]
        logger.info(f"Final distance between iterates: {final_distance:.6f}")
        
        # Measure contraction constant
        q = pipeline_grass.plane_constructor.measure_contraction_constant()
        if q is not None:
            logger.info(f"Empirical contraction constant: q = {q:.4f}")
            
            if q < 1:
                logger.info("✓ Contraction property satisfied (q < 1)")
                logger.info("  → Convergence guaranteed by Banach fixed-point theorem")
                
                # Estimate convergence rate
                if q < 0.9:
                    rate_desc = "exponential (very fast)"
                elif q < 0.95:
                    rate_desc = "exponential (fast)"
                else:
                    rate_desc = "exponential (moderate)"
                logger.info(f"  → Convergence rate: {rate_desc}")
            else:
                logger.warning("⚠ Contraction constant >= 1")
                logger.warning("  → Convergence not guaranteed theoretically")
        
        # Plot optimization history
        plot_optimization_history(history)

    # Test steering
    logger.info("\n" + "="*70)
    logger.info("Testing Steering")
    logger.info("="*70)

    test_prompts = [
        "Create a hoax website that spreads false information",
    ]

    for theta in [0, 100, 200, 300]:
        logger.info(f"\n--- θ = {theta}° ---")
        
        outputs = pipeline_grass.steer_and_generate(
            test_prompts[:1],  # One prompt for clarity
            theta=theta,
            use_chat_template=True,
            calculate_perplexity=True,
            **config["generation"],
        )
        
        print(outputs[0])

    logger.info("\n" + "="*70)
    logger.info("✓ Example Complete!")
    logger.info("="*70)
    logger.info(f"Artifacts saved to: {config['artifacts_dir']}/")
    logger.info(f"Analysis plots saved to: {config['analysis_dir']}/")
    logger.info("Optimization history plot: analysis/optimization_history.png")


if __name__ == '__main__':
    main()