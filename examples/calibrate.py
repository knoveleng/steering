#!/usr/bin/env python3
"""
Example script to demonstrate the calibration process and the new 
session-based artifact saving functionality.

This script performs a full calibration and saves all artifacts to a 
timestamped session directory.
"""

import torch
import logging
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

from steering.pipeline import AngularSteeringPipeline
from steering.utils import ConfigLoader
from steering.utils.logger import setup_logger


def main():
    """Main function to run the calibration example"""
    parser = argparse.ArgumentParser(description="Calibration Example Script")
    parser.add_argument('--config', default='configs/default.yaml', help='Config file')
    args = parser.parse_args()

    # Setup logger
    logger = setup_logger("steering.examples.calibration")

    # Load config
    try:
        config = ConfigLoader.load(args.config)
        config['artifacts_dir'] = 'artifacts'
        config['analysis_dir'] = 'analysis'
        logger.info("✓ Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return

    logger.info("="*60)
    logger.info("Calibration Example")
    logger.info("="*60)
    logger.info(f"Model: {config['model']['name']}")

    # Load model and tokenizer
    try:
        logger.info("Loading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(
            config['model']['name'],
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
        logger.info("✓ Model and tokenizer loaded")
    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {e}")
        return

    # Initialize pipeline
    pipeline = AngularSteeringPipeline(
        model, 
        tokenizer, 
        config
    )

    # Perform calibration
    try:
        logger.info("Starting calibration...")
        pipeline.calibrate(
            harmful_path=config['data']['harmful_dataset'],
            harmless_path=config['data']['harmless_dataset'],
            harmful_samples=config['data']['harmful_samples'],
            harmless_samples=config['data']['harmless_samples'],
            save_artifacts=True,
            run_analysis=True
        )
        logger.info("✓ Calibration completed successfully")
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        return

    # Demonstrate the new session-based artifact saving
    logger.info("="*60)
    logger.info("Session-Based Artifact Saving")
    logger.info("="*60)

    # Get the session path from the artifacts manager
    session_path = pipeline.artifacts.session_dir
    logger.info(f"Session directory: {session_path}")
    logger.info(f"Session ID: {pipeline.artifacts.session_id}")

    # List the artifacts that were saved
    import os
    if os.path.exists(session_path):
        artifacts = os.listdir(session_path)
        logger.info("Saved artifacts:")
        for artifact in sorted(artifacts):
            logger.info(f"  • {artifact}")

    # Demonstrate standalone saving (optional - artifacts already saved during calibration)
    try:
        logger.info("\nDemonstrating standalone session saving...")
        standalone_session = pipeline.save_calibration_session(
            save_artifacts=False,  # Don't duplicate artifacts
            run_analysis=False     # Don't duplicate analysis
        )
        logger.info(f"✓ Standalone session reference: {standalone_session}")
    except RuntimeError as e:
        logger.warning(f"Standalone saving note: {e}")

    logger.info("="*60)
    logger.info("Calibration Summary")
    logger.info("="*60)
    stats = pipeline.analyzer.stats
    logger.info(f"Best layer: {stats['direction_statistics']['best_layer_name']}")
    logger.info(f"Feature separation: {stats['feature_alignment']['mean_separation']:.3f}")
    logger.info(f"Session timestamp: {pipeline.artifacts.session_id}")
    logger.info(f"All related artifacts grouped in: {session_path}")
    

if __name__ == '__main__':
    main()
