#!/bin/bash
# Calibration script for all models using selective mode
# Creates calibrations with names like: calibration_gemma-2-2b-it

set -e

MODELS=(
    "google/gemma-2-2b-it"
    "google/gemma-2-9b-it"
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
)

MODE="selective" # We use selective mode for calibration, because it saves layer_mask 
CONFIG="configs/selective.yaml"

echo "============================================================"
echo "Running Calibration for All Models"
echo "Mode: ${MODE}"
echo "Config: ${CONFIG}"
echo "============================================================"

for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT=$(echo "$MODEL" | rev | cut -d'/' -f1 | rev)
    
    echo ""
    echo "============================================================"
    echo "Calibrating: ${MODEL}"
    echo "Short name: ${MODEL_SHORT}"
    echo "============================================================"
    
    python examples/calibrate.py \
        --config "${CONFIG}" \
        --model-id "${MODEL}" \
        --mode "${MODE}" \
        --no-analysis
    
    echo "✓ Completed: ${MODEL_SHORT}"
done

echo ""
echo "============================================================"
echo "All calibrations completed!"
echo "Calibrations saved to: artifacts/"
echo "============================================================"
