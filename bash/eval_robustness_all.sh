#!/bin/bash

# Evaluate robustness benchmarks for all models
# This script runs both generation and evaluation for robustness benchmarks

# Models to evaluate
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
# MODELS=(
#     "meta-llama/Llama-3.1-8B-Instruct"
#     "meta-llama/Llama-3.2-1B-Instruct"
#     "Qwen/Qwen2.5-1.5B-Instruct"
#     "Qwen/Qwen2.5-3B-Instruct"
#     "Qwen/Qwen2.5-7B-Instruct"
#     "google/gemma-2-2b-it"
# )

# Steering modes to evaluate
STEERING_MODES=(
    # "basic"
    "adaptive"
    "selective"
)

# Benchmarks to evaluate
BENCHMARKS=(
    "tinyMMLU"
    "tinyGSM8k"
    "tinyAI2_arc"
    "tinyWinogrande"
    "tinyTruthfulQA"
)

# Folders
CALIBRATION_FOLDER="artifacts"
GENERATION_FOLDER="logs/robustness-generation"
EVALUATION_FOLDER="logs/robustness-evaluation"

# Degree range for steering
DEGREE_START=0
DEGREE_END=360
DEGREE_STEP=10

echo "=========================================="
echo "Robustness Evaluation for All Models"
echo "=========================================="

for MODEL_ID in "${MODELS[@]}"; do
    # Get base_name from model_id. Eg. google/gemma-2-2b-it -> gemma-2-2b-it
    base_name=$(echo $MODEL_ID | cut -d '/' -f 2)
    
    for STEERING_MODE in "${STEERING_MODES[@]}"; do
        calibration_path="$CALIBRATION_FOLDER/calibration_${base_name}_${STEERING_MODE}"
        
        # Check if calibration exists
        if [ ! -d "$calibration_path" ]; then
            echo "Skipping $MODEL_ID ($STEERING_MODE): calibration not found at $calibration_path"
            continue
        fi
        
        for BENCHMARK in "${BENCHMARKS[@]}"; do
            echo ""
            echo "=========================================="
            echo "Model: $MODEL_ID"
            echo "Mode: $STEERING_MODE"
            echo "Benchmark: $BENCHMARK"
            echo "=========================================="
            
            generation_dir="$GENERATION_FOLDER/$STEERING_MODE/$base_name/$BENCHMARK"
            evaluation_dir="$EVALUATION_FOLDER/$STEERING_MODE/$base_name/$BENCHMARK"
            
            # Step 1: Generate responses
            echo "Step 1: Generating responses..."
            python examples/generate_robustness.py \
                --calibration $calibration_path \
                --benchmark $BENCHMARK \
                --output-dir $generation_dir \
                --degree-start $DEGREE_START \
                --degree-end $DEGREE_END \
                --degree-step $DEGREE_STEP
            
            # Step 2: Evaluate responses
            echo "Step 2: Evaluating responses..."
            python examples/eval_robustness.py \
                --input-dir $generation_dir \
                --output-dir $evaluation_dir \
                --benchmark $BENCHMARK
            
            echo "Completed: $MODEL_ID / $STEERING_MODE / $BENCHMARK"
        done
    done
done

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "Generation outputs: $GENERATION_FOLDER"
echo "Evaluation outputs: $EVALUATION_FOLDER"
echo "=========================================="
