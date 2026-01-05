#!/bin/bash

# Evaluate robustness benchmarks for all models
# Uses single calibration per model with --mode override

# Models to evaluate (just the base names)
MODELS=(
    "Qwen2.5-1.5B-Instruct"
    "Qwen2.5-3B-Instruct"
    "Qwen2.5-7B-Instruct"
    "Llama-3.2-1B-Instruct"
    "Llama-3.2-3B-Instruct"
    "Llama-3.1-8B-Instruct"
    "gemma-2-9b-it"
    "gemma-2-2b-it"
)

# Steering modes to evaluate
STEERING_MODES=(
    # "addition"   # ignore because we can extract from best theta via standard mode
    "ablation"
    "standard"
    "adaptive"
    "selective"
)

# Benchmarks to evaluate
BENCHMARKS=(
    "tinyGSM8k"
    "tinyWinogrande"
    "tinyTruthfulQA"
    "tinyMMLU"
    "tinyAI2_arc"
)

# Folders
CALIBRATION_FOLDER="artifacts"
GENERATION_FOLDER="logs/robustness-generation"
EVALUATION_FOLDER="logs/robustness-evaluation"

# Degree range for steering
DEGREE_START=0
DEGREE_END=360
DEGREE_STEP=30

echo "=========================================="
echo "Robustness Evaluation for All Models"
echo "=========================================="

for model_name in "${MODELS[@]}"; do
    # Use single calibration per model (no mode suffix)
    calibration_path="$CALIBRATION_FOLDER/calibration_${model_name}"
    
    # Check if calibration exists
    if [ ! -d "$calibration_path" ]; then
        echo "Skipping $model_name: calibration not found at $calibration_path"
        continue
    fi
    
    for STEERING_MODE in "${STEERING_MODES[@]}"; do
        for BENCHMARK in "${BENCHMARKS[@]}"; do
            echo ""
            echo "=========================================="
            echo "Model: $model_name"
            echo "Mode: $STEERING_MODE"
            echo "Benchmark: $BENCHMARK"
            echo "=========================================="
            
            generation_dir="$GENERATION_FOLDER/$STEERING_MODE/$model_name/$BENCHMARK"
            evaluation_dir="$EVALUATION_FOLDER/$STEERING_MODE/$model_name/$BENCHMARK"
            
            # Step 1: Generate responses with mode override
            echo "Step 1: Generating responses..."
            
            # For ablation mode, only use one degree (it doesn't depend on theta)
            if [ "$STEERING_MODE" == "ablation" ]; then
                echo "  Ablation mode: using single degree (0)"
                python examples/generate_robustness.py \
                    --calibration $calibration_path \
                    --mode $STEERING_MODE \
                    --benchmark $BENCHMARK \
                    --output-dir $generation_dir \
                    --degree-start 0 \
                    --degree-end 0 \
                    --degree-step 10
            else
                python examples/generate_robustness.py \
                    --calibration $calibration_path \
                    --mode $STEERING_MODE \
                    --benchmark $BENCHMARK \
                    --output-dir $generation_dir \
                    --degree-start $DEGREE_START \
                    --degree-end $DEGREE_END \
                    --degree-step $DEGREE_STEP
            fi
            
            # Step 2: Evaluate responses
            echo "Step 2: Evaluating responses..."
            python examples/eval_robustness.py \
                --input-dir $generation_dir \
                --output-dir $evaluation_dir \
                --benchmark $BENCHMARK
            
            echo "Completed: $model_name / $STEERING_MODE / $BENCHMARK"
        done
    done
done

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "Generation outputs: $GENERATION_FOLDER"
echo "Evaluation outputs: $EVALUATION_FOLDER"
echo "=========================================="
