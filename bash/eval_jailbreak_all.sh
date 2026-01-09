#!/bin/bash

# Evaluate jailbreak for each model
# This script uses perplexity logs as input (doesn't need calibration directly)

# Models to evaluate (just the base names)
MODELS=(
    "gemma-2-2b-it"
    "gemma-2-9b-it"
    "Llama-3.2-1B-Instruct"
    "Llama-3.2-3B-Instruct"
    "Llama-3.1-8B-Instruct"
    "Qwen2.5-1.5B-Instruct"
    "Qwen2.5-3B-Instruct"
    "Qwen2.5-7B-Instruct"
)

# Steering modes to evaluate
STEERING_MODES=(
    # "addition"   # ignore because we can extract from best theta via standard mode
    "ablation"
    "standard"
    "adaptive"
    "selective"
)

# Input/Output folders
INPUT_FOLDER="logs/perplexity"
OUTPUT_FOLDER="logs/jailbreak"

# Evaluators to use
EVALUATORS="substring llama_guard harmbench polyguard llm_judge ngram_repetition language_consistency compression_ratio"

echo "=========================================="
echo "Jailbreak Evaluation for All Models"
echo "=========================================="

for model_name in "${MODELS[@]}"; do
    for STEERING_MODE in "${STEERING_MODES[@]}"; do
        echo ""
        echo "=========================================="
        echo "Model: $model_name"
        echo "Mode: $STEERING_MODE"
        echo "=========================================="
        
        input_dir="$INPUT_FOLDER/$STEERING_MODE/$model_name"
        output_dir="$OUTPUT_FOLDER/$STEERING_MODE/$model_name"
        
        # Check if input directory exists
        if [ ! -d "$input_dir" ]; then
            echo "Skipping $model_name/$STEERING_MODE: input not found at $input_dir"
            continue
        fi
        
        python examples/eval_jailbreak.py \
            --input-dir $input_dir \
            --output-dir $output_dir \
            --evaluator $EVALUATORS
        
        echo "Completed: $model_name / $STEERING_MODE"
    done
done

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "Output: $OUTPUT_FOLDER"
echo "=========================================="