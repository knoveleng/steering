#!/bin/bash

# Evaluate jailbreak for each model
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

# Steering modes to evaluate
STEERING_MODES=(
    "basic"
    "adaptive"
    "selective"
)

# Input folder
INPUT_FOLDER="logs/perplexity"
OUTPUT_FOLDER="logs/jailbreak"

for MODEL_ID in "${MODELS[@]}"; do
    for STEERING_MODE in "${STEERING_MODES[@]}"; do
        echo "Evaluating $MODEL_ID with $STEERING_MODE steering"
        
        # Get base_nanme from model_id. Eg. google/gemma-2-2b-it -> gemma-2-2b-it
        base_name=$(echo $MODEL_ID | cut -d '/' -f 2)
        input_dir="$INPUT_FOLDER/$STEERING_MODE/$base_name"
        output_dir="$OUTPUT_FOLDER/$STEERING_MODE/$base_name"
        # python examples/eval_jailbreak.py \
        #     --input-dir $input_dir \
        #     --output-dir $output_dir \
        #     --evaluator substring llama_guard harmbench ngram_repetition language_consistency compression_ratio qwen3guard polyguard
        python examples/eval_jailbreak.py \
            --input-dir $input_dir \
            --output-dir $output_dir \
            --evaluator polyguard llm_judge
    done
done