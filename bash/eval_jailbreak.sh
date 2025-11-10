#!/bin/bash

# Evaluate jailbreak for each model
MODEL_IDS=("Qwen/Qwen2.5-3B-Instruct" "meta-llama/Llama-3.2-3B-Instruct")
INPUT_DIR="eval/selective"
OUTPUT_DIR="eval/jailbreak"

for MODEL_ID in "${MODEL_IDS[@]}"; do
    echo "Evaluating $MODEL_ID"
    input_dir="$INPUT_DIR/$MODEL_ID"
    output_dir="$OUTPUT_DIR/$MODEL_ID"
    python examples/eval_jailbreak.py \
        --input-dir $input_dir \
        --output-dir $output_dir
done