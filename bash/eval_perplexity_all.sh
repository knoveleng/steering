#!/bin/bash
# Evaluation script for perplexity experiments across multiple models and steering modes

set -e  # Exit on error
set -u  # Exit on undefined variable

# Configuration
DATA_FILE="./data/advbench_test.json"
DEGREE_START=0
DEGREE_END=360
DEGREE_STEP=10
GPU_MEMORY_UTILIZATION=0.85
BASE_OUTPUT_DIR="./logs/perplexity"

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
    "basic"     # standard steering
    "adaptive"
    "selective"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper function to print colored messages
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Validate calibration directory exists
check_calibration_exists() {
    local calibration_dir=$1
    if [ ! -d "$calibration_dir" ]; then
        log_warning "Calibration directory not found: $calibration_dir"
        return 1
    fi
    if [ ! -f "$calibration_dir/config.json" ]; then
        log_warning "config.json not found in: $calibration_dir"
        return 1
    fi
    return 0
}

# Main evaluation loop
main() {
    log_info "Starting batch perplexity evaluation"
    log_info "Models: ${#MODELS[@]}"
    log_info "Steering modes: ${#STEERING_MODES[@]}"
    log_info "Output directory: $BASE_OUTPUT_DIR"
    echo ""

    local total_combinations=$((${#MODELS[@]} * ${#STEERING_MODES[@]}))
    local current=0
    local success_count=0
    local skip_count=0
    local error_count=0

    for model in "${MODELS[@]}"; do
        # Extract model name from model path. E.g google/gemma-2-9b-it -> gemma-2-9b-it
        model_name=$(basename $model)
        echo "Model name: $model_name"
        
        for steering_mode in "${STEERING_MODES[@]}"; do
            current=$((current + 1))
            
            # Build paths
            calibration_dir="./artifacts/calibration_${model_name}_${steering_mode}"
            output_dir="${BASE_OUTPUT_DIR}/${steering_mode}"
            
            # Log progress
            echo ""
            log_info "[$current/$total_combinations] Evaluating $model with $steering_mode steering"
            log_info "  Calibration: $calibration_dir"
            log_info "  Output: $output_dir"
            
            # Check if calibration exists
            if ! check_calibration_exists "$calibration_dir"; then
                log_warning "Skipping $model/$steering_mode (calibration not found)"
                skip_count=$((skip_count + 1))
                continue
            fi
            
            # Run evaluation - If you wanna use naive transformers, please use eval_perplexity.py
            if python examples/eval_perplexity_vllm.py \
                --data "$DATA_FILE" \
                --calibration "$calibration_dir" \
                --degree-start "$DEGREE_START" \
                --degree-end "$DEGREE_END" \
                --degree-step "$DEGREE_STEP" \
                --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
                --output-dir "$output_dir"; then
                log_success "Completed $model/$steering_mode"
                success_count=$((success_count + 1))
            else
                log_error "Failed $model/$steering_mode"
                error_count=$((error_count + 1))
                # Continue with next combination even if one fails
            fi
        done
    done
    
    # Summary
    echo ""
    log_info "Batch evaluation complete!"
    log_info "  Successful: $success_count"
    log_info "  Skipped: $skip_count"
    log_info "  Errors: $error_count"
    log_info "  Total: $total_combinations"
}

# Run main function
main