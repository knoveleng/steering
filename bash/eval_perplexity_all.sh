#!/bin/bash
# Evaluation script for perplexity experiments across multiple models and steering modes
# Uses single calibration per model with --mode override

set -e  # Exit on error
set -u  # Exit on undefined variable

# Configuration
DATA_FILE="./data/advbench_test.json"
DEGREE_START=0
DEGREE_END=360
DEGREE_STEP=10
GPU_MEMORY_UTILIZATION=0.85
BASE_OUTPUT_DIR="./logs/perplexity"
CALIBRATION_FOLDER="./artifacts"

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

    for model_name in "${MODELS[@]}"; do
        # Use single calibration per model (no mode suffix)
        calibration_dir="${CALIBRATION_FOLDER}/calibration_${model_name}"
        
        for steering_mode in "${STEERING_MODES[@]}"; do
            current=$((current + 1))
            
            # Use mode name directly for output folder
            output_dir="${BASE_OUTPUT_DIR}/${steering_mode}"
            
            # Log progress
            echo ""
            log_info "[$current/$total_combinations] Evaluating $model_name with $steering_mode steering"
            log_info "  Calibration: $calibration_dir"
            log_info "  Mode: $steering_mode"
            log_info "  Output: $output_dir"
            
            # Check if calibration exists
            if ! check_calibration_exists "$calibration_dir"; then
                log_warning "Skipping $model_name/$steering_mode (calibration not found)"
                skip_count=$((skip_count + 1))
                continue
            fi
            
            # For ablation mode, only use one degree (it doesn't depend on theta)
            if [ "$steering_mode" == "ablation" ]; then
                log_info "  Ablation mode: using single degree (0)"
                if python examples/eval_perplexity_vllm.py \
                    --data "$DATA_FILE" \
                    --calibration "$calibration_dir" \
                    --mode "$steering_mode" \
                    --degree-start 0 \
                    --degree-end 0 \
                    --degree-step 10 \
                    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
                    --output-dir "$output_dir"; then
                    log_success "Completed $model_name/$steering_mode"
                    success_count=$((success_count + 1))
                else
                    log_error "Failed $model_name/$steering_mode"
                    error_count=$((error_count + 1))
                fi
            else
                # Standard evaluation with full degree range
                if python examples/eval_perplexity_vllm.py \
                    --data "$DATA_FILE" \
                    --calibration "$calibration_dir" \
                    --mode "$steering_mode" \
                    --degree-start "$DEGREE_START" \
                    --degree-end "$DEGREE_END" \
                    --degree-step "$DEGREE_STEP" \
                    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
                    --output-dir "$output_dir"; then
                    log_success "Completed $model_name/$steering_mode"
                    success_count=$((success_count + 1))
                else
                    log_error "Failed $model_name/$steering_mode"
                    error_count=$((error_count + 1))
                fi
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