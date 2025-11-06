# Basic Steering
python examples/eval_perplexity.py \
    --config ./configs/default.yaml \
    --data ./data/advbench_test.json \
    --calibration ./artifacts/calibration_basic \
    --degree-start 0 \
    --degree-end 360 \
    --degree-step 30 \
    --output-dir ./eval/basic

# Adaptive Steering
python examples/eval_perplexity.py \
    --config ./configs/adaptive.yaml \
    --data ./data/advbench_test.json \
    --calibration ./artifacts/calibration_basic \
    --degree-start 0 \
    --degree-end 360 \
    --degree-step 30 \
    --output-dir ./eval/adaptive

# Selective Steering
python examples/eval_perplexity.py \
    --config ./configs/selective.yaml \
    --data ./data/advbench_test.json \
    --calibration ./artifacts/calibration_selective \
    --degree-start 0 \
    --degree-end 360 \
    --degree-step 30 \
    --output-dir ./eval/selective

# Selective Steering with Grassmann
python examples/eval_perplexity.py \
    --config ./configs/grassmannian.yaml \
    --data ./data/advbench_test.json \
    --calibration ./artifacts/calibration_grassmann \
    --degree-start 0 \
    --degree-end 360 \
    --degree-step 30 \
    --output-dir ./eval/grassmann

