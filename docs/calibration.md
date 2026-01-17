# Calibration Artifacts Documentation

This document describes the structure and contents of calibration artifacts saved by the steering pipeline.

## Calibration Directory Structure

When calibration is run, artifacts are saved to:
```
artifacts/calibration_{model_short_name}_{timestamp}/
├── activations.pt     # Raw activation data
├── config.json        # Configuration used for calibration
├── directions.pt      # Computed steering directions
├── metadata.json      # Session metadata
└── plane.pt           # Steering plane and basis vectors
```

## File Contents

### `plane.pt`

Contains the steering plane data:

```python
{
    "basis": (b1, b2),           # Tuple of basis vectors [L, D]
    "projections": ...,          # Candidate projections onto plane
    "plane_type": "pca",         # Plane construction method
    "extra_info": {
        "target_layers": [...],   # List of layer names (ALL modes)
        "layer_steering_mask": {  # Only for selective mode
            "model.layers.0.input_layernorm": True,
            "model.layers.0.post_attention_layernorm": False,
            ...
        }
    },
    "metadata": {...}
}
```

**Key fields in `extra_info`:**
- `target_layers`: Always present. List of all layers used for steering.
- `layer_steering_mask`: Only for selective mode. Dict mapping layer names to True/False indicating whether to steer.

### `config.json`

Configuration used during calibration:

```json
{
    "model": {
        "name": "Qwen/Qwen2.5-3B-Instruct",
        "device": "cuda",
        "dtype": "bfloat16"
    },
    "steering": {
        "mode": "selective"
    },
    ...
}
```

### `directions.pt`

Steering directions computed during calibration:

```python
{
    "directions": {...},      # Per-layer directions
    "best_layer": "...",      # Best layer for steering
    "candidates": [...]       # Direction candidates
}
```

### `activations.pt`

Raw activation data:

```python
{
    "harmful_activations": {...},   # Activations for harmful prompts
    "harmless_activations": {...}   # Activations for harmless prompts
}
```

### `metadata.json`

Session metadata:

```json
{
    "session_id": "20231218_100000",
    "model_name": "Qwen2.5-3B-Instruct",
    "steering_mode": "selective",
    "timestamp": "2023-12-18T10:00:00"
}
```

## Loading Calibration

### For vLLM

```python
from ui.utils import load_calibration
from steering.vllm_steering import SteeringLLM

calibration = load_calibration("artifacts/calibration_Qwen2.5-3B-Instruct/")

# Override mode if needed
calibration['mode'] = 'standard'

llm = SteeringLLM.from_calibration(calibration)
```

### For Transformers

```python
from steering.pipeline import AngularSteeringPipeline

pipeline = AngularSteeringPipeline(model, tokenizer, config)
pipeline.load_calibration("artifacts/calibration_Qwen2.5-3B-Instruct/", mode='standard')
```

## Mode Override

You can use calibration from one mode with another mode at runtime:

```bash
# Use selective calibration with standard mode
python examples/eval_perplexity_vllm.py \
    --calibration artifacts/calibration_Qwen2.5-3B-Instruct/ \
    --mode standard
```

This works because:
1. `target_layers` is stored for ALL calibration modes
2. The steering operator uses the appropriate formula based on runtime mode
3. `layer_steering_mask` is only used when mode is `selective`
