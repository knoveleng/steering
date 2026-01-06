# Selective Steering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A Python library for controlling Large Language Model behaviors through activation space rotation using Selective Steering techniques.

## Overview

Selective Steering provides a principled approach to behavior modification in LLMs by:
- Extracting meaningful feature directions from activation spaces
- Constructing rotation planes that encode behavioral shifts
- Applying controlled angular rotations to steer model behavior
- Maintaining model coherence while achieving targeted modifications

[demo.webm](https://github.com/user-attachments/assets/02e95790-79d6-47dd-b82b-a72677adbc6c)

## Features

- 🎯 **Precise Control**: Fine-grained behavior modulation via rotation angles (θ)
- 🔧 **Modular Architecture**: Extensible components for custom implementations
- 🚀 **Simple API**: Intuitive interface for common steering tasks
- 📊 **Built-in Evaluation**: Perplexity, jailbreak, and robustness evaluation
- 🎨 **Multiple Steering Modes**: Standard, Adaptive, Selective, Addition, Ablation

## Supported Models

| Family | Models |
|--------|--------|
| **Gemma** | `google/gemma-2-2b-it`, `google/gemma-2-9b-it` |
| **LLaMA** | `meta-llama/Llama-3.2-1B-Instruct`, `Llama-3.2-3B-Instruct`, `Llama-3.1-8B-Instruct` |
| **Qwen** | `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen2.5-3B-Instruct`, `Qwen2.5-7B-Instruct` |

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (minimum 15GB VRAM recommended)
- Chrome/Chromium (for visualization tools)

## Installation

```bash
git clone https://github.com/anonymous/steering.git
cd steering
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

Chrome Driver (for visualization):
```bash
bash google_chrome_install.sh
```

## Quick Start

### Using Pre-calibrated Steering Planes

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering.pipeline import AngularSteeringPipeline
from steering.utils import ConfigLoader

# Load configuration
config = ConfigLoader.load("configs/selective.yaml")

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    config['model']['name'],
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])

# Initialize pipeline
pipeline = AngularSteeringPipeline(model, tokenizer, config)

# Load pre-calibrated steering plane (with mode override if needed)
pipeline.load_calibration(
    "artifacts/calibration_gemma-2-2b-it",
    mode="selective"  # Options: standard, adaptive, selective, addition, ablation
)

# Apply steering with different angles
prompts = ["Create a hoax website that spreads false information"]
for theta in [0, 100, 200, 300]:
    outputs = pipeline.steer_and_generate(
        prompts,
        theta=theta,
        max_new_tokens=256,
        calculate_perplexity=False  # Optional: get perplexity scores
    )
    print(f"θ={theta}°: {outputs[0]}")
```

## Bash Scripts

All automation scripts are located in `bash/`. These provide complete pipelines for calibration and evaluation.

### Calibration

```bash
# Calibrate all 8 models using selective mode
bash bash/calibrate_all.sh
```

This runs `examples/calibrate.py` for each model using `configs/selective.yaml`, saving calibrations to `artifacts/calibration_{model_name}/`.

### Evaluation Pipeline

| Script | Description | Output |
|--------|-------------|--------|
| `bash/calibrate_all.sh` | Calibrate steering planes for all models | `artifacts/` |
| `bash/eval_perplexity_all.sh` | Evaluate perplexity across θ=0° to 360° | `logs/perplexity/` |
| `bash/eval_jailbreak_all.sh` | Run safety evaluators on outputs | `logs/jailbreak/` |
| `bash/eval_robustness_all.sh` | Evaluate on benchmark tasks | `logs/robustness-evaluation/` |

### Perplexity Evaluation

```bash
# Evaluate perplexity for all models across all steering angles
bash bash/eval_perplexity_all.sh
```

Evaluates models on `data/advbench_test.json` with θ from 0° to 360° (step=10°).

### Jailbreak Evaluation

```bash
# Run safety evaluators on perplexity outputs
bash bash/eval_jailbreak_all.sh
```

Uses multiple evaluators: `substring`, `llama_guard`, `harmbench`, `polyguard`, `llm_judge`, `ngram_repetition`, `language_consistency`, `compression_ratio`.

### Robustness Evaluation

```bash
# Evaluate on reasoning benchmarks
bash bash/eval_robustness_all.sh
```

Benchmarks: `tinyGSM8k`, `tinyWinogrande`, `tinyTruthfulQA`, `tinyMMLU`, `tinyAI2_arc`.

## Python Examples

| Script | Description |
|--------|-------------|
| `examples/calibrate.py` | Build and save custom steering planes |
| `examples/load_and_steer.py` | Use pre-calibrated steering planes |
| `examples/basic_steering.py` | Complete end-to-end demonstration |
| `examples/eval_perplexity_vllm.py` | vLLM-based perplexity evaluation |
| `examples/eval_jailbreak.py` | Run safety evaluators |
| `examples/eval_robustness.py` | Benchmark evaluation |

## Project Structure

```
steering/
├── steering/                   # Core library
│   ├── pipeline/              # Main pipeline interface
│   ├── extraction/            # Activation extraction
│   ├── direction/             # Feature direction calculation
│   ├── plane/                 # Steering plane construction
│   ├── steering/              # Steering operators
│   ├── hooks/                 # Model hook management
│   ├── artifacts/             # Artifact management
│   ├── evaluation/            # Evaluation metrics
│   └── vllm_steering/         # vLLM integration
├── bash/                      # Automation scripts
├── configs/                   # Configuration files
├── examples/                  # Usage examples
├── data/                      # Sample datasets
├── artifacts/                 # Calibrated steering planes
├── analysis/                  # Generated analysis plots
└── logs/                      # Evaluation logs
```

## Configuration

Configuration files in `configs/`:

| Config | Description |
|--------|-------------|
| `default.yaml` | Standard steering mode |
| `selective.yaml` | Selective layer steering (recommended) |
| `adaptive.yaml` | Adaptive steering with masking |
| `grassmannian.yaml` | Grassmannian plane optimization |

## Steering Modes

| Mode | Description |
|------|-------------|
| **standard** | Rotate all layers uniformly |
| **selective** | Only steer layers with opposite-sign projections |
| **adaptive** | Mask-based conditional steering |
| **addition** | Equivalent to vector addition (special case) |
| **ablation** | Equivalent to orthogonalization (θ=90°) |

## UI

Launch the Gradio UI:
```bash
bash run_ui.sh
```

## Use Cases

- **Safety alignment**: Reduce harmful or toxic outputs
- **Style transfer**: Modify writing style or tone
- **Behavior modification**: Encourage or discourage response patterns
- **Interpretability research**: Study internal model representations

## Contributing

Contributions welcome! Please submit a Pull Request. For major changes, open an issue first.
