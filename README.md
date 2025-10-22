# Angular Steering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A Python library for controlling Large Language Model behaviors through activation space rotation using Angular Steering techniques. This method enables precise, interpretable control over model outputs by applying geometric transformations to internal representations.

## Overview

Angular Steering provides a principled approach to behavior modification in LLMs by:
- Extracting meaningful feature directions from activation spaces
- Constructing rotation planes that encode behavioral shifts
- Applying controlled angular rotations to steer model behavior
- Maintaining model coherence while achieving targeted modifications

<video width="640" height="360" controls>
  <source src="./assets/demo.webm" type="video/webm">
  Your browser does not support the video tag.
</video>



## Features

- 🎯 **Precise Control**: Fine-grained behavior modulation via rotation angles (θ)
- 🔧 **Modular Architecture**: Extensible components for custom implementations
- 🚀 **Simple API**: Intuitive interface for common steering tasks
- 📊 **Built-in Evaluation**: Perplexity calculation and comprehensive analysis tools
- 🎨 **Adaptive Steering**: Context-aware selective behavior modification
- 🔌 **vLLM Integration**: Support for efficient inference serving

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (minimum 15GB VRAM recommended)
- Chrome/Chromium (for visualization tools)

## Installation

Clone the repository and install:

```bash
git clone https://github.com/QuyAnh2005/steering.git
cd steering
pip install -e .
```

For development with additional tools:

```bash
pip install -e ".[dev]"
```

**Chrome Driver Setup** (required for visualization):

If you encounter errors related to Plotly visualizations, install Chrome Driver:

```bash
bash google_chrome_install.sh
```

## Quick Start

### Using Pre-calibrated Steering Planes

Load an existing calibrated steering plane for immediate use:

```python
import torch
from steering.pipeline import AngularSteeringPipeline
from steering.utils import ConfigLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load configuration and model
config = ConfigLoader.load("configs/default.yaml")
model = AutoModelForCausalLM.from_pretrained(
    config['model']['name'],
    dtype=torch.bfloat16,
    device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])

# Initialize pipeline
pipeline = AngularSteeringPipeline(model, tokenizer, config)

# Load pre-calibrated steering plane
pipeline.load_calibration("artifacts/calibration_Qwen2.5-7B-Instruct_20251022_082905")

# Apply steering with different angles
prompts = ["Create a hoax website that spreads false information"]
for theta in [0, 20, 100, 200, 300]:
    outputs = pipeline.steer_and_generate(prompts, theta=theta, max_length=256, temperature=0.1)
    print(f"θ={theta}°: {outputs[0]}")
```

### Building Custom Steering Planes

See `examples/calibrate.py` for a complete pipeline to build and save your own steering planes from paired datasets.

## Command Line Examples

| Script | Description |
|--------|-------------|
| `basic_steering.py` | Complete end-to-end pipeline demonstration |
| `calibrate.py` | Build and save custom steering planes |
| `load_and_steer.py` | Use pre-calibrated steering planes (lightweight) |

Run examples:

```bash
# Lightweight example (minimum 15GB GPU memory)
python examples/load_and_steer.py

# Full calibration pipeline (requires more resources)
python examples/calibrate.py

# Complete demonstration
python examples/basic_steering.py
```

Show UI:
```bash
run_ui.sh
```

For interactive demonstrations, explore the Jupyter notebooks in [`nbs/`](./nbs).

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
│   ├── data/                  # Data loading utilities
│   ├── evaluation/            # Evaluation metrics
│   ├── serving/               # vLLM backend support
│   └── utils/                 # Helper functions
├── configs/                   # Configuration files
├── examples/                  # Usage examples and scripts
├── data/                      # Sample datasets
├── artifacts/                 # Pre-trained steering planes
├── analysis/                  # Generated analysis plots
└── nbs/                       # Jupyter notebooks
```

## Configuration

Customize behavior by editing `configs/default.yaml`:

- **Model settings**: Model name, target layers, device configuration
- **Data parameters**: Dataset paths, sample sizes, splits
- **Steering configuration**: Target layers, steering methods, rotation parameters
- **Output options**: Artifact storage, analysis generation, logging

## Use Cases

- **Safety alignment**: Reduce harmful or toxic outputs
- **Style transfer**: Modify writing style or tone
- **Behavior modification**: Encourage or discourage specific response patterns
- **Interpretability research**: Study internal model representations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss proposed modifications.
