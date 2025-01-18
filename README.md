# NGPT for Mac

A Mac-compatible implementation of NVIDIA's Normalized GPT (NGPT), based on [lucidrains' PyTorch implementation](https://github.com/lucidrains/nGPT-pytorch).

## Installation

1. Ensure you have Python 3.9 or later installed
2. Clone this repository:
```bash
git clone https://github.com/unktomi/ngpt-mac.git
cd ngpt-mac
```

3. Install the package:
```bash
pip install -e .
```

## Usage

```python
import torch
from ngpt_mac.model import NGPT
from ngpt_mac.train import create_model, train

# Create a model
model = create_model(
    vocab_size=50257,  # GPT-2 vocabulary size
    dim=512,           # Model dimension
    depth=6,           # Number of transformer layers
    heads=8           # Number of attention heads
)

# Train the model
train(
    model,
    train_dataset,    # Your dataset here
    batch_size=16,
    learning_rate=3e-4,
    max_iters=100000
)
```

## Features

- Full implementation of NGPT architecture with hypersphere normalization
- Automatic device selection (CUDA/MPS/CPU)
- Gradient accumulation for larger effective batch sizes
- Learning rate decay

## Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- Other dependencies are listed in pyproject.toml

## Mac-specific Notes

This implementation automatically detects and uses Metal Performance Shaders (MPS) on Apple Silicon Macs when available, falling back to CPU if MPS is not available.

## License

MIT