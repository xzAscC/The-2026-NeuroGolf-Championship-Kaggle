# The 2026 NeuroGolf Championship

> Kaggle Competition: Solve ARC-AGI tasks with the smallest possible ONNX networks.

## Competition Overview

The 2026 NeuroGolf Championship is a Kaggle competition that challenges participants to solve ARC-AGI reasoning tasks using **minimal ONNX neural networks**. Unlike typical ML competitions that maximize accuracy, this competition uniquely focuses on **network minimization** — solving each task correctly while using as few resources as possible.

- **Competition URL**: [https://www.kaggle.com/competitions/neurogolf-2026](https://www.kaggle.com/competitions/neurogolf-2026)
- **Task Type**: ARC-AGI abstract reasoning puzzles
- **Objective**: Build the smallest possible ONNX network that correctly transforms input grids to output grids
- **Evaluation**: Network cost-based scoring

## Problem Statement

Given ARC-AGI tasks consisting of input-output grid pairs (with values 0-9 representing colors), participants must create ONNX neural networks that:
1. Correctly solve each task (output must match ground truth exactly)
2. Minimize the total "cost" of the network

## Evaluation Metric

**Score per task**: `max(1, 25 - ln(cost))`

Where `cost = params + memory_bytes + MACs`

| Component | Description |
|-----------|-------------|
| `params` | Total number of learnable parameters |
| `memory_bytes` | Memory consumed by intermediate activations (bytes) |
| `MACs` | Multiply-accumulate operations during forward pass |

**Key implication**: Lower cost → higher score. The logarithmic scaling means diminishing returns for already-small networks, but every parameter matters.

## Input/Output Format

- **Tensor shape**: `[1, 10, 30, 30]` — one-hot float32 tensors
  - `1` = batch size
  - `10` = number of color channels (colors 0-9)
  - `30 x 30` = maximum grid dimensions
- **ONNX opset**: 11
- **IR version**: 10
- Grids are represented as one-hot encoded tensors where each color channel indicates the presence of that color at each spatial position.

## Data

The ARC-AGI training data with all 400 tasks (including test outputs) is available at:
- `https://huggingface.co/LuciferMrng/neurogolf-2026/blob/main/all_tasks.json`

## Approach

Our solver uses a **two-phase pipeline** that prioritizes low-cost analytical solutions before falling back to learned convolution-based solutions.

### Phase 1: Analytical Solvers (Near-Zero Cost)

These solvers detect and encode simple transformations directly into ONNX graphs with minimal parameters:

| Solver | Transformation | Typical Cost |
|--------|---------------|-------------|
| `identity` | Input equals output | Near zero |
| `constant` | Fixed output regardless of input | Near zero |
| `color_map` | 1x1 conv color transformation | Very low (10x10 kernel) |
| `transpose` | Matrix transpose | Near zero |
| `flip` | Horizontal/vertical flip | Near zero |
| `rotate` | 90/180/270 degree rotation | Near zero |
| `tile` | Repeat input pattern | Very low |
| `upscale` | Nearest-neighbor upscaling | Very low |
| `concat` | Concatenate transformed inputs | Very low |
| `spatial_gather` | Pixel remapping | Very low |
| `crop` | Centered crop | Near zero |

### Phase 2: Convolution Solvers (Learned Transformations)

When analytical methods fail, we use learned convolutions:

- **Fixed shape**: `Slice → Conv → ArgMax → OneHot → Pad`
- **Variable shape**: `Conv(30x30) → ArgMax → OneHot → Mul(mask)`
- **Diff shape**: `Slice → Conv → Slice(crop) → ArgMax → OneHot → Pad`

The conv solver learns optimal weights via least-squares fitting on one-hot patches, trying kernel sizes from 1 to 29 (smallest first).

### Key Optimizations

1. **Smallest kernel first** — Tries kernel size 1, then 3, 5, etc. (smaller = fewer params)
2. **Analytical before learned** — Analytical solvers have near-zero cost, always tried first
3. **One-hot ArgMax trick** — Converts softmax-like output to discrete one-hot via ArgMax, eliminating numerical precision issues
4. **No bias preferred** — Slightly fewer parameters when bias is unnecessary

## Project Structure

```
.
├── README.md                    # This file
├── solver.py                    # Main solver implementation
├── data/
│   └── all_tasks.json          # ARC-AGI tasks (400 tasks)
├── submission/                  # Generated ONNX files
│   ├── task_001.onnx
│   ├── task_002.onnx
│   └── ...
└── submission.zip              # Final submission archive
```

## Quick Start

### One-command setup (local venv, no global pollution)

```bash
bash setup.sh
```

This creates a `.venv/`, installs dependencies, and downloads the data. Then:

```bash
source .venv/bin/activate
```

### Run Solver

```bash
# Solve all 400 tasks
python solver.py --data_file data/all_tasks.json --output_dir submission --conv_budget 30

# Solve specific tasks
python solver.py --data_file data/all_tasks.json --output_dir submission --tasks 0,1,2,3,4
```

### Create Submission

The solver automatically creates `submission.zip` with all generated ONNX files. Upload this file to the Kaggle competition page.

## Current Results

| Configuration | Tasks Solved | Time |
|---------------|-------------|------|
| Analytical only | 36/400 | <1s |
| Analytical + Conv (2s/task) | 120/400 | ~9min |

Solver distribution: color_map(4), concat(8), conv_fixed(84), rotate(3), spatial_gather(18), transpose(2), upscale(1)

## Future Improvements

- [ ] Multi-step reasoning networks (chaining multiple conv layers)
- [ ] Attention-based spatial transformers for complex remapping
- [ ] Genetic algorithm for kernel size and architecture search
- [ ] Program synthesis for discovering task-specific DSL programs
- [ ] Recursive reasoning (TRM-inspired) for iterative refinement

## References

- [ARC-AGI Benchmark](https://github.com/fchollet/ARC-AGI) - Original abstract reasoning benchmark
- [rogermt/neurogolf-solver](https://huggingface.co/rogermt/neurogolf-solver) - Base solver reference
- [ashhhhhh26/neurogolf-2026](https://huggingface.co/ashhhhhh26/neurogolf-2026) - Enhanced solver with analytical methods
- [CompressARC](https://openreview.net/pdf?id=TbbMyr3E0x) - MDL-based inference-time learning approach
- [Tiny Recursive Models](https://github.com/olivkoch/TinyRecursiveModels) - Recursive reasoning with tiny networks (45% on ARC-AGI-1)
- [ARC Prize 2026](https://arcprize.org/competitions/2026) - Related ARC-AGI competitions

## License

MIT
