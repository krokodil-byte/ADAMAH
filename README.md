# ADAMAH v4.3.0

**Vulkan to FFI GPU Compute Library*

Zero-CUDA alternative using Vulkan. Supports transformers, embeddings, and neural network operations.

## Quick Install

```bash
# One-liner install from GitHub
pip install git+https://github.com/krokodil-byte/ADAMAH.git

# Requirements: Vulkan drivers + GCC
sudo apt install libvulkan-dev build-essential  # Ubuntu/Debian
```

## Features

- **Matrix Multiplication**: `map_matmul()` - batched matmul on GPU
- **Reduce Operations**: `map_reduce_sum/max/min()` - for softmax, layernorm
- **Broadcast**: `map_scale()` - scalar-vector operations
- **Element-wise**: `map_sin/cos/exp/tanh/relu/gelu/add/mul...`
- **Scatter/Gather**: Sparse access patterns for embeddings

## Quick Start

```python
import adamah
import numpy as np

gpu = adamah.Adamah()

# Create GPU memory map
gpu.map_init(0, word_size=4, pack_size=768, n_packs=50000)  # 50k embeddings

# Load data (CPU → GPU)
embeddings = np.random.randn(50000, 768).astype(np.float32)
gpu.scatter(0, np.arange(50000, dtype=np.uint32), embeddings.flatten())

# GPU operations
locs = np.array([0, 1, 2], dtype=np.uint32)
gpu.map_op1(0, adamah.OP_GELU, locs, locs)  # GELU activation

# Read back (GPU → CPU)
result = gpu.gather(0, locs)

gpu.shutdown()
```

## Matrix Multiplication

```python
# C = A @ B  (batched)
gpu.map_matmul(map_id, locs_A, locs_B, locs_C, M, K, N)
```

## Softmax (using reduce + broadcast)

```python
# softmax = exp(x - max) / sum(exp(x - max))
gpu.map_reduce_max(0, x_locs, max_locs)      # max per row
gpu.map_broadcast(0, BROADCAST_SUB, x_locs, max_locs, tmp_locs)  # x - max
gpu.map_exp(0, tmp_locs, tmp_locs)           # exp
gpu.map_reduce_sum(0, tmp_locs, sum_locs)    # sum
gpu.map_div_scalar(0, tmp_locs, sum_locs, out_locs)  # normalize
```

## API Reference

| Function | Description |
|----------|-------------|
| `map_init(id, word_size, pack_size, n_packs)` | Create GPU memory map |
| `scatter(id, locs, data)` | CPU → GPU transfer |
| `gather(id, locs)` | GPU → CPU transfer |
| `map_matmul(id, A, B, C, M, K, N)` | Matrix multiplication |
| `map_reduce_sum/max/min(id, src, dst)` | Reduce along pack |
| `map_scale(id, src, scalar, dst)` | Broadcast multiply |
| `map_op1(id, op, src, dst)` | Unary ops (sin, exp, gelu...) |
| `map_op2(id, op, a, b, dst)` | Binary ops (add, mul...) |

## Operations

**Unary**: NEG, ABS, SQRT, EXP, LOG, TANH, RELU, GELU, SIN, COS, RECIP, SQR

**Binary**: ADD, SUB, MUL, DIV, POW, MIN, MAX

**Reduce**: SUM, MAX, MIN

**Broadcast**: MUL, DIV, ADD, SUB

## Performance

- 65+ GB/s memory bandwidth on RTX 3070
- Zero-copy GPU compute (data stays in VRAM)
- Cross-platform via Vulkan (NVIDIA, AMD, Intel)

## Requirements

- Linux (Ubuntu 20.04+)
- Vulkan drivers (`vulkaninfo` to check)
- GCC (`build-essential`)
- Python 3.8+

## License

**CC BY-NC 4.0** - Free for non-commercial use with attribution.

© 2026 Samuele Scuglia
