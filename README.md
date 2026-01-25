# ADAMAH v5.0.0

Map-centric GPU compute library (Vulkan) for AI/ML workloads.
Zero-CUDA alternative with cached locations, batching, and map-oriented ops.

## Quick Install

```bash
# One-liner install from GitHub
pip install git+https://github.com/krokodil-byte/ADAMAH.git

# Requirements: Vulkan drivers + GCC
sudo apt install libvulkan-dev build-essential  # Ubuntu/Debian
```

## Quick Start (Python, recommended)

```python
import numpy as np
import adamah

gpu = adamah.init(cache_mb=512)  # optional cache
u = gpu.uucis

# 1D map (array) with 1024 float32 elements
u.array_init(map_id=0, n_cells=1024, wordlength=4)
locs = np.arange(1024, dtype=np.uint32)
locs_c = u.cache_locs(0, locs)

x = np.random.randn(1024).astype(np.float32)

# CPU -> GPU
u.scatter(0, locs_c, x)

# Unary op: exp(x)
u.mop1("EXP", map_id=0, target=0, locs_src=locs_c, locs_dst=locs_c)

# GPU -> CPU
out = u.gather(0, locs_c)

gpu.shutdown()
```

## Features (Python API)

- UUCIS wrapper with cached locs and device-only ops
- Unary, binary, reduce, broadcast, softmax, layernorm, matmul
- Scatter/gather for sparse map access
- Auto-batching (with optional manual batching)

## UUCIS Ops (string-based)

`mop1` (unary / reduce / softmax / layernorm)
- Unary: NEG, ABS, SQRT, EXP, LOG, TANH, RELU, GELU, SIN, COS, RECIP, SQR
- Reduce: SUM, MAX, MIN
- SOFTMAX, LAYERNORM

`mop2` (binary / broadcast / matmul)
- Binary: ADD, SUB, MUL, DIV, POW, MIN, MAX
- Broadcast: ADD, SUB, MUL, DIV
- MATMUL

Examples:

```python
u.mop1("REDUCE:SUM", 0, 0, locs_src=src_c, locs_dst=dst_c)

u.mop2("BROADCAST:ADD", 0, 0, 0, locs_a=src_c, locs_b=scalar_c, locs_dst=dst_c)

u.mop2("MATMUL", 3, 3, 3, extra={
    "locs_a": a_c,
    "locs_b": b_c,
    "locs_c": c_c,
    "M": M, "K": K, "N": N,
})
```

## Batching and Sync

- Auto-batching is enabled by default:

```python
u.set_auto_batching(True, limit=4096)
```

- Disable auto-batching (submit each op immediately):

```python
u.set_auto_batching(False)
```

- Manual batching:

```python
with gpu.batch():
    u.mop1(...)
    u.mop2(...)
```

- Sync:

```python
gpu.synchronize_all()
```

## Low-level Adamah API (optional)

The `Adamah` object exposes direct map ops (no string parsing):

- `map_init`, `scatter`, `gather`
- `map_op1`, `map_op2`, `map_reduce`, `map_broadcast`
- `map_softmax`, `map_layernorm`, `map_matmul`
- `batch_begin`, `batch_end`, `synchronize_all`

## Array API (status)

The `gpu.array`, `gpu.add`, and `gpu.mul` methods exist but are not implemented yet.
Use maps + UUCIS instead.

## Requirements

- Linux (Ubuntu 20.04+)
- Vulkan drivers (`vulkaninfo` to check)
- GCC (`build-essential`)
- Python 3.8+

## License

CC BY-NC 4.0 - Free for non-commercial use with attribution.

(c) 2026 Samuele Scuglia
