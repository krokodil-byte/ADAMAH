# ADAMAH v4.0.0

**Map-Centric GPU Compute Library**

Pure GPU operations on Memory Maps with scatter/gather for CPU I/O.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MAP (GPU VRAM)                           │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┐                    │
│  │ [0] │ [1] │ [2] │ ... │[n-1]│     │  ← packs           │
│  └──┬──┴──┬──┴──┬──┴─────┴─────┴─────┘                    │
│     │     │     │                                          │
│  scatter  │  map_op (GPU compute)                         │
│     ↑     │     │                                          │
│   CPU    GPU    ↓                                          │
│   data   only  gather → CPU data                          │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install adamah
```

**Requirements:** Vulkan SDK & drivers

## Quick Start

```python
import adamah

gpu = adamah.Adamah()

# Create map: 1M packs of 128 floats
gpu.map_init(0, word_size=4, pack_size=128, n_packs=1_000_000)

# CPU → GPU
locs = np.array([0, 1, 2], dtype=np.uint32)
data = np.random.randn(3 * 128).astype(np.float32)
gpu.scatter(0, locs, data)

# Pure GPU operations
gpu.map_sin(0, src_locs, dst_locs)           # sin
gpu.map_add(0, a_locs, b_locs, out_locs)     # add

# GPU → CPU
result = gpu.gather(0, out_locs)

# Persistence
gpu.map_save(0, "model.bin")
gpu.map_load(0, "model.bin")

gpu.shutdown()
```

## API Reference

### Core
| Function | Description |
|----------|-------------|
| `Adamah()` | Initialize GPU context |
| `shutdown()` | Cleanup resources |

### Memory Maps
| Function | Description |
|----------|-------------|
| `map_init(id, word_size, pack_size, n_packs)` | Create map |
| `map_destroy(id)` | Destroy map |
| `map_size(id)` | Get number of packs |
| `map_save(id, path)` | Save to file |
| `map_load(id, path)` | Load from file |

### Data Transfer (CPU ↔ GPU)
| Function | Description |
|----------|-------------|
| `scatter(id, locs, data)` | Write data to map[locs] |
| `gather(id, locs)` | Read data from map[locs] |

### GPU Operations
| Function | Description |
|----------|-------------|
| `map_op1(id, op, src, dst)` | Unary: map[dst] = op(map[src]) |
| `map_op2(id, op, a, b, dst)` | Binary: map[dst] = map[a] op map[b] |

**Shortcuts:**
- `map_sin`, `map_cos`, `map_exp`, `map_tanh`, `map_relu`
- `map_add`, `map_mul`

### Operations

**Unary (OP_*):** NEG, ABS, SQRT, EXP, LOG, TANH, RELU, GELU, SIN, COS, RECIP, SQR

**Binary (OP_*):** ADD, SUB, MUL, DIV, POW, MIN, MAX

## Example: Neural Network Embedding

```python
gpu = adamah.Adamah()

# Embedding table: 50k vectors of 768 dims
gpu.map_init(0, word_size=4, pack_size=768, n_packs=50_000)

# Load pretrained embeddings
gpu.scatter(0, all_indices, embeddings)

# Lookup batch of 32 tokens
batch_locs = np.array([tok1, tok2, ..., tok32], dtype=np.uint32)
vectors = gpu.gather(0, batch_locs)  # Shape: (32 * 768,)

# GPU-side operations on embeddings
gpu.map_sin(0, batch_locs, output_locs)  # Apply activation
```

## Performance

- **Zero-copy GPU ops**: Operations stay in VRAM
- **Sparse access**: scatter/gather by pack index
- **Async execution**: Vulkan compute shaders
- **65+ GB/s** memory bandwidth on RTX 3070

## License

**CC BY-NC 4.0** - Creative Commons Attribution-NonCommercial 4.0

Copyright (c) 2026 Samuele Scuglia

Free for non-commercial use with attribution.
