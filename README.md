# ADAMAH 5.1.0

**High-Performance Cross-Platform GPU Computing Framework**

ADAMAH is a GPU compute framework that runs on **any GPU** via Vulkan - NVIDIA, AMD, Intel, Apple Silicon, and more. No CUDA required.

## 🚀 Performance

ADAMAH outperforms PyTorch (CUDA) by **2-4x** on transformer workloads:

| Workload | vs PyTorch (CUDA) | vs CuPy |
|----------|-------------------|---------|
| Attention-FFN Block | **4x faster** | 20x |
| Residual Chain | **3.5x faster** | 17x |
| Multi-Head Attention | **2.5x faster** | 17x |

*Benchmarked on RTX 3070 with identical logical operations*

## ✨ Key Features

- **Automatic Operation Fusion** - Batches independent operations for minimal GPU dispatch overhead
- **Dependency-Aware Scheduling** - Automatically orders operations by data dependencies
- **Dynamic GPU Tuning** - Adapts buffer sizes and batch limits to your GPU's capabilities
- **Cross-Platform** - Works on any Vulkan-capable GPU (NVIDIA, AMD, Intel, Apple, ARM)
- **Zero CUDA Dependency** - Pure Vulkan compute, no proprietary toolchains

## 📦 Installation

### Requirements
- Python 3.8+
- Vulkan-capable GPU with drivers installed
- NumPy

### Quick Start
```bash
# Clone/download the package
cd adamah-clean

# Install
pip install -e .

# Verify
python -c "import adamah; gpu = adamah.Adamah(); print('ADAMAH ready!')"
```

## 🔧 Supported Operations

### Unary Operations
| Op | Description |
|----|-------------|
| `NEG` | Negate |
| `ABS` | Absolute value |
| `SQRT` | Square root |
| `EXP` | Exponential |
| `LOG` | Natural logarithm |
| `TANH` | Hyperbolic tangent |
| `RELU` | Rectified linear unit |
| `GELU` | Gaussian error linear unit |
| `SIN` | Sine |
| `COS` | Cosine |
| `RECIP` | Reciprocal (1/x) |
| `SQR` | Square (x²) |

### Binary Operations
| Op | Description |
|----|-------------|
| `ADD` | Element-wise addition |
| `SUB` | Element-wise subtraction |
| `MUL` | Element-wise multiplication |
| `DIV` | Element-wise division |
| `POW` | Element-wise power |
| `MIN` | Element-wise minimum |
| `MAX` | Element-wise maximum |

### Reduction Operations
| Op | Description |
|----|-------------|
| `SUM` | Sum reduction |
| `MAX` | Max reduction |
| `MIN` | Min reduction |

### Neural Network Operations
| Op | Description |
|----|-------------|
| `SOFTMAX` | Row-wise softmax |
| `LAYERNORM` | Layer normalization |
| `MATMUL` | Matrix multiplication |
| `BROADCAST:ADD/MUL/...` | Broadcast operations |

## 💡 Usage Examples

### Basic Usage
```python
import adamah
import numpy as np

# Initialize
gpu = adamah.Adamah()

# Create a map (GPU memory region)
map_id = 0
gpu.map_create(map_id, word_size=4, pack_size=1, n_packs=1024)

# Upload data
data = np.random.randn(1024).astype(np.float32)
locs = np.arange(1024, dtype=np.uint32)
gpu.scatter(map_id, locs, data)

# Compute exp(x)
gpu.map_op1(map_id, op=3, locs_in=locs, locs_out=locs)  # 3 = EXP

# Download result
result = gpu.gather(map_id, locs)
print(result[:5])
```

### Using UUCIS High-Level API
```python
import adamah
import numpy as np

gpu = adamah.Adamah()
u = gpu.uucis

# Create and initialize map
map_id = 0
u.make_map(map_id, n=1024)

# Cache locations for fast repeated operations
locs = u.cache_locs(map_id, np.arange(1024, dtype=np.uint32))

# Upload data
x = np.random.randn(1024).astype(np.float32)
u.scatter(map_id, locs, u.to_cached(x))

# Chain operations - automatically fused!
u.mop1("EXP", map_id, map_id, locs_src=locs, locs_dst=locs)
u.mop1("TANH", map_id, map_id, locs_src=locs, locs_dst=locs)
u.mop1("RELU", map_id, map_id, locs_src=locs, locs_dst=locs)

# Sync and download
gpu.synchronize_all()
result = u.gather(map_id, locs)
```

### Matrix Multiplication
```python
# Setup matrices A (M×K) and B (K×N) -> C (M×N)
M, K, N = 128, 256, 128

# Allocate space
a_base = 0
b_base = M * K
c_base = b_base + K * N

gpu.map_create(0, 4, 1, c_base + M * N)

# Upload A and B
gpu.scatter(0, np.arange(M*K, dtype=np.uint32) + a_base, A.flatten())
gpu.scatter(0, np.arange(K*N, dtype=np.uint32) + b_base, B.flatten())

# Matmul
locs_a = u.cache_locs(0, np.array([a_base], dtype=np.uint32))
locs_b = u.cache_locs(0, np.array([b_base], dtype=np.uint32))
locs_c = u.cache_locs(0, np.array([c_base], dtype=np.uint32))

u.mop2("MATMUL", 0, 0, 0, extra={
    "locs_a": locs_a, "locs_b": locs_b, "locs_c": locs_c,
    "M": M, "K": K, "N": N
})

gpu.synchronize_all()
```

## ⚡ Automatic Fusion System

ADAMAH automatically fuses operations to minimize GPU dispatch overhead:

```python
# These operations are automatically batched:
u.mop1("EXP", ...)   # Level 0 - queued
u.mop1("TANH", ...)  # Level 1 - queued (depends on EXP)
u.mop1("RELU", ...)  # Level 2 - queued (depends on TANH)
u.mop2("ADD", ...)   # Level 0 - queued (independent)

# All operations execute with a single GPU dispatch when you:
gpu.synchronize_all()  # or gpu.gather(...)
```

**How it works:**
1. Operations are queued with their dependency levels
2. Independent operations (same level) execute in parallel
3. Dependent operations wait for their inputs
4. Single GPU submission for the entire batch

**Automatic triggers for execution:**
- `gpu.synchronize_all()` - Lightweight sync
- `gpu.gather(...)` - When you need results
- `gpu.scatter(...)` - Before uploading new data
- Queue full (8192 ops) - Auto-flush

## 🔬 Benchmarks

Run the included benchmarks:

```bash
# Mixed non-linear operations benchmark
python benchmarks/benchmark_mixed_nonlinear.py

# Comprehensive operations benchmark  
python benchmarks/benchmark_simple_batches.py
```

## 📁 Package Structure

```
adamah-clean/
├── adamah/
│   ├── __init__.py      # Main Python API
│   ├── adamah.c         # C/Vulkan core
│   ├── adamah.so        # Compiled library
│   ├── uucis.py         # High-level API
│   └── shaders/         # Precompiled SPIR-V shaders
│       ├── map_op1.spv
│       ├── map_op2.spv
│       ├── map_matmul.spv
│       ├── map_softmax.spv
│       ├── map_layernorm.spv
│       └── ...
├── benchmarks/
│   ├── benchmark_mixed_nonlinear.py
│   └── benchmark_simple_batches.py
├── tests/
│   └── test_all_ops.py
├── pyproject.toml
├── LICENSE
└── README.md
```


## 📄 License

CC-BY-NC 4.0 License - see LICENSE file.

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional operations (conv2d, attention kernels)
- Performance optimizations
- Support for more platforms
- Documentation improvements

---

**ADAMAH** - *The Ground for computation.*
