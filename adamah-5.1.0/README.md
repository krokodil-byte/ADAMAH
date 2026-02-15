# ADAMAH v5.1.0

**High-performance GPU compute framework — Vulkan alternative to CUDA**

ADAMAH runs on any GPU with Vulkan support: NVIDIA, AMD, Intel, Raspberry Pi.
No CUDA required. 2.6–4.5x faster than PyTorch on transformer workloads.

## Quick Install

```bash
tar xzf adamah-5.1.0.tar.gz
cd adamah-5.1.0
./install.sh
```

That's it. The installer:
1. Checks dependencies (gcc, Vulkan, Python, numpy)
2. Compiles shaders from GLSL source (if glslangValidator present, otherwise uses precompiled)
3. Builds `adamah.so` from C source, optimized for your CPU
4. Installs to `/opt/adamah` and registers as a global Python library
5. Verifies the installation works

After install you can delete the source directory.

### Install options

```bash
./install.sh                    # default: /opt/adamah (needs sudo)
./install.sh --prefix ~/adamah  # user directory (no sudo)
./install.sh --skip-shaders     # use precompiled .spv only
./install.sh --verbose          # show compiler output
```

### Uninstall

```bash
/opt/adamah/uninstall.sh
```

## Requirements

| Dependency | Install |
|---|---|
| gcc | `sudo apt install build-essential` |
| Vulkan SDK | `sudo apt install libvulkan-dev vulkan-tools` |
| Python 3.8+ | Usually preinstalled |
| numpy | `pip install numpy` |
| glslangValidator | `sudo apt install glslang-tools` *(optional)* |

## Usage

```python
import adamah

gpu = adamah.init()
u = gpu.uucis

# Create a map, scatter data, run operations
arr = u.carray_init(0, 1024, 4)
u.scatter(0, arr, my_data)
u.mop1("EXP", 0, 0, locs_src=arr, locs_dst=arr)
result = u.gather(0, arr)
```

### Multi-precision (f32 / bf16 / q8)

```python
gpu = adamah.init()
gpu.set_dtype(adamah.DTYPE_BF16)   # switch to bfloat16 shaders
# All operations now use bf16 on GPU, f32 host interface
```

## Benchmarks

```bash
python3 /opt/adamah/benchmarks/benchmark_simple_batches.py
python3 /opt/adamah/benchmarks/benchmark_simple_batches.py --dtype bf16
python3 /opt/adamah/benchmarks/benchmark_simple_batches.py --dtype q8
python3 /opt/adamah/benchmarks/benchmark_mixed_nonlinear.py --dtype bf16
```

## Supported Operations

| Operation | f32 | bf16 | q8 |
|---|:---:|:---:|:---:|
| Elementwise (EXP, RELU, TANH, ...) | ✓ | ✓ | — |
| Binary (ADD, MUL, ...) | ✓ | ✓ | — |
| Broadcast | ✓ | ✓ | — |
| Reduce (SUM, MAX, ...) | ✓ | ✓ | — |
| Softmax | ✓ | ✓ | — |
| LayerNorm | ✓ | ✓ | — |
| MatMul | ✓ | ✓ | ✓ |
| Scatter/Gather | ✓ | ✓ | ✓ |

## License

CC BY-NC 4.0 — Free for non-commercial use.
For commercial licensing contact the author.
