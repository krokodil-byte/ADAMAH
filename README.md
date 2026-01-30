# ADAMAH 5.0

GPU compute library using Vulkan. No CUDA required.

## Quick Start

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt install libvulkan-dev glslang-tools

# Build
./compile.sh

# Install
pip install -e .

# Test
python tests/test_all_ops.py
```

## Usage

```python
import adamah
import numpy as np

gpu = adamah.Adamah()

a = np.random.randn(1024, 1024).astype(np.float32)
b = np.random.randn(1024, 1024).astype(np.float32)

# Matrix multiply
c = gpu.matmul(a, b)

# Element-wise ops
x = gpu.relu(a)
y = gpu.sigmoid(b)
z = gpu.add(x, y)

gpu.shutdown()
```

## Operations

**Unary**: neg, abs, sqrt, exp, log, tanh, relu, gelu, sigmoid, swish, mish, selu, elu, leaky_relu, softplus, sin, cos, tan, hardsigmoid, hardswish, reciprocal, square, cube, sign, ceil, floor, round

**Binary**: add, sub, mul, div, pow, min, max, mod, eq, ne, lt, le, gt, ge, and, or, xor

**Reduction**: sum, mean, max, min, prod

**Matrix**: matmul, softmax, layernorm, broadcast

## Requirements

- Linux with Vulkan-capable GPU
- Python 3.8+
- numpy

## License

CC-BY-NC-4.0
