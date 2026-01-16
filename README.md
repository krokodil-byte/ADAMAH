# ADAMAH

**GPU Memory & Math Library - As simple as possible, but complete.**

```python
import adamah

adamah.init()
adamah.put("x", [0, 1, 2, 3])
adamah.sin("y", "x", 4)
print(adamah.get("y"))  # [0, 0.841, 0.909, 0.141]
adamah.shutdown()
```

## Install

```bash
pip install adamah
```

Requires: `libvulkan-dev` (auto-compiles on first use)

## Usage

```python
import adamah

# Context manager (recommended)
with adamah.Adamah() as gpu:
    gpu.put("a", [1, 2, 3, 4])
    gpu.put("b", [4, 3, 2, 1])
    gpu.add("c", "a", "b", 4)
    print(gpu.get("c"))  # [5, 5, 5, 5]

# Or functional API
adamah.init()
adamah.put("x", data)
adamah.sin("y", "x", len(data))
result = adamah.get("y")
adamah.shutdown()
```

## Operations

- **Math**: sin, cos, tan, exp, log, sqrt, tanh, relu, gelu
- **Ops**: add, sub, mul, div, pow
- **Reduce**: sum, max, min, mean, prod
- **Calculus**: cumsum, diff, integrate, derivative
- **Linear Algebra**: dot, matvec, softmax
- **Generators**: linspace, arange
- **Sparse**: scatter/gather memory maps

## License

CC BY-NC 4.0, Scuglia Samuele january 2026
see adamah(3.0.0)/src/adamah/LICENSE
