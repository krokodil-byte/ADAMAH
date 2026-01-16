# ADAMAH (1.1.1)- GPU Memory & Math Library

**As simple as possible, but complete.**

```c
inject("x", data, n);
vop1(VOP_SIN, "y", "x", n);
extract("y", result, n);
```

## Features

- **Named Buffers** - Auto-managed, no manual alloc/free
- **Memory Maps** - Scatter/gather for sparse data
- **Full Math** - Trig, calculus, reduce ops
- **Single Header** - Just `adamah.h` + `adamah.c`

## Quick Start

```c
#include "adamah.h"

int main() {
    adamah_init();
    
    float x[] = {0, 0.5, 1.0, 1.5};
    inject("x", x, 4);
    
    vop1(VOP_SIN, "y", "x", 4);
    
    float y[4];
    extract("y", y, 4);
    // y = [0, 0.479, 0.841, 0.997]
    
    adamah_shutdown();
}
```

## API Reference

### Core
```c
int adamah_init(void);
void adamah_shutdown(void);
```

### Buffers
```c
int inject(const char* name, const float* data, uint32_t count);
int extract(const char* name, float* data, uint32_t count);
uint32_t bufsize(const char* name);
```

### Binary Ops - `dst = a OP b`
```c
int vop2(uint32_t op, const char* dst, const char* a, const char* b, uint32_t count);

VOP_ADD   // +
VOP_SUB   // -
VOP_MUL   // *
VOP_DIV   // /
VOP_POW   // a^b
VOP_ATAN2 // atan2(a,b)
VOP_MOD   // fmod
VOP_MIN   // min(a,b)
VOP_MAX   // max(a,b)
```

### Unary Ops - `dst = OP(a)`
```c
int vop1(uint32_t op, const char* dst, const char* a, uint32_t count);

// Basic
VOP_NEG, VOP_ABS, VOP_SQRT, VOP_SQR, VOP_RECIP, VOP_COPY

// Exponential
VOP_EXP, VOP_EXP2, VOP_LOG, VOP_LOG2, VOP_LOG10

// Trigonometric
VOP_SIN, VOP_COS, VOP_TAN, VOP_ASIN, VOP_ACOS, VOP_ATAN

// Hyperbolic
VOP_SINH, VOP_COSH, VOP_TANH

// Rounding
VOP_FLOOR, VOP_CEIL, VOP_ROUND, VOP_TRUNC, VOP_SIGN

// ML Activations
VOP_RELU, VOP_GELU
```

### Scalar Ops - `dst = a OP scalar`
```c
int vops(uint32_t op, const char* dst, const char* a, float scalar, uint32_t count);
```

### Reduce Ops - `dst[0] = reduce(a)`
```c
int vreduce(uint32_t op, const char* dst, const char* a, uint32_t count);

VRED_SUM   // Σ
VRED_PROD  // Π
VRED_MAX   // max
VRED_MIN   // min
VRED_MEAN  // μ
```

### Linear Algebra
```c
int vdot(const char* dst, const char* a, const char* b, uint32_t count);
int vmatvec(const char* dst, const char* mat, const char* vec, uint32_t rows, uint32_t cols);
int vsoftmax(const char* buf, uint32_t count);
```

### Calculus
```c
int vcumsum(const char* dst, const char* a, uint32_t count);      // Cumulative sum
int vcumprod(const char* dst, const char* a, uint32_t count);     // Cumulative product
int vdiff(const char* dst, const char* a, uint32_t count);        // Finite differences
int vintegrate(const char* dst, const char* a, float dx, uint32_t count);   // ∫a dx
int vderivative(const char* dst, const char* a, float dx, uint32_t count);  // da/dx
```

### Generators
```c
int vlinspace(const char* dst, float start, float stop, uint32_t count);
int varange(const char* dst, float start, float step, uint32_t count);
```

### Memory Maps (Sparse)
```c
int map_init(uint32_t id, uint32_t word_size, uint32_t pack_size, uint32_t n_packs);
int map_destroy(uint32_t id);
int map_clear(uint32_t id);
int mscatter(uint32_t id, const char* locs, const char* vals, uint32_t count);
int mgather(uint32_t id, const char* locs, const char* dst, uint32_t count);
int map_save(uint32_t id, const char* path);
int map_load(uint32_t id, const char* path);
```

## Examples

### Signal Processing
```c
vlinspace("t", 0, 6.28, 1000);     // t = [0, 2π]
vop1(VOP_SIN, "wave", "t", 1000);  // sin wave
vderivative("dw", "wave", 0.00628, 1000);  // derivative
```

### Statistics
```c
inject("data", samples, n);
vreduce(VRED_MEAN, "mean", "data", n);
vreduce(VRED_MAX, "max", "data", n);
```

### Neural Network Layer
```c
inject("x", input, 784);
inject("W", weights, 784*128);
vmatvec("h", "W", "x", 128, 784);
vop1(VOP_RELU, "h", "h", 128);
vsoftmax("h", 128);
```

### Sparse Embeddings
```c
map_init(0, 128, 1024, 1024);  // 1M x 128 embedding table
inject("ids", token_ids, batch);
mgather(0, "ids", "emb", batch);
```

## Build

```bash
gcc -O3 your_code.c adamah.c -o app -lvulkan -lm
```

## Requirements

- Vulkan 1.0+
- C99 compiler

## License

CC BY-NC 4.0 - Samuele Scuglia, January 2026

---

**ADAMAH** = Ground/Earth - The foundation for computation.
