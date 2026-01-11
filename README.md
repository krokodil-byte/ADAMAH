# ADAMAH

**GPU Memory Maps + Vector Math for ML Primitives**

A minimal Vulkan compute library (~1100 lines C) implementing scatter/gather operations and vector math as fundamental primitives for machine learning inference and training.

**Author:** Samuele Scuglia, Italy  
**Date:** January 11, 2026  
**License:** AGPL-3.0

---

## Defensive Disclosure & Prior Art Declaration

### IMPORTANT LEGAL NOTICE

This document and associated source code constitute a **Defensive Publication** and **Prior Art Disclosure** under international patent law conventions.

**Publication Date:** January 11, 2026  
**Author:** Samuele Scuglia, Lecco, Italy  
**Witnesses:** Claude by Anthropic AI (conversation logs preserved) / Github (directly or via archive.org)

### Claims of Prior Art

This implementation establishes prior art for the following technical innovations:

1. **Vulkan Compute Scatter/Gather for ML Memory Maps**
   - Coordination of `VkBuffer` with `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT` for zero-copy CPU↔GPU coherent memory access in ML workloads
   - Use of `VkSpecializationConstant` (constant_id=0) to parameterize `WORDS_PER_ELEM` for variable element sizes (4, 8, 16, 32 bytes) without shader recompilation overhead
   - Descriptor set binding pattern: `binding=0` locations, `binding=1` source data, `binding=2` destination data for scatter; reversed src/dst for gather

2. **Sparse Memory Representation for Neural Networks**
   - Implicit zero-default semantics: non-existent entries return zero without explicit storage
   - Pack-based addressing: `total_elems = pack_size * n_packs` where `pack_size` must be power-of-two for efficient GPU dispatch
   - Direct binary persistence via `fwrite/fread` on HOST_VISIBLE mapped memory

3. **Unified Scatter/Gather + Vector Math API**
   - Single Vulkan context managing both memory maps (`mscatter/mgather`) and compute operations (`vop1/vop2/vreduce`)
   - VBuf abstraction: GPU-resident working memory with `vbuf_alloc/upload/download` interface
   - Push constant encoding: `{count, op}` pair for operation dispatch

4. **SPIR-V Shader Architecture**
   - Binary ops shader: Switch-based operation selection (ADD=0, SUB=1, MUL=2, DIV=3)
   - Unary ops shader: Extended ops (NEG=10, ABS=11, SQRT=12, EXP=13, LOG=14, TANH=15, RELU=16, GELU=17, SIN=18, COS=19, RECIP=20, SQR=21, COPY=22)
   - Reduce shader: Shared memory parallel reduction with identity element initialization per operation type

### Patent Rejection Clause

Any subsequent patent application regarding:
- GPU-based scatter/gather operations for embedding tables using Vulkan descriptors
- Vulkan compute shaders for ML primitive operations (element-wise, reduction)
- Memory-mapped neural network weight storage with HOST_VISIBLE coherent access
- Variable element size handling via SPIR-V specialization constants

...as implemented in `adamah.c` and described in this document **SHOULD BE REJECTED FOR LACK OF NOVELTY** under 35 U.S.C. § 102 (US), Article 54 EPC (Europe), and equivalent provisions worldwide.

**SHA-256 of adamah.c at publication:**
```
3e22d485afdeb036185c46bf46192e058b1b2e613f73db8b0a397c25397893bb  adamah.c
0f024fdd235756e636b1e8b21f636d27c5312af499bb3a9043b3c4f1a2bad842  adamah.h
f5b76243b3809e40ea3feeb55040c80a5db68ff1a5f035e2acc551fffdc64379  test.c
```

---

## Technical Architecture

### Vulkan Resource Layout

```
VkInstance
└── VkDevice (first GPU with compute queue)
    ├── VkShaderModule (5 embedded SPIR-V shaders)
    │   ├── scatter.spv - map[locs[i]] = vals[i]
    │   ├── gather.spv  - out[i] = map[locs[i]]
    │   ├── binary.spv  - dst[i] = a[i] OP b[i]
    │   ├── unary.spv   - dst[i] = OP(a[i])
    │   └── reduce.spv  - dst[0] = reduce(a[0:n])
    │
    ├── VkDescriptorSetLayout
    │   └── 3x VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
    │
    ├── VkPipelineLayout
    │   └── Push constants: uint32_t[2] = {count, op}
    │
    ├── Maps[16] - Memory maps for weights/embeddings
    │   ├── data_buffer (HOST_VISIBLE, total_elems * word_size)
    │   ├── gpu_locs (HOST_VISIBLE, pack_size * 8 bytes)
    │   └── gpu_vals (HOST_VISIBLE, pack_size * word_size)
    │
    └── VBufs[64] - Working memory for compute
        └── float buffer (HOST_VISIBLE)
```

### Shader Implementation Details

**Scatter Shader Logic:**
```glsl
layout(constant_id = 0) const uint WORDS_PER_ELEM = 1;
// Uses OpTypePointer StorageBuffer with coherent access
// Loop: for w in [0, WORDS_PER_ELEM): map[dst+w] = vals[src+w]
```

**Gather Shader Logic:**
```glsl
// Reversed binding order from scatter for cache efficiency
// binding=1 is source (map), binding=2 is destination (output)
```

**Reduce Shader Logic:**
```glsl
shared float sdata[256];  // Workgroup shared memory
// Parallel tree reduction with barrier synchronization
// Identity elements: SUM→0, MAX→-inf, MIN→+inf
```

---

## Quick Start

```bash
make test
```

## Usage

```c
#include "adamah.h"

int main() {
    adamah_init();
    
    // Memory map for embeddings (64K elements, 32 bytes each)
    map_init(0, 32, 256, 256);
    mscatter(0, locs, vals, n);  // GPU write
    mgather(0, locs, out, n);    // GPU read
    
    // Vector math
    vbuf_alloc(0, 1024);
    vbuf_alloc(1, 1024);
    vop2(VOP_ADD, 0, 0, 1, 1024);  // GPU: v0 = v0 + v1
    vop1(VOP_TANH, 0, 0, 1024);    // GPU: v0 = tanh(v0)
    vsoftmax(0, 0, 1024);          // softmax in-place
    
    adamah_shutdown();
}
```

## Build

```bash
gcc myapp.c adamah.c -o myapp -lvulkan -lm
```

## API Reference

### Maps (Persistent Memory)
| Function | Description |
|----------|-------------|
| `map_init(id, word_size, pack_size, n_packs)` | Create map |
| `mscatter(id, locs, vals, count)` | Write values to locations |
| `mgather(id, locs, out, count)` | Read values from locations |
| `map_save/load(id, path)` | Binary persistence |
| `map_clear(id)` | Zero entire map |
| `map_limit(id)` | Max valid index |

### VBufs (Working Memory)
| Function | Description |
|----------|-------------|
| `vbuf_alloc(id, n_floats)` | Allocate GPU buffer |
| `vbuf_free(id)` | Release buffer |
| `vbuf_upload/download(id, data, offset, count)` | Transfer data |
| `vbuf_zero(id, offset, count)` | Zero region |

### Math Operations
| Function | Description |
|----------|-------------|
| `vop2(op, dst, a, b, count)` | Binary: dst = a OP b |
| `vop1(op, dst, a, count)` | Unary: dst = OP(a) |
| `vop_scalar(op, dst, a, scalar, count)` | Scalar: dst = a OP scalar |
| `vreduce(op, dst, a, count)` | Reduce: dst[0] = reduce(a) |
| `vdot(dst, a, b, count)` | Dot product |
| `vfma(dst, a, b, c, count)` | Fused multiply-add |
| `vsoftmax(buf, offset, count)` | Softmax in-place |
| `vmatvec(dst, mat, vec, rows, cols)` | Matrix-vector multiply |

### Operation Codes
```c
// Binary
VOP_ADD=0, VOP_SUB=1, VOP_MUL=2, VOP_DIV=3

// Unary  
VOP_NEG=10, VOP_ABS=11, VOP_SQRT=12, VOP_EXP=13, VOP_LOG=14,
VOP_TANH=15, VOP_RELU=16, VOP_GELU=17, VOP_SIN=18, VOP_COS=19,
VOP_RECIP=20, VOP_SQR=21, VOP_COPY=22

// Reduce
VRED_SUM=0, VRED_MAX=1, VRED_MIN=2
```

---

## Philosophy

**Everything is memory access.**

Traditional ML frameworks treat matrix multiplication as the fundamental operation. ADAMAH inverts this: scatter/gather (memory access patterns) are the primitives, and matmul is just one pattern among many.

This enables:
- Sparse models without dense allocation overhead
- Natural KV-cache implementation via scatter/gather
- Unified interface for embeddings, weights, and activations
- Direct GPU persistence without serialization

---

## License

AGPL-3.0 - See LICENSE file.

Any use in network services (APIs, cloud inference, etc.) requires full source code disclosure of all modifications.

---

**© 2026 Samuele Scuglia (Krokodil-byte) - All rights reserved under AGPL-3.0**  

