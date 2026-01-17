# ADAMAH GPU/CPU Hybrid Architecture

## Design Goals
- Maintain backward compatibility
- Global mode setting + per-operation override
- GPU compute shaders for parallel ops
- CPU parallelism with OpenMP
- Simple, clean API

## Architecture

### 1. Execution Modes

```c
typedef enum {
    ADAMAH_MODE_CPU = 0,  // CPU execution (OpenMP parallel)
    ADAMAH_MODE_GPU = 1,  // GPU execution (Vulkan compute)
    ADAMAH_MODE_AUTO = 2  // Auto-select based on size
} AdamahMode;
```

### 2. API Design

#### C API
```c
// Global mode
void adamah_set_mode(int mode);  // 0=CPU, 1=GPU, 2=AUTO
int adamah_get_mode(void);

// Extended functions with mode override
void vop1_ex(int op, const char* dst, const char* a, uint32_t count, int mode);
void vop2_ex(int op, const char* dst, const char* a, const char* b, uint32_t count, int mode);
void vops_ex(int op, const char* dst, const char* a, float scalar, uint32_t count, int mode);

// Backward compatible (use global mode)
void vop1(int op, const char* dst, const char* a, uint32_t count);
void vop2(int op, const char* dst, const char* a, const char* b, uint32_t count);
void vops(int op, const char* dst, const char* a, float scalar, uint32_t count);
```

#### Python API
```python
# Global mode
adamah.set_mode('gpu')   # or 'cpu', 'auto'

# Operations use global mode by default
adamah.add("c", "a", "b", 4)  # uses GPU

# Override per operation
adamah.sin("y", "x", 4, mode='cpu')  # force CPU
```

### 3. GPU Implementation

#### Compute Shaders Structure
```
src/adamah/shaders/
├── unary.comp       # Unary operations (sin, cos, exp, etc)
├── binary.comp      # Binary operations (add, mul, etc)
├── scalar.comp      # Scalar operations (a + scalar)
├── reduce.comp      # Reduction (sum, max, min)
└── compile.sh       # GLSL → SPIR-V compiler
```

#### Shader Strategy
- **Generalized shaders** with push constants for operation type
- **Push Constants**: `{ uint op_code; float scalar; }`
- **Work groups**: 256 threads per workgroup
- **Dispatch**: ceil(count / 256) workgroups

Example unary shader:
```glsl
#version 450
layout (local_size_x = 256) in;

layout(push_constant) uniform PushConstants {
    uint op_code;
} pc;

layout(std430, binding = 0) buffer A { float a[]; };
layout(std430, binding = 1) buffer Dst { float dst[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= a.length()) return;

    switch (pc.op_code) {
        case 0: dst[i] = -a[i]; break;        // NEG
        case 1: dst[i] = abs(a[i]); break;    // ABS
        case 2: dst[i] = sqrt(a[i]); break;   // SQRT
        case 3: dst[i] = exp(a[i]); break;    // EXP
        case 4: dst[i] = log(a[i]); break;    // LOG
        case 5: dst[i] = tanh(a[i]); break;   // TANH
        case 6: dst[i] = max(0.0, a[i]); break; // RELU
        case 7: dst[i] = sin(a[i]); break;    // SIN
        case 8: dst[i] = cos(a[i]); break;    // COS
        // ... etc
    }
}
```

### 4. CPU Parallelism

Use OpenMP for multi-threading:
```c
#pragma omp parallel for
for (uint32_t i = 0; i < count; i++) {
    pd[i] = pa[i] + pb[i];
}
```

Compile flags: `-fopenmp`

### 5. Mode Selection Logic

```c
static int global_mode = ADAMAH_MODE_CPU;  // default

int resolve_mode(int mode_override, uint32_t count) {
    int mode = (mode_override < 0) ? global_mode : mode_override;

    if (mode == ADAMAH_MODE_AUTO) {
        // Auto-select: GPU for large arrays (> 10k elements)
        return (count > 10000) ? ADAMAH_MODE_GPU : ADAMAH_MODE_CPU;
    }
    return mode;
}
```

### 6. GPU Dispatch Flow

```
vop2_ex(ADD, "c", "a", "b", 1000000, GPU)
  │
  ├→ resolve_mode() → GPU
  │
  ├→ find_buffer("a") → NamedBuffer *na
  ├→ find_buffer("b") → NamedBuffer *nb
  ├→ get_or_create_buffer("c", 1000000) → NamedBuffer *nd
  │
  ├→ vkCmdBindPipeline(cmd, binary_pipeline)
  ├→ vkCmdPushConstants(cmd, {op_code: ADD})
  ├→ vkCmdBindDescriptorSets(cmd, {na->buffer, nb->buffer, nd->buffer})
  ├→ vkCmdDispatch(cmd, ceil(1000000/256), 1, 1)
  │
  ├→ vkQueueSubmit(queue, cmd)
  └→ vkWaitForFences(fence)
```

### 7. Memory Considerations

**Current:** `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | HOST_COHERENT_BIT`
- Pro: Easy CPU access
- Con: Slower GPU access

**GPU Mode:** Should use `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`
- Pro: Fast GPU access
- Con: Need staging buffers for inject/extract

**Solution:** Dual allocation strategy
- CPU mode: host-visible memory (current)
- GPU mode: device-local + staging buffers

### 8. Pipeline Setup

```c
typedef struct {
    VkDescriptorSetLayout desc_layout;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;
    VkDescriptorPool desc_pool;
} ComputePipeline;

static ComputePipeline unary_pipe;
static ComputePipeline binary_pipe;
static ComputePipeline scalar_pipe;

void init_pipelines() {
    unary_pipe = create_compute_pipeline("shaders/unary.spv");
    binary_pipe = create_compute_pipeline("shaders/binary.spv");
    scalar_pipe = create_compute_pipeline("shaders/scalar.spv");
}
```

### 9. Implementation Phases

**Phase 1: Infrastructure**
- Add mode selection (set_mode, get_mode)
- Create shader directory structure
- Setup SPIR-V compilation

**Phase 2: GPU Core**
- Create unary/binary compute shaders
- Implement GPU dispatch system
- Add descriptor set management

**Phase 3: CPU Parallel**
- Add OpenMP directives
- Benchmark single vs multi-threaded

**Phase 4: Polish**
- Auto mode heuristics
- Performance benchmarks
- Documentation

## File Structure

```
ADAMAH/
├── LICENSE
├── README.md
├── pyproject.toml
├── DESIGN.md              # This file
└── src/
    └── adamah/
        ├── __init__.py
        ├── adamah.c
        ├── adamah.h
        ├── test.c
        └── shaders/
            ├── unary.comp      # GLSL source
            ├── binary.comp
            ├── scalar.comp
            ├── unary.spv       # Compiled SPIR-V
            ├── binary.spv
            ├── scalar.spv
            └── compile.sh      # glslc compiler script
```

## Backward Compatibility

All existing code continues to work:
```python
# Old code (CPU by default)
import adamah
adamah.init()
adamah.put("a", [1,2,3,4])
adamah.add("c", "a", "b", 4)  # runs on CPU
result = adamah.get("c")
adamah.shutdown()
```

New capabilities:
```python
# Enable GPU globally
adamah.set_mode('gpu')
adamah.add("c", "a", "b", 1000000)  # runs on GPU

# Or per-operation override
adamah.set_mode('cpu')
adamah.add("c", "a", "b", 1000000, mode='gpu')  # override to GPU
```

## Performance Expectations

| Operation | Size | CPU (single) | CPU (OpenMP 8-core) | GPU (Vulkan) |
|-----------|------|--------------|---------------------|--------------|
| add       | 1M   | ~5ms         | ~1ms                | ~0.2ms       |
| sin       | 1M   | ~20ms        | ~3ms                | ~0.5ms       |
| reduce    | 1M   | ~3ms         | ~0.5ms              | ~0.3ms       |

(Approximate - needs benchmarking)
