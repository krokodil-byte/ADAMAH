/*
 * ADAMAH v3 - Named Buffers + Auto Management
 * As simple as possible, but complete
 * 
 * AGPL-3.0 - Sam 2026
 */

#ifndef ADAMAH_H
#define ADAMAH_H

#include <stdint.h>

// Error codes
#define ADAMAH_OK           0
#define ADAMAH_ERR_VULKAN  -1
#define ADAMAH_ERR_MEMORY  -2
#define ADAMAH_ERR_INVALID -3
#define ADAMAH_ERR_NOT_FOUND -4

// Execution modes
#define ADAMAH_MODE_CPU  0
#define ADAMAH_MODE_GPU  1
#define ADAMAH_MODE_AUTO 2
#define ADAMAH_MODE_USE_GLOBAL -1  // Use global mode (for _ex functions)

// Binary ops
#define VOP_ADD  0
#define VOP_SUB  1
#define VOP_MUL  2
#define VOP_DIV  3

// Unary ops
#define VOP_NEG   10
#define VOP_ABS   11
#define VOP_SQRT  12
#define VOP_EXP   13
#define VOP_LOG   14
#define VOP_TANH  15
#define VOP_RELU  16
#define VOP_GELU  17
#define VOP_SIN   18
#define VOP_COS   19
#define VOP_RECIP 20
#define VOP_SQR   21
#define VOP_COPY  22
#define VOP_TAN   23
#define VOP_ASIN  24
#define VOP_ACOS  25
#define VOP_ATAN  26
#define VOP_SINH  27
#define VOP_COSH  28
#define VOP_LOG2  29
#define VOP_LOG10 30
#define VOP_EXP2  31
#define VOP_FLOOR 32
#define VOP_CEIL  33
#define VOP_ROUND 34
#define VOP_TRUNC 35
#define VOP_SIGN  36

// Binary ops (extended)
#define VOP_POW   50
#define VOP_ATAN2 51
#define VOP_MOD   52
#define VOP_MIN   53
#define VOP_MAX   54

// Reduce ops
#define VRED_SUM  0
#define VRED_MAX  1
#define VRED_MIN  2
#define VRED_PROD 3
#define VRED_MEAN 4

// ============================================
// Core
// ============================================
int adamah_init(void);
void adamah_shutdown(void);

// Mode selection (CPU/GPU/AUTO)
void adamah_set_mode(int mode);
int adamah_get_mode(void);

// ============================================
// Named Buffers (auto-managed)
// ============================================

// Inject data CPU → GPU (creates/resizes buffer automatically)
int inject(const char* name, const float* data, uint32_t count);

// Extract data GPU → CPU
int extract(const char* name, float* data, uint32_t count);

// Get buffer size (0 if not exists)
uint32_t bufsize(const char* name);

// ============================================
// Vector Ops (named buffers)
// ============================================

// dst = a OP b
int vop2(uint32_t op, const char* dst, const char* a, const char* b, uint32_t count);

// dst = OP(a)
int vop1(uint32_t op, const char* dst, const char* a, uint32_t count);

// dst = a OP scalar
int vops(uint32_t op, const char* dst, const char* a, float scalar, uint32_t count);

// dst[0] = reduce(a)
int vreduce(uint32_t op, const char* dst, const char* a, uint32_t count);

// Extended versions with mode override
int vop1_ex(uint32_t op, const char* dst, const char* a, uint32_t count, int mode);
int vop2_ex(uint32_t op, const char* dst, const char* a, const char* b, uint32_t count, int mode);
int vops_ex(uint32_t op, const char* dst, const char* a, float scalar, uint32_t count, int mode);

// dst[0] = dot(a, b)
int vdot(const char* dst, const char* a, const char* b, uint32_t count);

// Softmax in-place
int vsoftmax(const char* buf, uint32_t count);

// dst = mat @ vec
int vmatvec(const char* dst, const char* mat, const char* vec, uint32_t rows, uint32_t cols);

// ============================================
// Calculus / Numerical
// ============================================

// Cumulative sum: dst[i] = sum(a[0..i])
int vcumsum(const char* dst, const char* a, uint32_t count);

// Cumulative product: dst[i] = prod(a[0..i])
int vcumprod(const char* dst, const char* a, uint32_t count);

// Difference: dst[i] = a[i+1] - a[i] (count-1 elements)
int vdiff(const char* dst, const char* a, uint32_t count);

// Numerical integral: dst[i] = sum(a[0..i]) * dx
int vintegrate(const char* dst, const char* a, float dx, uint32_t count);

// Numerical derivative: dst[i] = (a[i+1] - a[i]) / dx
int vderivative(const char* dst, const char* a, float dx, uint32_t count);

// Linspace: dst = [start, start+step, start+2*step, ...]
int vlinspace(const char* dst, float start, float stop, uint32_t count);

// Arange: dst = [start, start+step, ...] until stop
int varange(const char* dst, float start, float step, uint32_t count);

// ============================================
// Memory Maps
// ============================================

// Create map
int map_init(uint32_t id, uint32_t word_size, uint32_t pack_size, uint32_t n_packs);
int map_destroy(uint32_t id);
int map_clear(uint32_t id);
uint64_t map_limit(uint32_t id);

// Scatter: map[locs] = vals
int mscatter(uint32_t id, const char* locs, const char* vals, uint32_t count);

// Gather: dst = map[locs]
int mgather(uint32_t id, const char* locs, const char* dst, uint32_t count);

// Persistence
int map_save(uint32_t id, const char* path);
int map_load(uint32_t id, const char* path);

#endif
