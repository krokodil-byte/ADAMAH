/*
 * ADAMAH - GPU Memory Maps + Vector Math
 * Single-header library for scatter/gather and compute operations
 * 
 * Usage:
 *   #include "adamah.h"
 *   adamah_init();
 *   map_init(0, 32, 256, 256);
 *   vbuf_alloc(0, 1024);
 *   adamah_shutdown();
 * 
 * Build:
 *   gcc myapp.c adamah.c -o myapp -lvulkan -lm
 *
 * CC BY-NC 4.0 - Sam 2026
 */

#ifndef ADAMAH_H
#define ADAMAH_H

#include <stdint.h>

#define ADAMAH_OK           0
#define ADAMAH_ERR_VULKAN  -1
#define ADAMAH_ERR_MEMORY  -2
#define ADAMAH_ERR_INVALID -3
#define ADAMAH_ERR_BOUNDS  -4
#define ADAMAH_ERR_NOT_FOUND -5
#define ADAMAH_ERR_EXISTS  -6

#define VOP_ADD 0
#define VOP_SUB 1
#define VOP_MUL 2
#define VOP_DIV 3
#define VOP_NEG 10
#define VOP_ABS 11
#define VOP_SQRT 12
#define VOP_EXP 13
#define VOP_LOG 14
#define VOP_TANH 15
#define VOP_RELU 16
#define VOP_GELU 17
#define VOP_SIN 18
#define VOP_COS 19
#define VOP_RECIP 20
#define VOP_SQR 21
#define VOP_COPY 22

#define VRED_SUM 0
#define VRED_MAX 1
#define VRED_MIN 2

// Core
int adamah_init(void);
void adamah_shutdown(void);

// Maps
int map_init(uint32_t id, uint32_t word_size, uint32_t pack_size, uint32_t n_packs);
int map_destroy(uint32_t id);
int map_clear(uint32_t id);
uint64_t map_limit(uint32_t id);
int mscatter(uint32_t id, const uint64_t* locs, const void* vals, uint32_t count);
int mgather(uint32_t id, const uint64_t* locs, void* out, uint32_t count);
int map_save(uint32_t id, const char* path);
int map_load(uint32_t id, const char* path);

// VBufs
int vbuf_alloc(uint32_t id, uint32_t n_floats);
int vbuf_free(uint32_t id);
int vbuf_upload(uint32_t id, const float* data, uint32_t offset, uint32_t count);
int vbuf_download(uint32_t id, float* data, uint32_t offset, uint32_t count);
int vbuf_zero(uint32_t id, uint32_t offset, uint32_t count);
uint32_t vbuf_size(uint32_t id);

// Math
int vop2(uint32_t op, uint32_t dst, uint32_t a, uint32_t b, uint32_t count);
int vop_scalar(uint32_t op, uint32_t dst, uint32_t a, float scalar, uint32_t count);
int vop1(uint32_t op, uint32_t dst, uint32_t a, uint32_t count);
int vreduce(uint32_t op, uint32_t dst, uint32_t a, uint32_t count);
int vdot(uint32_t dst, uint32_t a, uint32_t b, uint32_t count);
int vfma(uint32_t dst, uint32_t a, uint32_t b, uint32_t c, uint32_t count);
int vsoftmax(uint32_t buf, uint32_t offset, uint32_t count);
int vmatvec(uint32_t dst, uint32_t mat, uint32_t vec, uint32_t rows, uint32_t cols);

#endif
