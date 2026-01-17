/*
 * ADAMAH v3 - Named Buffers + Auto Management
 * 
 * AGPL-3.0 - Sam 2026
 */

#include "adamah.h"
#include <vulkan/vulkan.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define MAX_BUFFERS 256
#define MAX_MAPS 16
#define MAX_NAME_LEN 64
#define LOCAL_SIZE 256

// ============================================
// Named Buffer
// ============================================
typedef struct {
    char name[MAX_NAME_LEN];
    VkBuffer buf;
    VkDeviceMemory mem;
    float* ptr;
    uint32_t capacity;  // allocated size
    uint32_t size;      // used size
} NamedBuffer;

// ============================================
// Map
// ============================================
typedef struct {
    int active;
    uint32_t word_size;
    uint32_t pack_size;
    uint32_t n_packs;
    uint64_t total_elems;
    VkBuffer data_buf;
    VkDeviceMemory data_mem;
    void* data_ptr;
} Map;

// ============================================
// Global Context
// ============================================
static struct {
    int initialized;
    int mode;  // Execution mode (CPU/GPU/AUTO)
    VkInstance instance;
    VkPhysicalDevice phys;
    VkDevice device;
    VkQueue queue;
    uint32_t queue_family;
    VkCommandPool cmd_pool;
    VkCommandBuffer cmd;
    VkFence fence;
    NamedBuffer buffers[MAX_BUFFERS];
    int buffer_count;
    Map maps[MAX_MAPS];
} ctx = {0};

// ============================================
// Helpers
// ============================================

static uint32_t find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(ctx.phys, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) && (mem_props.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    return UINT32_MAX;
}

static NamedBuffer* find_buffer(const char* name) {
    for (int i = 0; i < ctx.buffer_count; i++) {
        if (strcmp(ctx.buffers[i].name, name) == 0) return &ctx.buffers[i];
    }
    return NULL;
}

static NamedBuffer* get_or_create_buffer(const char* name, uint32_t min_size) {
    NamedBuffer* nb = find_buffer(name);
    
    if (nb) {
        // Resize if needed
        if (nb->capacity < min_size) {
            if (nb->ptr) vkUnmapMemory(ctx.device, nb->mem);
            if (nb->buf) vkDestroyBuffer(ctx.device, nb->buf, NULL);
            if (nb->mem) vkFreeMemory(ctx.device, nb->mem, NULL);
            nb->capacity = 0;
            nb->ptr = NULL;
        } else {
            nb->size = min_size;
            return nb;
        }
    } else {
        // Create new
        if (ctx.buffer_count >= MAX_BUFFERS) return NULL;
        nb = &ctx.buffers[ctx.buffer_count++];
        memset(nb, 0, sizeof(*nb));
        strncpy(nb->name, name, MAX_NAME_LEN - 1);
    }
    
    // Allocate
    uint32_t alloc_size = min_size < 1024 ? 1024 : min_size;  // min 1KB
    VkDeviceSize byte_size = alloc_size * sizeof(float);
    
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = byte_size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    };
    if (vkCreateBuffer(ctx.device, &bci, NULL, &nb->buf) != VK_SUCCESS) return NULL;
    
    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(ctx.device, nb->buf, &reqs);
    
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = reqs.size,
        .memoryTypeIndex = find_memory_type(reqs.memoryTypeBits, 
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
    };
    if (vkAllocateMemory(ctx.device, &mai, NULL, &nb->mem) != VK_SUCCESS) {
        vkDestroyBuffer(ctx.device, nb->buf, NULL);
        return NULL;
    }
    
    vkBindBufferMemory(ctx.device, nb->buf, nb->mem, 0);
    vkMapMemory(ctx.device, nb->mem, 0, byte_size, 0, (void**)&nb->ptr);
    
    nb->capacity = alloc_size;
    nb->size = min_size;
    memset(nb->ptr, 0, byte_size);
    
    return nb;
}

// ============================================
// Init / Shutdown
// ============================================

int adamah_init(void) {
    if (ctx.initialized) return ADAMAH_OK;
    
    VkApplicationInfo ai = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "ADAMAH",
        .apiVersion = VK_API_VERSION_1_0
    };
    VkInstanceCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &ai
    };
    if (vkCreateInstance(&ici, NULL, &ctx.instance) != VK_SUCCESS) return ADAMAH_ERR_VULKAN;
    
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(ctx.instance, &count, NULL);
    if (count == 0) return ADAMAH_ERR_VULKAN;
    VkPhysicalDevice* devs = malloc(count * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(ctx.instance, &count, devs);
    ctx.phys = devs[0];
    free(devs);
    
    uint32_t qcount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.phys, &qcount, NULL);
    VkQueueFamilyProperties* qprops = malloc(qcount * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.phys, &qcount, qprops);
    ctx.queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < qcount; i++) {
        if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { ctx.queue_family = i; break; }
    }
    free(qprops);
    if (ctx.queue_family == UINT32_MAX) return ADAMAH_ERR_VULKAN;
    
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = ctx.queue_family,
        .queueCount = 1,
        .pQueuePriorities = &priority
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci
    };
    if (vkCreateDevice(ctx.phys, &dci, NULL, &ctx.device) != VK_SUCCESS) return ADAMAH_ERR_VULKAN;
    vkGetDeviceQueue(ctx.device, ctx.queue_family, 0, &ctx.queue);
    
    VkCommandPoolCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = ctx.queue_family
    };
    vkCreateCommandPool(ctx.device, &cpci, NULL, &ctx.cmd_pool);
    
    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx.cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    vkAllocateCommandBuffers(ctx.device, &cbai, &ctx.cmd);
    
    VkFenceCreateInfo fci = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT
    };
    vkCreateFence(ctx.device, &fci, NULL, &ctx.fence);
    
    ctx.initialized = 1;
    return ADAMAH_OK;
}

void adamah_shutdown(void) {
    if (!ctx.initialized) return;
    vkDeviceWaitIdle(ctx.device);
    
    for (int i = 0; i < ctx.buffer_count; i++) {
        NamedBuffer* nb = &ctx.buffers[i];
        if (nb->ptr) vkUnmapMemory(ctx.device, nb->mem);
        if (nb->buf) vkDestroyBuffer(ctx.device, nb->buf, NULL);
        if (nb->mem) vkFreeMemory(ctx.device, nb->mem, NULL);
    }
    
    for (int i = 0; i < MAX_MAPS; i++) {
        if (ctx.maps[i].active) map_destroy(i);
    }
    
    if (ctx.fence) vkDestroyFence(ctx.device, ctx.fence, NULL);
    if (ctx.cmd_pool) vkDestroyCommandPool(ctx.device, ctx.cmd_pool, NULL);
    if (ctx.device) vkDestroyDevice(ctx.device, NULL);
    if (ctx.instance) vkDestroyInstance(ctx.instance, NULL);
    
    memset(&ctx, 0, sizeof(ctx));
}

// ============================================
// Mode Selection
// ============================================

void adamah_set_mode(int mode) {
    if (mode < ADAMAH_MODE_CPU || mode > ADAMAH_MODE_AUTO) return;
    ctx.mode = mode;
}

int adamah_get_mode(void) {
    return ctx.mode;
}

// Resolve actual mode (handle AUTO)
static int resolve_mode(int mode_override, uint32_t count) {
    int mode = (mode_override == ADAMAH_MODE_USE_GLOBAL) ? ctx.mode : mode_override;

    if (mode == ADAMAH_MODE_AUTO) {
        // Auto-select: GPU for large arrays (> 10k elements)
        // For now, use CPU since GPU not implemented yet
        return ADAMAH_MODE_CPU;
        // TODO: return (count > 10000) ? ADAMAH_MODE_GPU : ADAMAH_MODE_CPU;
    }

    // For now, force CPU since GPU not implemented
    if (mode == ADAMAH_MODE_GPU) {
        fprintf(stderr, "Warning: GPU mode not yet implemented, using CPU\n");
        return ADAMAH_MODE_CPU;
    }

    return mode;
}

// ============================================
// Named Buffers
// ============================================

int inject(const char* name, const float* data, uint32_t count) {
    if (!ctx.initialized || !name || !data || !count) return ADAMAH_ERR_INVALID;
    NamedBuffer* nb = get_or_create_buffer(name, count);
    if (!nb) return ADAMAH_ERR_MEMORY;
    memcpy(nb->ptr, data, count * sizeof(float));
    return ADAMAH_OK;
}

int extract(const char* name, float* data, uint32_t count) {
    if (!ctx.initialized || !name || !data) return ADAMAH_ERR_INVALID;
    NamedBuffer* nb = find_buffer(name);
    if (!nb) return ADAMAH_ERR_NOT_FOUND;
    uint32_t n = count < nb->size ? count : nb->size;
    memcpy(data, nb->ptr, n * sizeof(float));
    return ADAMAH_OK;
}

uint32_t bufsize(const char* name) {
    if (!ctx.initialized || !name) return 0;
    NamedBuffer* nb = find_buffer(name);
    return nb ? nb->size : 0;
}

// ============================================
// Vector Ops (CPU fallback, simple)
// ============================================

int vop2(uint32_t op, const char* dst, const char* a, const char* b, uint32_t count) {
    if (!ctx.initialized) return ADAMAH_ERR_INVALID;
    NamedBuffer *na = find_buffer(a), *nb_src = find_buffer(b);
    if (!na || !nb_src) return ADAMAH_ERR_NOT_FOUND;
    NamedBuffer* nd = get_or_create_buffer(dst, count);
    if (!nd) return ADAMAH_ERR_MEMORY;
    
    float *pa = na->ptr, *pb = nb_src->ptr, *pd = nd->ptr;
    switch (op) {
        case VOP_ADD:   for (uint32_t i = 0; i < count; i++) pd[i] = pa[i] + pb[i]; break;
        case VOP_SUB:   for (uint32_t i = 0; i < count; i++) pd[i] = pa[i] - pb[i]; break;
        case VOP_MUL:   for (uint32_t i = 0; i < count; i++) pd[i] = pa[i] * pb[i]; break;
        case VOP_DIV:   for (uint32_t i = 0; i < count; i++) pd[i] = pa[i] / pb[i]; break;
        case VOP_POW:   for (uint32_t i = 0; i < count; i++) pd[i] = powf(pa[i], pb[i]); break;
        case VOP_ATAN2: for (uint32_t i = 0; i < count; i++) pd[i] = atan2f(pa[i], pb[i]); break;
        case VOP_MOD:   for (uint32_t i = 0; i < count; i++) pd[i] = fmodf(pa[i], pb[i]); break;
        case VOP_MIN:   for (uint32_t i = 0; i < count; i++) pd[i] = pa[i] < pb[i] ? pa[i] : pb[i]; break;
        case VOP_MAX:   for (uint32_t i = 0; i < count; i++) pd[i] = pa[i] > pb[i] ? pa[i] : pb[i]; break;
        default: return ADAMAH_ERR_INVALID;
    }
    return ADAMAH_OK;
}

int vop1(uint32_t op, const char* dst, const char* a, uint32_t count) {
    if (!ctx.initialized) return ADAMAH_ERR_INVALID;
    NamedBuffer* na = find_buffer(a);
    if (!na) return ADAMAH_ERR_NOT_FOUND;
    NamedBuffer* nd = get_or_create_buffer(dst, count);
    if (!nd) return ADAMAH_ERR_MEMORY;
    
    float *pa = na->ptr, *pd = nd->ptr;
    for (uint32_t i = 0; i < count; i++) {
        float x = pa[i];
        switch (op) {
            case VOP_NEG:   pd[i] = -x; break;
            case VOP_ABS:   pd[i] = fabsf(x); break;
            case VOP_SQRT:  pd[i] = sqrtf(x); break;
            case VOP_EXP:   pd[i] = expf(x); break;
            case VOP_LOG:   pd[i] = logf(x); break;
            case VOP_TANH:  pd[i] = tanhf(x); break;
            case VOP_RELU:  pd[i] = x > 0 ? x : 0; break;
            case VOP_GELU:  pd[i] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x*x*x))); break;
            case VOP_SIN:   pd[i] = sinf(x); break;
            case VOP_COS:   pd[i] = cosf(x); break;
            case VOP_RECIP: pd[i] = 1.0f / x; break;
            case VOP_SQR:   pd[i] = x * x; break;
            case VOP_COPY:  pd[i] = x; break;
            case VOP_TAN:   pd[i] = tanf(x); break;
            case VOP_ASIN:  pd[i] = asinf(x); break;
            case VOP_ACOS:  pd[i] = acosf(x); break;
            case VOP_ATAN:  pd[i] = atanf(x); break;
            case VOP_SINH:  pd[i] = sinhf(x); break;
            case VOP_COSH:  pd[i] = coshf(x); break;
            case VOP_LOG2:  pd[i] = log2f(x); break;
            case VOP_LOG10: pd[i] = log10f(x); break;
            case VOP_EXP2:  pd[i] = exp2f(x); break;
            case VOP_FLOOR: pd[i] = floorf(x); break;
            case VOP_CEIL:  pd[i] = ceilf(x); break;
            case VOP_ROUND: pd[i] = roundf(x); break;
            case VOP_TRUNC: pd[i] = truncf(x); break;
            case VOP_SIGN:  pd[i] = (x > 0) - (x < 0); break;
            default: return ADAMAH_ERR_INVALID;
        }
    }
    return ADAMAH_OK;
}

int vops(uint32_t op, const char* dst, const char* a, float scalar, uint32_t count) {
    if (!ctx.initialized) return ADAMAH_ERR_INVALID;
    NamedBuffer* na = find_buffer(a);
    if (!na) return ADAMAH_ERR_NOT_FOUND;
    NamedBuffer* nd = get_or_create_buffer(dst, count);
    if (!nd) return ADAMAH_ERR_MEMORY;
    
    float *pa = na->ptr, *pd = nd->ptr;
    switch (op) {
        case VOP_ADD: for (uint32_t i = 0; i < count; i++) pd[i] = pa[i] + scalar; break;
        case VOP_SUB: for (uint32_t i = 0; i < count; i++) pd[i] = pa[i] - scalar; break;
        case VOP_MUL: for (uint32_t i = 0; i < count; i++) pd[i] = pa[i] * scalar; break;
        case VOP_DIV: for (uint32_t i = 0; i < count; i++) pd[i] = pa[i] / scalar; break;
        default: return ADAMAH_ERR_INVALID;
    }
    return ADAMAH_OK;
}

int vreduce(uint32_t op, const char* dst, const char* a, uint32_t count) {
    if (!ctx.initialized) return ADAMAH_ERR_INVALID;
    NamedBuffer* na = find_buffer(a);
    if (!na) return ADAMAH_ERR_NOT_FOUND;
    NamedBuffer* nd = get_or_create_buffer(dst, 1);
    if (!nd) return ADAMAH_ERR_MEMORY;
    
    float* pa = na->ptr;
    float result = pa[0];
    switch (op) {
        case VRED_SUM:  result = 0; for (uint32_t i = 0; i < count; i++) result += pa[i]; break;
        case VRED_MAX:  for (uint32_t i = 1; i < count; i++) if (pa[i] > result) result = pa[i]; break;
        case VRED_MIN:  for (uint32_t i = 1; i < count; i++) if (pa[i] < result) result = pa[i]; break;
        case VRED_PROD: result = 1; for (uint32_t i = 0; i < count; i++) result *= pa[i]; break;
        case VRED_MEAN: result = 0; for (uint32_t i = 0; i < count; i++) result += pa[i]; result /= count; break;
    }
    nd->ptr[0] = result;
    return ADAMAH_OK;
}

int vdot(const char* dst, const char* a, const char* b, uint32_t count) {
    if (!ctx.initialized) return ADAMAH_ERR_INVALID;
    NamedBuffer *na = find_buffer(a), *nb_src = find_buffer(b);
    if (!na || !nb_src) return ADAMAH_ERR_NOT_FOUND;
    NamedBuffer* nd = get_or_create_buffer(dst, 1);
    if (!nd) return ADAMAH_ERR_MEMORY;
    
    float sum = 0;
    for (uint32_t i = 0; i < count; i++) sum += na->ptr[i] * nb_src->ptr[i];
    nd->ptr[0] = sum;
    return ADAMAH_OK;
}

int vsoftmax(const char* buf, uint32_t count) {
    if (!ctx.initialized) return ADAMAH_ERR_INVALID;
    NamedBuffer* nb = find_buffer(buf);
    if (!nb) return ADAMAH_ERR_NOT_FOUND;
    
    float* p = nb->ptr;
    float max_val = p[0];
    for (uint32_t i = 1; i < count; i++) if (p[i] > max_val) max_val = p[i];
    float sum = 0;
    for (uint32_t i = 0; i < count; i++) { p[i] = expf(p[i] - max_val); sum += p[i]; }
    for (uint32_t i = 0; i < count; i++) p[i] /= sum;
    return ADAMAH_OK;
}

int vmatvec(const char* dst, const char* mat, const char* vec, uint32_t rows, uint32_t cols) {
    if (!ctx.initialized) return ADAMAH_ERR_INVALID;
    NamedBuffer *nm = find_buffer(mat), *nv = find_buffer(vec);
    if (!nm || !nv) return ADAMAH_ERR_NOT_FOUND;
    NamedBuffer* nd = get_or_create_buffer(dst, rows);
    if (!nd) return ADAMAH_ERR_MEMORY;
    
    for (uint32_t i = 0; i < rows; i++) {
        float sum = 0;
        for (uint32_t j = 0; j < cols; j++) sum += nm->ptr[i * cols + j] * nv->ptr[j];
        nd->ptr[i] = sum;
    }
    return ADAMAH_OK;
}

// ============================================
// Calculus / Numerical
// ============================================

int vcumsum(const char* dst, const char* a, uint32_t count) {
    if (!ctx.initialized) return ADAMAH_ERR_INVALID;
    NamedBuffer* na = find_buffer(a);
    if (!na) return ADAMAH_ERR_NOT_FOUND;
    NamedBuffer* nd = get_or_create_buffer(dst, count);
    if (!nd) return ADAMAH_ERR_MEMORY;
    
    float sum = 0;
    for (uint32_t i = 0; i < count; i++) {
        sum += na->ptr[i];
        nd->ptr[i] = sum;
    }
    return ADAMAH_OK;
}

int vcumprod(const char* dst, const char* a, uint32_t count) {
    if (!ctx.initialized) return ADAMAH_ERR_INVALID;
    NamedBuffer* na = find_buffer(a);
    if (!na) return ADAMAH_ERR_NOT_FOUND;
    NamedBuffer* nd = get_or_create_buffer(dst, count);
    if (!nd) return ADAMAH_ERR_MEMORY;
    
    float prod = 1;
    for (uint32_t i = 0; i < count; i++) {
        prod *= na->ptr[i];
        nd->ptr[i] = prod;
    }
    return ADAMAH_OK;
}

int vdiff(const char* dst, const char* a, uint32_t count) {
    if (!ctx.initialized || count < 2) return ADAMAH_ERR_INVALID;
    NamedBuffer* na = find_buffer(a);
    if (!na) return ADAMAH_ERR_NOT_FOUND;
    NamedBuffer* nd = get_or_create_buffer(dst, count - 1);
    if (!nd) return ADAMAH_ERR_MEMORY;
    
    for (uint32_t i = 0; i < count - 1; i++) {
        nd->ptr[i] = na->ptr[i + 1] - na->ptr[i];
    }
    return ADAMAH_OK;
}

int vintegrate(const char* dst, const char* a, float dx, uint32_t count) {
    if (!ctx.initialized) return ADAMAH_ERR_INVALID;
    NamedBuffer* na = find_buffer(a);
    if (!na) return ADAMAH_ERR_NOT_FOUND;
    NamedBuffer* nd = get_or_create_buffer(dst, count);
    if (!nd) return ADAMAH_ERR_MEMORY;
    
    float sum = 0;
    for (uint32_t i = 0; i < count; i++) {
        sum += na->ptr[i] * dx;
        nd->ptr[i] = sum;
    }
    return ADAMAH_OK;
}

int vderivative(const char* dst, const char* a, float dx, uint32_t count) {
    if (!ctx.initialized || count < 2) return ADAMAH_ERR_INVALID;
    NamedBuffer* na = find_buffer(a);
    if (!na) return ADAMAH_ERR_NOT_FOUND;
    NamedBuffer* nd = get_or_create_buffer(dst, count - 1);
    if (!nd) return ADAMAH_ERR_MEMORY;
    
    for (uint32_t i = 0; i < count - 1; i++) {
        nd->ptr[i] = (na->ptr[i + 1] - na->ptr[i]) / dx;
    }
    return ADAMAH_OK;
}

int vlinspace(const char* dst, float start, float stop, uint32_t count) {
    if (!ctx.initialized || count < 2) return ADAMAH_ERR_INVALID;
    NamedBuffer* nd = get_or_create_buffer(dst, count);
    if (!nd) return ADAMAH_ERR_MEMORY;
    
    float step = (stop - start) / (count - 1);
    for (uint32_t i = 0; i < count; i++) {
        nd->ptr[i] = start + i * step;
    }
    return ADAMAH_OK;
}

int varange(const char* dst, float start, float step, uint32_t count) {
    if (!ctx.initialized) return ADAMAH_ERR_INVALID;
    NamedBuffer* nd = get_or_create_buffer(dst, count);
    if (!nd) return ADAMAH_ERR_MEMORY;
    
    for (uint32_t i = 0; i < count; i++) {
        nd->ptr[i] = start + i * step;
    }
    return ADAMAH_OK;
}

// ============================================
// Maps
// ============================================

int map_init(uint32_t id, uint32_t word_size, uint32_t pack_size, uint32_t n_packs) {
    if (!ctx.initialized || id >= MAX_MAPS || ctx.maps[id].active) return ADAMAH_ERR_INVALID;
    
    Map* m = &ctx.maps[id];
    m->word_size = word_size;
    m->pack_size = pack_size;
    m->n_packs = n_packs;
    m->total_elems = (uint64_t)pack_size * n_packs;
    
    VkDeviceSize byte_size = m->total_elems * word_size;
    
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = byte_size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    };
    if (vkCreateBuffer(ctx.device, &bci, NULL, &m->data_buf) != VK_SUCCESS) return ADAMAH_ERR_VULKAN;
    
    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(ctx.device, m->data_buf, &reqs);
    
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = reqs.size,
        .memoryTypeIndex = find_memory_type(reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
    };
    if (vkAllocateMemory(ctx.device, &mai, NULL, &m->data_mem) != VK_SUCCESS) {
        vkDestroyBuffer(ctx.device, m->data_buf, NULL);
        return ADAMAH_ERR_MEMORY;
    }
    
    vkBindBufferMemory(ctx.device, m->data_buf, m->data_mem, 0);
    vkMapMemory(ctx.device, m->data_mem, 0, byte_size, 0, &m->data_ptr);
    memset(m->data_ptr, 0, byte_size);
    
    m->active = 1;
    return ADAMAH_OK;
}

int map_destroy(uint32_t id) {
    if (!ctx.initialized || id >= MAX_MAPS || !ctx.maps[id].active) return ADAMAH_ERR_INVALID;
    Map* m = &ctx.maps[id];
    if (m->data_ptr) vkUnmapMemory(ctx.device, m->data_mem);
    if (m->data_buf) vkDestroyBuffer(ctx.device, m->data_buf, NULL);
    if (m->data_mem) vkFreeMemory(ctx.device, m->data_mem, NULL);
    memset(m, 0, sizeof(*m));
    return ADAMAH_OK;
}

int map_clear(uint32_t id) {
    if (!ctx.initialized || id >= MAX_MAPS || !ctx.maps[id].active) return ADAMAH_ERR_INVALID;
    Map* m = &ctx.maps[id];
    memset(m->data_ptr, 0, m->total_elems * m->word_size);
    return ADAMAH_OK;
}

uint64_t map_limit(uint32_t id) {
    if (!ctx.initialized || id >= MAX_MAPS || !ctx.maps[id].active) return 0;
    return ctx.maps[id].total_elems - 1;
}

int mscatter(uint32_t id, const char* locs_name, const char* vals_name, uint32_t count) {
    if (!ctx.initialized || id >= MAX_MAPS || !ctx.maps[id].active) return ADAMAH_ERR_INVALID;
    NamedBuffer *locs_buf = find_buffer(locs_name);
    NamedBuffer *vals_buf = find_buffer(vals_name);
    if (!locs_buf || !vals_buf) return ADAMAH_ERR_NOT_FOUND;
    
    Map* m = &ctx.maps[id];
    uint32_t word_floats = m->word_size / sizeof(float);
    float* data = (float*)m->data_ptr;
    
    for (uint32_t i = 0; i < count; i++) {
        uint64_t loc = (uint64_t)locs_buf->ptr[i];
        if (loc < m->total_elems) {
            for (uint32_t w = 0; w < word_floats; w++) {
                data[loc * word_floats + w] = vals_buf->ptr[i * word_floats + w];
            }
        }
    }
    return ADAMAH_OK;
}

int mgather(uint32_t id, const char* locs_name, const char* dst_name, uint32_t count) {
    if (!ctx.initialized || id >= MAX_MAPS || !ctx.maps[id].active) return ADAMAH_ERR_INVALID;
    NamedBuffer* locs_buf = find_buffer(locs_name);
    if (!locs_buf) return ADAMAH_ERR_NOT_FOUND;
    
    Map* m = &ctx.maps[id];
    uint32_t word_floats = m->word_size / sizeof(float);
    NamedBuffer* dst_buf = get_or_create_buffer(dst_name, count * word_floats);
    if (!dst_buf) return ADAMAH_ERR_MEMORY;
    
    float* data = (float*)m->data_ptr;
    
    for (uint32_t i = 0; i < count; i++) {
        uint64_t loc = (uint64_t)locs_buf->ptr[i];
        if (loc < m->total_elems) {
            for (uint32_t w = 0; w < word_floats; w++) {
                dst_buf->ptr[i * word_floats + w] = data[loc * word_floats + w];
            }
        } else {
            for (uint32_t w = 0; w < word_floats; w++) {
                dst_buf->ptr[i * word_floats + w] = 0;
            }
        }
    }
    return ADAMAH_OK;
}

int map_save(uint32_t id, const char* path) {
    if (!ctx.initialized || id >= MAX_MAPS || !ctx.maps[id].active) return ADAMAH_ERR_INVALID;
    Map* m = &ctx.maps[id];
    FILE* f = fopen(path, "wb");
    if (!f) return ADAMAH_ERR_INVALID;
    fwrite(m->data_ptr, 1, m->total_elems * m->word_size, f);
    fclose(f);
    return ADAMAH_OK;
}

int map_load(uint32_t id, const char* path) {
    if (!ctx.initialized || id >= MAX_MAPS || !ctx.maps[id].active) return ADAMAH_ERR_INVALID;
    Map* m = &ctx.maps[id];
    FILE* f = fopen(path, "rb");
    if (!f) return ADAMAH_ERR_INVALID;
    fread(m->data_ptr, 1, m->total_elems * m->word_size, f);
    fclose(f);
    return ADAMAH_OK;
}
