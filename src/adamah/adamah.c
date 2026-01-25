/*
 * ADAMAH v5.0 - Map-Centric GPU Compute
 * 
 * Pure GPU operations on Memory Maps
 * scatter/gather for CPU I/O
 *
 * CC BY-NC 4.0 - Samuele Scuglia - 2026-01-18
 */

#include <vulkan/vulkan.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

// Error codes
#define ADAMAH_OK           0
#define ADAMAH_ERR_VULKAN  -1
#define ADAMAH_ERR_MEMORY  -2
#define ADAMAH_ERR_INVALID -3
#define ADAMAH_ERR_NOT_FOUND -4

#define MAX_MAPS 16
#define MAX_BUFS 64
#define MAX_RES 2048
#define MAX_FREE 4096
#define LOCAL_SIZE 256
#define CMD_RING 4
#define HOT_POOL_BYTES_DEFAULT (512ull << 20)
#define COLD_POOL_BYTES_DEFAULT (512ull << 20)

#define RES_TYPE_CVAR 1
#define RES_TYPE_LOCS 2

// Forward declarations
int map_destroy(uint32_t id);

// ============================================
// Structures
// ============================================

// Internal GPU buffer (for locs, temp data)
typedef struct {
    char name[64];
    VkBuffer buf;
    VkDeviceMemory mem;
    void* ptr;          // Mapped if HOST_VISIBLE
    VkDeviceSize bytes_capacity;
    uint32_t elem_size; // Bytes per element (for bookkeeping)
    int device_local;   // 1 = VRAM, 0 = HOST_VISIBLE
    VkBufferUsageFlags usage;
} GpuBuf;

static GpuBuf* get_or_create_buf_ex(const char* base_name, uint32_t n_elems, uint32_t elem_size,
                                    int device_local, VkBufferUsageFlags usage);

// Memory Map - the core data structure
typedef struct {
    int active;
    uint32_t word_size;   // Bytes per word (4 = float)
    uint32_t pack_size;   // Words per pack
    uint32_t n_packs;     // Number of packs
    uint64_t total_bytes;

    VkBuffer buf;
    VkDeviceMemory mem;

    // Staging for CPU transfer
    VkBuffer staging;
    VkDeviceMemory staging_mem;
    void* staging_ptr;
    VkMemoryPropertyFlags staging_mem_props;  // Track memory properties for flush/invalidate
} Map;

// Compute pipeline
typedef struct {
    VkShaderModule shader;
    VkDescriptorSetLayout desc_layout;
    VkPipelineLayout pipe_layout;
    VkPipeline pipeline;
    VkDescriptorPool desc_pool;
    VkDescriptorSet desc_set;
} Pipeline;

typedef struct {
    VkDeviceSize offset;
    VkDeviceSize size;
} FreeSeg;

typedef struct {
    uint32_t active;
    uint32_t type;
    uint32_t size_bytes;
    VkDeviceSize alloc_size;
    VkDeviceSize hot_offset;
    VkDeviceSize cold_offset;
    uint64_t last_used;
    uint8_t hot_valid;
    uint8_t dirty;
    uint8_t pinned;
} ResEntry;

// Push constants
typedef struct { uint32_t op; uint32_t n; } PushOp;
typedef struct { uint32_t op; uint32_t n; float scalar; } PushOpS;

// Global context
static struct {
    int initialized;
    VkInstance instance;
    VkPhysicalDevice phys;
    VkDevice device;
    VkQueue queue;
    uint32_t queue_family;
    VkCommandPool cmd_pool;
    VkCommandBuffer cmd;
    VkFence fence;
    VkCommandBuffer cmd_ring[CMD_RING];
    VkFence fence_ring[CMD_RING];
    uint64_t submit_id_ring[CMD_RING];
    int in_flight_ring[CMD_RING];
    uint32_t cmd_ring_next;
    uint64_t submit_counter;
    uint64_t last_completed;
    uint32_t devbuf_counter;
    uint32_t num_buffer_recreates;
    uint32_t stage_upload_grow_events;
    uint32_t stage_download_grow_events;
    int pending_desc_reset;
    VkDeviceSize copy_align;
    VkDeviceSize storage_align;

    Map maps[MAX_MAPS];
    GpuBuf bufs[MAX_BUFS];
    int buf_count;

    GpuBuf* hot_pool;
    GpuBuf* cold_pool;
    VkDeviceSize hot_pool_bytes;
    VkDeviceSize cold_pool_bytes;
    VkDeviceSize cold_alloc;
    FreeSeg hot_free[MAX_FREE];
    uint32_t hot_free_count;
    ResEntry res[MAX_RES];
    uint32_t res_count;
    uint64_t res_tick;
    
    Pipeline unary_pipe;
    Pipeline binary_pipe;
    Pipeline matmul_pipe;
    Pipeline reduce_pipe;
    Pipeline reduce_small_pipe;
    Pipeline broadcast_pipe;
    Pipeline softmax_pipe;
    Pipeline layernorm_pipe;
    Pipeline scatter_pipe;
    Pipeline gather_pipe;
    
    char shader_path[512];
} ctx = {0};

// ============================================
// Vulkan Helpers
// ============================================

static uint32_t find_mem_type(uint32_t bits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(ctx.phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    return UINT32_MAX;
}

static uint32_t find_device_local(uint32_t bits) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(ctx.phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if (!(bits & (1u << i))) continue;
        VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && !(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
            return i;
    }
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            return i;
    }
    return UINT32_MAX;
}

// Find optimal host-visible memory for staging buffers
// Universal Vulkan: HOST_CACHED (with flush) > HOST_COHERENT (fallback)
static uint32_t find_host_staging(uint32_t bits, VkMemoryPropertyFlags* props_out) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(ctx.phys, &mp);
    static int staging_logged = 0;

    // Try: HOST_VISIBLE + HOST_CACHED (fast memcpy, needs explicit flush/invalidate)
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if (!(bits & (1u << i))) continue;
        VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) {
            *props_out = f;
            if (!staging_logged) {
                fprintf(stderr, "[ADAMAH] Staging: HOST_CACHED (fast memcpy with explicit flush)\n");
                staging_logged = 1;
            }
            return i;
        }
    }

    // Fallback: HOST_VISIBLE + HOST_COHERENT (slower but no flush needed)
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if (!(bits & (1u << i))) continue;
        VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            *props_out = f;
            if (!staging_logged) {
                fprintf(stderr, "[ADAMAH] Staging: HOST_COHERENT (fallback, slower memcpy)\n");
                staging_logged = 1;
            }
            return i;
        }
    }

    return UINT32_MAX;
}

// Extended version that returns memory properties
static int create_buffer_ex(VkBuffer* buf, VkDeviceMemory* mem, VkDeviceSize size,
                            VkBufferUsageFlags usage, int device_local, VkMemoryPropertyFlags* props_out) {
    VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size, .usage = usage };
    if (vkCreateBuffer(ctx.device, &bci, NULL, buf) != VK_SUCCESS) return -1;

    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(ctx.device, *buf, &reqs);

    VkMemoryPropertyFlags props = 0;
    uint32_t mem_type;

    if (device_local) {
        mem_type = find_device_local(reqs.memoryTypeBits);
    } else {
        // For staging buffers, use optimized memory
        mem_type = find_host_staging(reqs.memoryTypeBits, &props);
    }

    if (mem_type == UINT32_MAX) {
        vkDestroyBuffer(ctx.device, *buf, NULL);
        return -1;
    }

    if (props_out) *props_out = props;

    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = reqs.size, .memoryTypeIndex = mem_type };
    if (vkAllocateMemory(ctx.device, &mai, NULL, mem) != VK_SUCCESS) {
        vkDestroyBuffer(ctx.device, *buf, NULL);
        return -1;
    }
    vkBindBufferMemory(ctx.device, *buf, *mem, 0);
    return 0;
}

static int create_buffer(VkBuffer* buf, VkDeviceMemory* mem, VkDeviceSize size,
                         VkBufferUsageFlags usage, int device_local) {
    return create_buffer_ex(buf, mem, size, usage, device_local, NULL);
}

// ============================================
// True Vulkan Batching - accumulate in single command buffer
// ============================================
static int batch_mode = 0;
static int cmd_recording = 0;  // Is command buffer currently recording?
static int batch_op_counter = 0;  // Counter for unique buffer names in batch
static int cmd_recording_async = 0;
static int cmd_async_slot = -1;

#define HANDLE_FROM_BUF(b) ((uint32_t)((b) - ctx.bufs) + 1u)
#define BUF_FROM_HANDLE(h) (((h) == 0 || (h) > (uint32_t)ctx.buf_count) ? NULL : &ctx.bufs[(h) - 1u])

static void reset_pipeline_desc_pool(Pipeline* p);
static int async_all_done(void);

static VkDescriptorSet alloc_desc_set(Pipeline* p) {
    VkDescriptorSet ds = p->desc_set;
    if (batch_mode) {
        VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = p->desc_pool, .descriptorSetCount = 1, .pSetLayouts = &p->desc_layout };
        if (vkAllocateDescriptorSets(ctx.device, &dsai, &ds) != VK_SUCCESS) return VK_NULL_HANDLE;
    }
    return ds;
}

static VkDescriptorSet alloc_desc_set_async(Pipeline* p) {
    VkDescriptorSet ds = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = p->desc_pool, .descriptorSetCount = 1, .pSetLayouts = &p->desc_layout };
    if (vkAllocateDescriptorSets(ctx.device, &dsai, &ds) != VK_SUCCESS) {
        if (async_all_done()) {
            reset_pipeline_desc_pool(p);
            if (vkAllocateDescriptorSets(ctx.device, &dsai, &ds) != VK_SUCCESS) return VK_NULL_HANDLE;
        } else {
            return VK_NULL_HANDLE;
        }
    }
    ctx.pending_desc_reset = 1;
    return ds;
}

static void cmd_barrier_after_dispatch(void) {
    if (!batch_mode) return;
    VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
    vkCmdPipelineBarrier(ctx.cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &mb, 0, NULL, 0, NULL);
}

static void cmd_buffer_barrier(VkPipelineStageFlags src_stage, VkAccessFlags src_access,
                               VkPipelineStageFlags dst_stage, VkAccessFlags dst_access,
                               VkBuffer* bufs, uint32_t count) {
    if (count == 0) return;
    VkBufferMemoryBarrier barriers[4];
    if (count > 4) count = 4;
    for (uint32_t i = 0; i < count; i++) {
        barriers[i] = (VkBufferMemoryBarrier){ .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = src_access, .dstAccessMask = dst_access,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = bufs[i], .offset = 0, .size = VK_WHOLE_SIZE };
    }
    vkCmdPipelineBarrier(ctx.cmd, src_stage, dst_stage, 0, 0, NULL, count, barriers, 0, NULL);
}

static void cmd_begin(void);
static void cmd_submit(void);

static VkDeviceSize align_up(VkDeviceSize v, VkDeviceSize a) {
    if (a == 0) return v;
    return (v + a - 1) & ~(a - 1);
}

static VkDeviceSize align_storage(VkDeviceSize v) {
    return align_up(v, ctx.storage_align);
}

static void hot_free_insert(VkDeviceSize offset, VkDeviceSize size) {
    if (size == 0 || ctx.hot_free_count >= MAX_FREE) return;
    uint32_t idx = 0;
    while (idx < ctx.hot_free_count && ctx.hot_free[idx].offset < offset) idx++;
    for (uint32_t i = ctx.hot_free_count; i > idx; i--) {
        ctx.hot_free[i] = ctx.hot_free[i - 1];
    }
    ctx.hot_free[idx].offset = offset;
    ctx.hot_free[idx].size = size;
    ctx.hot_free_count++;

    // Coalesce with previous
    if (idx > 0) {
        FreeSeg* prev = &ctx.hot_free[idx - 1];
        FreeSeg* cur = &ctx.hot_free[idx];
        if (prev->offset + prev->size == cur->offset) {
            prev->size += cur->size;
            for (uint32_t i = idx; i + 1 < ctx.hot_free_count; i++) {
                ctx.hot_free[i] = ctx.hot_free[i + 1];
            }
            ctx.hot_free_count--;
            idx--;
        }
    }

    // Coalesce with next
    if (idx + 1 < ctx.hot_free_count) {
        FreeSeg* cur = &ctx.hot_free[idx];
        FreeSeg* next = &ctx.hot_free[idx + 1];
        if (cur->offset + cur->size == next->offset) {
            cur->size += next->size;
            for (uint32_t i = idx + 1; i + 1 < ctx.hot_free_count; i++) {
                ctx.hot_free[i] = ctx.hot_free[i + 1];
            }
            ctx.hot_free_count--;
        }
    }
}

static int hot_alloc(VkDeviceSize size, VkDeviceSize* out_offset) {
    size = align_storage(size);
    for (uint32_t i = 0; i < ctx.hot_free_count; i++) {
        FreeSeg* seg = &ctx.hot_free[i];
        if (seg->size >= size) {
            *out_offset = seg->offset;
            seg->offset += size;
            seg->size -= size;
            if (seg->size == 0) {
                for (uint32_t j = i; j + 1 < ctx.hot_free_count; j++) {
                    ctx.hot_free[j] = ctx.hot_free[j + 1];
                }
                ctx.hot_free_count--;
            }
            return 0;
        }
    }
    return -1;
}

static ResEntry* res_get(uint32_t id) {
    if (id == 0 || id > MAX_RES) return NULL;
    ResEntry* r = &ctx.res[id - 1];
    if (!r->active) return NULL;
    return r;
}

static void res_pin(uint32_t id) {
    ResEntry* r = res_get(id);
    if (!r) return;
    if (r->pinned < 255) r->pinned++;
}

static void res_unpin(uint32_t id) {
    ResEntry* r = res_get(id);
    if (!r) return;
    if (r->pinned > 0) r->pinned--;
}

static void res_copy_buffers(VkBuffer src, VkDeviceSize src_off, VkBuffer dst, VkDeviceSize dst_off,
                             VkDeviceSize size, int src_hot, int dst_hot) {
    VkDeviceSize copy_size = align_up(size, ctx.copy_align);
    if (copy_size == 0) return;
    cmd_begin();
    if (src_hot) {
        VkBuffer bufs[1] = { src };
        cmd_buffer_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                           bufs, 1);
    }
    VkBufferCopy copy = { .srcOffset = src_off, .dstOffset = dst_off, .size = copy_size };
    vkCmdCopyBuffer(ctx.cmd, src, dst, 1, &copy);
    if (dst_hot) {
        VkBuffer bufs[1] = { dst };
        cmd_buffer_barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                           bufs, 1);
    }
    cmd_submit();
}

static int res_evict_one(void) {
    int victim = -1;
    uint64_t best = UINT64_MAX;
    for (uint32_t i = 0; i < MAX_RES; i++) {
        ResEntry* r = &ctx.res[i];
        if (!r->active || !r->hot_valid || r->pinned) continue;
        if (r->last_used < best) {
            best = r->last_used;
            victim = (int)i;
        }
    }
    if (victim < 0) return -1;
    ResEntry* r = &ctx.res[victim];
    if (r->dirty) {
        res_copy_buffers(ctx.hot_pool->buf, r->hot_offset, ctx.cold_pool->buf, r->cold_offset, r->alloc_size, 1, 0);
        r->dirty = 0;
    }
    hot_free_insert(r->hot_offset, r->alloc_size);
    r->hot_valid = 0;
    return 0;
}

static int res_require_hot(uint32_t id, VkDeviceSize* out_offset) {
    ResEntry* r = res_get(id);
    if (!r) return -1;
    if (r->hot_valid) {
        r->last_used = ++ctx.res_tick;
        *out_offset = r->hot_offset;
        return 0;
    }

    while (hot_alloc(r->alloc_size, &r->hot_offset) != 0) {
        if (res_evict_one() != 0) return -1;
    }

    r->hot_valid = 1;
    r->last_used = ++ctx.res_tick;

    if (!r->dirty) {
        res_copy_buffers(ctx.cold_pool->buf, r->cold_offset, ctx.hot_pool->buf, r->hot_offset, r->alloc_size, 0, 1);
    }
    *out_offset = r->hot_offset;
    return 0;
}

static int res_alloc(uint32_t type, uint32_t size_bytes, uint32_t* out_id) {
    if (size_bytes == 0) return -1;
    uint32_t id = 0;
    for (uint32_t i = 0; i < MAX_RES; i++) {
        if (!ctx.res[i].active) { id = i + 1; break; }
    }
    if (id == 0) return -1;
    ResEntry* r = &ctx.res[id - 1];
    memset(r, 0, sizeof(*r));
    r->active = 1;
    r->type = type;
    r->size_bytes = size_bytes;
    r->alloc_size = align_storage((VkDeviceSize)size_bytes);
    if (ctx.cold_alloc + r->alloc_size > ctx.cold_pool_bytes) return -1;
    r->cold_offset = ctx.cold_alloc;
    ctx.cold_alloc += r->alloc_size;
    r->hot_valid = 0;
    r->dirty = 0;
    r->pinned = 0;
    r->last_used = ++ctx.res_tick;
    ctx.res_count++;
    *out_id = id;
    return 0;
}

static int cache_init(VkDeviceSize hot_bytes, VkDeviceSize cold_bytes) {
    if (ctx.hot_pool && ctx.cold_pool) return 0;
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (hot_bytes == 0) hot_bytes = HOT_POOL_BYTES_DEFAULT;
    if (cold_bytes == 0) cold_bytes = COLD_POOL_BYTES_DEFAULT;
    if (hot_bytes > UINT32_MAX) hot_bytes = UINT32_MAX;
    if (cold_bytes > UINT32_MAX) cold_bytes = UINT32_MAX;

    ctx.hot_pool = get_or_create_buf_ex("_cache_hot", (uint32_t)hot_bytes, 1, 1, usage);
    ctx.cold_pool = get_or_create_buf_ex("_cache_cold", (uint32_t)cold_bytes, 1, 1, usage);
    if (!ctx.hot_pool || !ctx.cold_pool) return -1;

    ctx.hot_pool_bytes = ctx.hot_pool->bytes_capacity;
    ctx.cold_pool_bytes = ctx.cold_pool->bytes_capacity;
    ctx.cold_alloc = 0;
    ctx.hot_free_count = 0;
    hot_free_insert(0, ctx.hot_pool_bytes);
    memset(ctx.res, 0, sizeof(ctx.res));
    ctx.res_count = 0;
    ctx.res_tick = 0;
    return 0;
}

static int debug_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char* env = getenv("ADAMAH_DEBUG");
        cached = (env && env[0] && strcmp(env, "0") != 0) ? 1 : 0;
    }
    return cached;
}

static void debug_path(const char* op, const char* path) {
    if (debug_enabled()) {
        fprintf(stderr, "ADAMAH DEBUG: %s path: %s\n", op, path);
    }
}

static VkDeviceSize next_pow2(VkDeviceSize v) {
    if (v <= 1) return 1;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
#if UINTPTR_MAX > 0xFFFFFFFFu
    v |= v >> 32;
#endif
    return v + 1;
}

static VkDeviceSize device_local_bucket(VkDeviceSize v) {
    VkDeviceSize size = 64 * 1024;
    while (size < v) {
        size *= 4;
        if (size == 0) break;
    }
    return size;
}

static int is_stage_upload_name(const char* name) {
    return strncmp(name, "_stage_upload", 13) == 0;
}

static int is_stage_download_name(const char* name) {
    return strncmp(name, "_stage_download", 15) == 0;
}

static void reset_pipeline_desc_pool(Pipeline* p) {
    if (!p->desc_pool) return;
    vkResetDescriptorPool(ctx.device, p->desc_pool, 0);
    VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = p->desc_pool, .descriptorSetCount = 1, .pSetLayouts = &p->desc_layout };
    if (vkAllocateDescriptorSets(ctx.device, &dsai, &p->desc_set) != VK_SUCCESS) {
        p->desc_set = VK_NULL_HANDLE;
    }
}

static void update_async_completed(void) {
    for (int i = 0; i < CMD_RING; i++) {
        if (!ctx.in_flight_ring[i]) continue;
        VkResult st = vkGetFenceStatus(ctx.device, ctx.fence_ring[i]);
        if (st == VK_SUCCESS) {
            ctx.in_flight_ring[i] = 0;
            if (ctx.submit_id_ring[i] > ctx.last_completed) {
                ctx.last_completed = ctx.submit_id_ring[i];
            }
        }
    }
}

static int async_all_done(void) {
    update_async_completed();
    return ctx.last_completed >= ctx.submit_counter;
}

static int cmd_begin_async(void) {
    if (cmd_recording_async) return 0;
    if (batch_mode) return -1;

    update_async_completed();

    int slot = -1;
    for (int i = 0; i < CMD_RING; i++) {
        int idx = (ctx.cmd_ring_next + i) % CMD_RING;
        if (!ctx.in_flight_ring[idx]) { slot = idx; break; }
    }

    if (slot < 0) {
        slot = ctx.cmd_ring_next % CMD_RING;
        vkWaitForFences(ctx.device, 1, &ctx.fence_ring[slot], VK_TRUE, UINT64_MAX);
        ctx.in_flight_ring[slot] = 0;
        if (ctx.submit_id_ring[slot] > ctx.last_completed) {
            ctx.last_completed = ctx.submit_id_ring[slot];
        }
    }

    ctx.cmd_ring_next = (uint32_t)((slot + 1) % CMD_RING);
    vkResetFences(ctx.device, 1, &ctx.fence_ring[slot]);
    vkResetCommandBuffer(ctx.cmd_ring[slot], 0);

    ctx.cmd = ctx.cmd_ring[slot];
    cmd_async_slot = slot;

    VkCommandBufferBeginInfo cbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    vkBeginCommandBuffer(ctx.cmd, &cbi);
    cmd_recording_async = 1;
    return 0;
}

static uint64_t cmd_submit_async(void) {
    if (!cmd_recording_async) return 0;
    vkEndCommandBuffer(ctx.cmd);
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &ctx.cmd };
    vkQueueSubmit(ctx.queue, 1, &si, ctx.fence_ring[cmd_async_slot]);
    uint64_t ticket = ++ctx.submit_counter;
    ctx.submit_id_ring[cmd_async_slot] = ticket;
    ctx.in_flight_ring[cmd_async_slot] = 1;
    cmd_recording_async = 0;
    return ticket;
}

static void cmd_abort_async(void) {
    if (!cmd_recording_async) return;
    vkEndCommandBuffer(ctx.cmd);
    cmd_recording_async = 0;
}

static void cmd_begin(void) {
    if (batch_mode && cmd_recording) {
        // Already recording in batch mode, just continue adding commands
        return;
    }
    
    // Wait for previous work to complete
    vkWaitForFences(ctx.device, 1, &ctx.fence, VK_TRUE, UINT64_MAX);
    vkResetFences(ctx.device, 1, &ctx.fence);
    vkResetCommandBuffer(ctx.cmd, 0);
    
    VkCommandBufferBeginInfo cbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    vkBeginCommandBuffer(ctx.cmd, &cbi);
    cmd_recording = 1;
}

static void cmd_submit(void) {
    if (batch_mode) {
        // In batch mode, don't submit yet - keep accumulating
        // Increment counter for next op to use different buffers
        batch_op_counter++;
        return;
    }
    
    if (!cmd_recording) return;
    
    vkEndCommandBuffer(ctx.cmd);
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &ctx.cmd };
    vkQueueSubmit(ctx.queue, 1, &si, ctx.fence);
    vkWaitForFences(ctx.device, 1, &ctx.fence, VK_TRUE, UINT64_MAX);
    cmd_recording = 0;
}

// Start batch mode - commands accumulated into single command buffer
void batch_begin(void) {
    // Ensure clean state
    if (cmd_recording) {
        vkEndCommandBuffer(ctx.cmd);
        VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1, .pCommandBuffers = &ctx.cmd };
        vkQueueSubmit(ctx.queue, 1, &si, ctx.fence);
        vkWaitForFences(ctx.device, 1, &ctx.fence, VK_TRUE, UINT64_MAX);
        cmd_recording = 0;
    }
    batch_mode = 1;
    batch_op_counter = 0;  // Reset counter for new batch
}

// End batch mode - submit all accumulated commands
void batch_end(void) {
    if (!batch_mode) return;
    
    batch_mode = 0;
    
    if (cmd_recording) {
        vkEndCommandBuffer(ctx.cmd);
        VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1, .pCommandBuffers = &ctx.cmd };
        vkQueueSubmit(ctx.queue, 1, &si, ctx.fence);
        vkWaitForFences(ctx.device, 1, &ctx.fence, VK_TRUE, UINT64_MAX);
        cmd_recording = 0;
    }

    // Batch completed, recycle descriptor sets for next batch
    reset_pipeline_desc_pool(&ctx.unary_pipe);
    reset_pipeline_desc_pool(&ctx.binary_pipe);
    reset_pipeline_desc_pool(&ctx.matmul_pipe);
    reset_pipeline_desc_pool(&ctx.reduce_pipe);
    reset_pipeline_desc_pool(&ctx.reduce_small_pipe);
    reset_pipeline_desc_pool(&ctx.broadcast_pipe);
    reset_pipeline_desc_pool(&ctx.softmax_pipe);
    reset_pipeline_desc_pool(&ctx.layernorm_pipe);
    
    batch_op_counter = 0;  // Reset counter
}

// Sync: wait for all queued GPU work to finish
void adamah_sync(void) {
    if (!ctx.initialized) return;
    // Ensure async work completes and descriptor pools can be recycled
    for (int i = 0; i < CMD_RING; i++) {
        if (ctx.in_flight_ring[i]) {
            vkWaitForFences(ctx.device, 1, &ctx.fence_ring[i], VK_TRUE, UINT64_MAX);
            ctx.in_flight_ring[i] = 0;
            if (ctx.submit_id_ring[i] > ctx.last_completed) {
                ctx.last_completed = ctx.submit_id_ring[i];
            }
        }
    }
    if (ctx.pending_desc_reset) {
        reset_pipeline_desc_pool(&ctx.unary_pipe);
        reset_pipeline_desc_pool(&ctx.binary_pipe);
        reset_pipeline_desc_pool(&ctx.matmul_pipe);
        reset_pipeline_desc_pool(&ctx.reduce_pipe);
        reset_pipeline_desc_pool(&ctx.reduce_small_pipe);
        reset_pipeline_desc_pool(&ctx.broadcast_pipe);
        reset_pipeline_desc_pool(&ctx.softmax_pipe);
        reset_pipeline_desc_pool(&ctx.layernorm_pipe);
            reset_pipeline_desc_pool(&ctx.scatter_pipe);
        reset_pipeline_desc_pool(&ctx.gather_pipe);
        ctx.pending_desc_reset = 0;
    }
    vkDeviceWaitIdle(ctx.device);
}

void adamah_synchronize(uint64_t ticket) {
    if (!ctx.initialized) return;
    if (ticket == 0 || ticket <= ctx.last_completed) return;
    update_async_completed();
    if (ticket <= ctx.last_completed) return;
    for (int i = 0; i < CMD_RING; i++) {
        if (ctx.in_flight_ring[i] && ctx.submit_id_ring[i] <= ticket) {
            vkWaitForFences(ctx.device, 1, &ctx.fence_ring[i], VK_TRUE, UINT64_MAX);
            ctx.in_flight_ring[i] = 0;
            if (ctx.submit_id_ring[i] > ctx.last_completed) {
                ctx.last_completed = ctx.submit_id_ring[i];
            }
        }
    }
    if (ctx.pending_desc_reset && async_all_done()) {
        reset_pipeline_desc_pool(&ctx.unary_pipe);
        reset_pipeline_desc_pool(&ctx.binary_pipe);
        reset_pipeline_desc_pool(&ctx.matmul_pipe);
        reset_pipeline_desc_pool(&ctx.reduce_pipe);
        reset_pipeline_desc_pool(&ctx.reduce_small_pipe);
        reset_pipeline_desc_pool(&ctx.broadcast_pipe);
        reset_pipeline_desc_pool(&ctx.softmax_pipe);
        reset_pipeline_desc_pool(&ctx.layernorm_pipe);
            reset_pipeline_desc_pool(&ctx.scatter_pipe);
        reset_pipeline_desc_pool(&ctx.gather_pipe);
        ctx.pending_desc_reset = 0;
    }
}

void adamah_synchronize_all(void) {
    if (!ctx.initialized) return;
    for (int i = 0; i < CMD_RING; i++) {
        if (ctx.in_flight_ring[i]) {
            vkWaitForFences(ctx.device, 1, &ctx.fence_ring[i], VK_TRUE, UINT64_MAX);
            ctx.in_flight_ring[i] = 0;
        }
    }
    ctx.last_completed = ctx.submit_counter;
    if (ctx.pending_desc_reset) {
        reset_pipeline_desc_pool(&ctx.unary_pipe);
        reset_pipeline_desc_pool(&ctx.binary_pipe);
        reset_pipeline_desc_pool(&ctx.matmul_pipe);
        reset_pipeline_desc_pool(&ctx.reduce_pipe);
        reset_pipeline_desc_pool(&ctx.reduce_small_pipe);
        reset_pipeline_desc_pool(&ctx.broadcast_pipe);
        reset_pipeline_desc_pool(&ctx.softmax_pipe);
        reset_pipeline_desc_pool(&ctx.layernorm_pipe);
            reset_pipeline_desc_pool(&ctx.scatter_pipe);
        reset_pipeline_desc_pool(&ctx.gather_pipe);
        ctx.pending_desc_reset = 0;
    }
}

void adamah_print_counters(void) {
    if (!debug_enabled()) return;
    fprintf(stderr, "ADAMAH DEBUG: buffer_recreates=%u stage_upload_grows=%u stage_download_grows=%u\n",
            ctx.num_buffer_recreates, ctx.stage_upload_grow_events, ctx.stage_download_grow_events);
}

// ============================================
// Init / Shutdown
// ============================================

int adamah_init_ex(uint64_t hot_bytes, uint64_t cold_bytes) {
    if (ctx.initialized) return ADAMAH_OK;
    
    // Create instance
    VkApplicationInfo ai = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "ADAMAH", .apiVersion = VK_API_VERSION_1_0 };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &ai };
    if (vkCreateInstance(&ici, NULL, &ctx.instance) != VK_SUCCESS) return ADAMAH_ERR_VULKAN;
    
    // Get physical device - prefer discrete GPU over integrated/software
    uint32_t dc = 0;
    vkEnumeratePhysicalDevices(ctx.instance, &dc, NULL);
    if (!dc) return ADAMAH_ERR_VULKAN;
    VkPhysicalDevice devs[8];
    vkEnumeratePhysicalDevices(ctx.instance, &dc, devs);

    // Debug: print all available devices
    fprintf(stderr, "[ADAMAH] Found %u Vulkan device(s):\n", dc);
    for (uint32_t i = 0; i < dc && i < 8; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        const char* type_str = "OTHER";
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) type_str = "DISCRETE_GPU";
        else if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) type_str = "INTEGRATED_GPU";
        else if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) type_str = "CPU";
        fprintf(stderr, "[ADAMAH]   [%u] %s (%s)\n", i, p.deviceName, type_str);
    }

    // Selection priority: DISCRETE_GPU > INTEGRATED_GPU > OTHER (skip CPU renderer)
    ctx.phys = devs[0];  // Fallback
    int best_score = -1;

    for (uint32_t i = 0; i < dc && i < 8; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devs[i], &props);

        int score = 0;
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score = 100;
        else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score = 50;
        else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) score = -100;  // Never use CPU renderer
        else score = 10;

        if (score > best_score) {
            ctx.phys = devs[i];
            best_score = score;
        }
    }

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(ctx.phys, &props);
    fprintf(stderr, "[ADAMAH] Selected: %s\n", props.deviceName);
    printf("ADAMAH v5: %s\n", props.deviceName);
    ctx.copy_align = props.limits.optimalBufferCopyOffsetAlignment;
    if (ctx.copy_align < 4) ctx.copy_align = 4;
    ctx.storage_align = props.limits.minStorageBufferOffsetAlignment;
    if (ctx.storage_align < ctx.copy_align) ctx.storage_align = ctx.copy_align;
    if (ctx.storage_align < 4) ctx.storage_align = 4;
    
    // Find compute queue
    uint32_t qfc = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.phys, &qfc, NULL);
    VkQueueFamilyProperties qfp[8];
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.phys, &qfc, qfp);
    for (uint32_t i = 0; i < qfc; i++) {
        if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { ctx.queue_family = i; break; }
    }
    
    // Create device
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = ctx.queue_family, .queueCount = 1, .pQueuePriorities = &prio };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci };
    if (vkCreateDevice(ctx.phys, &dci, NULL, &ctx.device) != VK_SUCCESS) return ADAMAH_ERR_VULKAN;
    vkGetDeviceQueue(ctx.device, ctx.queue_family, 0, &ctx.queue);
    
    // Command pool & buffer
    VkCommandPoolCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = ctx.queue_family };
    vkCreateCommandPool(ctx.device, &cpi, NULL, &ctx.cmd_pool);
    
    VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx.cmd_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
    vkAllocateCommandBuffers(ctx.device, &cai, &ctx.cmd);

    VkCommandBufferAllocateInfo cai_ring = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx.cmd_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = CMD_RING };
    vkAllocateCommandBuffers(ctx.device, &cai_ring, ctx.cmd_ring);
    
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT };
    vkCreateFence(ctx.device, &fci, NULL, &ctx.fence);
    for (int i = 0; i < CMD_RING; i++) {
        vkCreateFence(ctx.device, &fci, NULL, &ctx.fence_ring[i]);
        ctx.in_flight_ring[i] = 0;
        ctx.submit_id_ring[i] = 0;
    }
    ctx.cmd_ring_next = 0;
    ctx.submit_counter = 0;
    ctx.last_completed = 0;
    ctx.devbuf_counter = 0;
    ctx.pending_desc_reset = 0;
    
    if (cache_init(hot_bytes, cold_bytes) != 0) {
        return ADAMAH_ERR_MEMORY;
    }

    ctx.initialized = 1;
    return ADAMAH_OK;
}

int adamah_init(void) {
    return adamah_init_ex(HOT_POOL_BYTES_DEFAULT, COLD_POOL_BYTES_DEFAULT);
}

void adamah_shutdown(void) {
    if (!ctx.initialized) return;
    vkDeviceWaitIdle(ctx.device);
    
    // Destroy maps
    for (int i = 0; i < MAX_MAPS; i++) {
        if (ctx.maps[i].active) map_destroy(i);
    }
    
    // Destroy buffers
    for (int i = 0; i < ctx.buf_count; i++) {
        if (ctx.bufs[i].ptr) vkUnmapMemory(ctx.device, ctx.bufs[i].mem);
        vkDestroyBuffer(ctx.device, ctx.bufs[i].buf, NULL);
        vkFreeMemory(ctx.device, ctx.bufs[i].mem, NULL);
    }
    
    if (ctx.fence) vkDestroyFence(ctx.device, ctx.fence, NULL);
    for (int i = 0; i < CMD_RING; i++) {
        if (ctx.fence_ring[i]) vkDestroyFence(ctx.device, ctx.fence_ring[i], NULL);
    }
    if (ctx.cmd_pool) vkDestroyCommandPool(ctx.device, ctx.cmd_pool, NULL);
    if (ctx.device) vkDestroyDevice(ctx.device, NULL);
    if (ctx.instance) vkDestroyInstance(ctx.instance, NULL);
    
    memset(&ctx, 0, sizeof(ctx));
}

// ============================================
// Memory Maps
// ============================================

int map_init(uint32_t id, uint32_t word_size, uint32_t pack_size, uint32_t n_packs) {
    if (!ctx.initialized || id >= MAX_MAPS) return ADAMAH_ERR_INVALID;
    if (ctx.maps[id].active) map_destroy(id);
    
    Map* m = &ctx.maps[id];
    m->word_size = word_size;
    m->pack_size = pack_size;
    m->n_packs = n_packs;
    m->total_bytes = (uint64_t)word_size * pack_size * n_packs;
    
    // GPU buffer (DEVICE_LOCAL)
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (create_buffer(&m->buf, &m->mem, m->total_bytes, usage, 1) != 0)
        return ADAMAH_ERR_MEMORY;

    // Staging buffer (HOST_VISIBLE) - use optimized memory
    if (create_buffer_ex(&m->staging, &m->staging_mem, m->total_bytes, usage, 0, &m->staging_mem_props) != 0) {
        vkDestroyBuffer(ctx.device, m->buf, NULL);
        vkFreeMemory(ctx.device, m->mem, NULL);
        return ADAMAH_ERR_MEMORY;
    }
    vkMapMemory(ctx.device, m->staging_mem, 0, m->total_bytes, 0, &m->staging_ptr);
    
    // Clear
    memset(m->staging_ptr, 0, m->total_bytes);
    cmd_begin();
    VkBufferCopy copy = { .size = m->total_bytes };
    vkCmdCopyBuffer(ctx.cmd, m->staging, m->buf, 1, &copy);
    cmd_submit();
    
    m->active = 1;
    return ADAMAH_OK;
}

int map_destroy(uint32_t id) {
    if (id >= MAX_MAPS || !ctx.maps[id].active) return ADAMAH_ERR_INVALID;
    Map* m = &ctx.maps[id];
    
    vkUnmapMemory(ctx.device, m->staging_mem);
    vkDestroyBuffer(ctx.device, m->staging, NULL);
    vkFreeMemory(ctx.device, m->staging_mem, NULL);
    vkDestroyBuffer(ctx.device, m->buf, NULL);
    vkFreeMemory(ctx.device, m->mem, NULL);
    
    memset(m, 0, sizeof(Map));
    return ADAMAH_OK;
}

uint64_t map_size(uint32_t id) {
    if (id >= MAX_MAPS || !ctx.maps[id].active) return 0;
    return ctx.maps[id].n_packs;
}

// ============================================
// Scatter / Gather (CPU <-> Map)
// ============================================

static int locs_contiguous_in_range(const uint32_t* locs, uint32_t n_locs, uint32_t n_packs, uint32_t* start_out) {
    if (n_locs == 0) return 0;
    uint32_t start = locs[0];
    if (start >= n_packs) return 0;
    if ((uint64_t)start + (uint64_t)n_locs > (uint64_t)n_packs) return 0;
    for (uint32_t i = 1; i < n_locs; i++) {
        if (locs[i] != start + i) return 0;
    }
    if (start_out) *start_out = start;
    return 1;
}

// Forward declarations
static GpuBuf* get_or_create_buf_ex(const char* base_name, uint32_t n_elems, uint32_t elem_size,
                                    int device_local, VkBufferUsageFlags usage);
static GpuBuf* get_or_create_buf(const char* base_name, uint32_t n_elems, uint32_t elem_size);
static int init_pipelines(void);

// Scatter: write data to map at locations
// locs: array of pack indices (uint32)
// data: packed data (n_locs * pack_size * word_size bytes)
int map_scatter(uint32_t map_id, const uint32_t* locs, const void* data, uint32_t n_locs) {
    if (map_id >= MAX_MAPS || !ctx.maps[map_id].active) return ADAMAH_ERR_INVALID;
    Map* m = &ctx.maps[map_id];
    uint32_t pack_bytes = m->word_size * m->pack_size;
    const uint32_t gpu_threshold = 64;
    
    if (n_locs == 0) return ADAMAH_OK;

    // Fast path: contiguous locs - direct DMA
    uint32_t start = 0;
    if (locs_contiguous_in_range(locs, n_locs, m->n_packs, &start)) {
        debug_path("scatter", "contiguous");
        size_t off = (size_t)start * pack_bytes;
        size_t size = (size_t)n_locs * pack_bytes;

#ifdef ADAMAH_PROFILE
        struct timespec t0, t1, t2, t3;
        clock_gettime(CLOCK_MONOTONIC, &t0);
#endif

        memcpy((char*)m->staging_ptr + off, data, size);

        // Flush if using cached (non-coherent) memory
        if (!(m->staging_mem_props & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            VkMappedMemoryRange range = {
                .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                .memory = m->staging_mem,
                .offset = off,
                .size = size
            };
            vkFlushMappedMemoryRanges(ctx.device, 1, &range);
        }

#ifdef ADAMAH_PROFILE
        clock_gettime(CLOCK_MONOTONIC, &t1);
#endif

        cmd_begin();
        VkBufferCopy copy = { .srcOffset = off, .dstOffset = off, .size = size };
        vkCmdCopyBuffer(ctx.cmd, m->staging, m->buf, 1, &copy);

#ifdef ADAMAH_PROFILE
        clock_gettime(CLOCK_MONOTONIC, &t2);
#endif

        cmd_submit();

#ifdef ADAMAH_PROFILE
        clock_gettime(CLOCK_MONOTONIC, &t3);
        double memcpy_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
        double gpu_cmd_ms = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_nsec - t1.tv_nsec) / 1e6;
        double submit_ms = (t3.tv_sec - t2.tv_sec) * 1000.0 + (t3.tv_nsec - t2.tv_nsec) / 1e6;
        double total_ms = (t3.tv_sec - t0.tv_sec) * 1000.0 + (t3.tv_nsec - t0.tv_nsec) / 1e6;
        double bandwidth_gbs = (size / 1e9) / (total_ms / 1000.0);
        fprintf(stderr, "[SCATTER] size=%zu bytes (%.2f MB), memcpy=%.3fms, gpu_cmd=%.3fms, submit=%.3fms, total=%.3fms, BW=%.2f GB/s\n",
                size, size / 1e6, memcpy_ms, gpu_cmd_ms, submit_ms, total_ms, bandwidth_gbs);
#endif

        return ADAMAH_OK;
    }

    // GPU scatter path: use compute shader for sparse writes
    // This is faster than many small DMA copies for sparse data
    if (!ctx.scatter_pipe.pipeline) init_pipelines();  // Ensure pipeline is ready
    if (ctx.scatter_pipe.pipeline && n_locs >= gpu_threshold) {
        debug_path("scatter", "gpu_sparse");
        VkDeviceSize data_size = (VkDeviceSize)n_locs * pack_bytes;
        VkDeviceSize locs_size = (VkDeviceSize)n_locs * sizeof(uint32_t);
        VkDeviceSize locs_offset = align_up(data_size, ctx.copy_align);
        VkDeviceSize upload_size = locs_offset + locs_size;

        // Device-local buffers
        VkBufferUsageFlags dev_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        GpuBuf* src_buf = get_or_create_buf_ex("_scatter_src", n_locs * m->pack_size, m->word_size, 1, dev_usage);
        GpuBuf* locs_buf = get_or_create_buf_ex("_scatter_locs", n_locs, 4, 1, dev_usage);
        if (!src_buf || !locs_buf) goto fallback;

        // Staging upload buffer
        GpuBuf* stage_up = get_or_create_buf_ex("_stage_upload", (uint32_t)upload_size, 1, 0,
                                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        if (!stage_up || !stage_up->ptr) goto fallback;

        memcpy((char*)stage_up->ptr, data, (size_t)data_size);
        memcpy((char*)stage_up->ptr + (size_t)locs_offset, locs, (size_t)locs_size);

        // Update descriptor set
        VkDescriptorBufferInfo buf_infos[3] = {
            { .buffer = m->buf, .range = VK_WHOLE_SIZE },
            { .buffer = src_buf->buf, .range = VK_WHOLE_SIZE },
            { .buffer = locs_buf->buf, .range = VK_WHOLE_SIZE }
        };
        VkWriteDescriptorSet writes[3];
        for (int i = 0; i < 3; i++) {
            writes[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = ctx.scatter_pipe.desc_set, .dstBinding = i, .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &buf_infos[i] };
        }
        vkUpdateDescriptorSets(ctx.device, 3, writes, 0, NULL);

        uint32_t push[2] = { n_locs, m->pack_size };
        uint32_t total_threads = n_locs * m->pack_size;

        cmd_begin();
        VkBufferCopy copies[2] = {
            { .srcOffset = 0, .dstOffset = 0, .size = data_size },
            { .srcOffset = locs_offset, .dstOffset = 0, .size = locs_size }
        };
        vkCmdCopyBuffer(ctx.cmd, stage_up->buf, src_buf->buf, 1, &copies[0]);
        vkCmdCopyBuffer(ctx.cmd, stage_up->buf, locs_buf->buf, 1, &copies[1]);
        VkBuffer trans_bufs[2] = { src_buf->buf, locs_buf->buf };
        cmd_buffer_barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                           trans_bufs, 2);

        vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.scatter_pipe.pipeline);
        vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.scatter_pipe.pipe_layout,
                                0, 1, &ctx.scatter_pipe.desc_set, 0, NULL);
        vkCmdPushConstants(ctx.cmd, ctx.scatter_pipe.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, push);
        vkCmdDispatch(ctx.cmd, (total_threads + 255) / 256, 1, 1);
        cmd_submit();

        return ADAMAH_OK;
    }

fallback:
    // CPU fallback: for small n_locs or if GPU scatter not available
    debug_path("scatter", "fallback");
    // Heuristic: for very large writes, full buffer copy is cheaper than many regions.
    if ((uint64_t)n_locs * (uint64_t)pack_bytes >= (uint64_t)m->total_bytes / 2) {
        const char* src = (const char*)data;
        char* staging = (char*)m->staging_ptr;
        for (uint32_t i = 0; i < n_locs; i++) {
            uint32_t loc = locs[i];
            if (loc >= m->n_packs) continue;
            memcpy(staging + (size_t)loc * pack_bytes, src + (size_t)i * pack_bytes, pack_bytes);
        }
        cmd_begin();
        VkBufferCopy copy = { .size = m->total_bytes };
        vkCmdCopyBuffer(ctx.cmd, m->staging, m->buf, 1, &copy);
        cmd_submit();
        return ADAMAH_OK;
    }

    // Sparse copy: build per-loc regions.
    const char* src = (const char*)data;
    char* staging = (char*)m->staging_ptr;
    VkBufferCopy* regions = (VkBufferCopy*)malloc(sizeof(VkBufferCopy) * n_locs);
    if (!regions) return ADAMAH_ERR_MEMORY;
    uint32_t rcount = 0;

    for (uint32_t i = 0; i < n_locs; i++) {
        uint32_t loc = locs[i];
        if (loc >= m->n_packs) continue;
        size_t off = (size_t)loc * pack_bytes;
        memcpy(staging + off, src + (size_t)i * pack_bytes, pack_bytes);
        regions[rcount].srcOffset = off;
        regions[rcount].dstOffset = off;
        regions[rcount].size = pack_bytes;
        rcount++;
    }

    if (rcount > 0) {
        cmd_begin();
        vkCmdCopyBuffer(ctx.cmd, m->staging, m->buf, rcount, regions);
        cmd_submit();
    }
    free(regions);
    
    return ADAMAH_OK;
}

// Gather: read data from map at locations
int map_gather(uint32_t map_id, const uint32_t* locs, void* data, uint32_t n_locs) {
    if (map_id >= MAX_MAPS || !ctx.maps[map_id].active) return ADAMAH_ERR_INVALID;
    Map* m = &ctx.maps[map_id];
    uint32_t pack_bytes = m->word_size * m->pack_size;
    const uint32_t gpu_threshold = 64;
    
    if (n_locs == 0) return ADAMAH_OK;

    // Fast path: contiguous locs - direct DMA
    uint32_t start = 0;
    if (locs_contiguous_in_range(locs, n_locs, m->n_packs, &start)) {
        debug_path("gather", "contiguous");
        size_t off = (size_t)start * pack_bytes;
        size_t size = (size_t)n_locs * pack_bytes;

#ifdef ADAMAH_PROFILE
        struct timespec t0, t1, t2, t3;
        clock_gettime(CLOCK_MONOTONIC, &t0);
#endif

        cmd_begin();
        VkBufferCopy copy = { .srcOffset = off, .dstOffset = off, .size = size };
        vkCmdCopyBuffer(ctx.cmd, m->buf, m->staging, 1, &copy);

#ifdef ADAMAH_PROFILE
        clock_gettime(CLOCK_MONOTONIC, &t1);
#endif

        cmd_submit();

#ifdef ADAMAH_PROFILE
        clock_gettime(CLOCK_MONOTONIC, &t2);
#endif

        // Invalidate if using cached (non-coherent) memory
        if (!(m->staging_mem_props & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            VkMappedMemoryRange range = {
                .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                .memory = m->staging_mem,
                .offset = off,
                .size = size
            };
            vkInvalidateMappedMemoryRanges(ctx.device, 1, &range);
        }

        memcpy(data, (char*)m->staging_ptr + off, size);

#ifdef ADAMAH_PROFILE
        clock_gettime(CLOCK_MONOTONIC, &t3);
        double gpu_cmd_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
        double submit_ms = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_nsec - t1.tv_nsec) / 1e6;
        double memcpy_ms = (t3.tv_sec - t2.tv_sec) * 1000.0 + (t3.tv_nsec - t2.tv_nsec) / 1e6;
        double total_ms = (t3.tv_sec - t0.tv_sec) * 1000.0 + (t3.tv_nsec - t0.tv_nsec) / 1e6;
        double bandwidth_gbs = (size / 1e9) / (total_ms / 1000.0);
        fprintf(stderr, "[GATHER] size=%zu bytes (%.2f MB), gpu_cmd=%.3fms, submit=%.3fms, memcpy=%.3fms, total=%.3fms, BW=%.2f GB/s\n",
                size, size / 1e6, gpu_cmd_ms, submit_ms, memcpy_ms, total_ms, bandwidth_gbs);
#endif

        return ADAMAH_OK;
    }

    // GPU gather path: use compute shader for sparse reads
    if (!ctx.gather_pipe.pipeline) init_pipelines();  // Ensure pipeline is ready
    if (ctx.gather_pipe.pipeline && n_locs >= gpu_threshold) {
        debug_path("gather", "gpu_sparse");
        VkDeviceSize data_size = (VkDeviceSize)n_locs * pack_bytes;
        VkDeviceSize locs_size = (VkDeviceSize)n_locs * sizeof(uint32_t);

        VkBufferUsageFlags dev_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        GpuBuf* dst_buf = get_or_create_buf_ex("_gather_dst", n_locs * m->pack_size, m->word_size, 1, dev_usage);
        GpuBuf* locs_buf = get_or_create_buf_ex("_gather_locs", n_locs, 4, 1, dev_usage);
        if (!dst_buf || !locs_buf) goto fallback;

        GpuBuf* stage_up = get_or_create_buf_ex("_stage_upload", n_locs * 4, 1, 0,
                                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        GpuBuf* stage_down = get_or_create_buf_ex("_stage_download", (uint32_t)data_size, 1, 0,
                                                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        if (!stage_up || !stage_down || !stage_up->ptr || !stage_down->ptr) goto fallback;

        memcpy(stage_up->ptr, locs, (size_t)locs_size);

        VkDescriptorBufferInfo buf_infos[3] = {
            { .buffer = m->buf, .range = VK_WHOLE_SIZE },
            { .buffer = dst_buf->buf, .range = VK_WHOLE_SIZE },
            { .buffer = locs_buf->buf, .range = VK_WHOLE_SIZE }
        };
        VkWriteDescriptorSet writes[3];
        for (int i = 0; i < 3; i++) {
            writes[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = ctx.gather_pipe.desc_set, .dstBinding = i, .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &buf_infos[i] };
        }
        vkUpdateDescriptorSets(ctx.device, 3, writes, 0, NULL);

        uint32_t push[2] = { n_locs, m->pack_size };
        uint32_t total_threads = n_locs * m->pack_size;

        cmd_begin();
        VkBufferCopy locs_copy = { .srcOffset = 0, .dstOffset = 0, .size = locs_size };
        vkCmdCopyBuffer(ctx.cmd, stage_up->buf, locs_buf->buf, 1, &locs_copy);
        VkBuffer trans_bufs[1] = { locs_buf->buf };
        cmd_buffer_barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                           trans_bufs, 1);

        vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.gather_pipe.pipeline);
        vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.gather_pipe.pipe_layout,
                                0, 1, &ctx.gather_pipe.desc_set, 0, NULL);
        vkCmdPushConstants(ctx.cmd, ctx.gather_pipe.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, push);
        vkCmdDispatch(ctx.cmd, (total_threads + 255) / 256, 1, 1);

        VkBuffer comp_bufs[1] = { dst_buf->buf };
        cmd_buffer_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                           comp_bufs, 1);
        VkBufferCopy dst_copy = { .srcOffset = 0, .dstOffset = 0, .size = data_size };
        vkCmdCopyBuffer(ctx.cmd, dst_buf->buf, stage_down->buf, 1, &dst_copy);
        cmd_submit();

        memcpy(data, stage_down->ptr, (size_t)data_size);

        return ADAMAH_OK;
    }

fallback:
    // Heuristic: large reads -> full buffer copy
    debug_path("gather", "fallback");
    if ((uint64_t)n_locs * (uint64_t)pack_bytes >= (uint64_t)m->total_bytes / 2) {
        cmd_begin();
        VkBufferCopy copy = { .size = m->total_bytes };
        vkCmdCopyBuffer(ctx.cmd, m->buf, m->staging, 1, &copy);
        cmd_submit();
        char* dst = (char*)data;
        const char* staging = (const char*)m->staging_ptr;
        for (uint32_t i = 0; i < n_locs; i++) {
            uint32_t loc = locs[i];
            if (loc >= m->n_packs) continue;
            memcpy(dst + (size_t)i * pack_bytes, staging + (size_t)loc * pack_bytes, pack_bytes);
        }
        return ADAMAH_OK;
    }

    VkBufferCopy* regions = (VkBufferCopy*)malloc(sizeof(VkBufferCopy) * n_locs);
    if (!regions) return ADAMAH_ERR_MEMORY;
    uint32_t rcount = 0;

    for (uint32_t i = 0; i < n_locs; i++) {
        uint32_t loc = locs[i];
        if (loc >= m->n_packs) continue;
        size_t off = (size_t)loc * pack_bytes;
        regions[rcount].srcOffset = off;
        regions[rcount].dstOffset = off;
        regions[rcount].size = pack_bytes;
        rcount++;
    }

    if (rcount > 0) {
        cmd_begin();
        vkCmdCopyBuffer(ctx.cmd, m->buf, m->staging, rcount, regions);
        cmd_submit();
    }
    free(regions);

    // Copy from staging
    char* dst = (char*)data;
    const char* staging = (const char*)m->staging_ptr;
    for (uint32_t i = 0; i < n_locs; i++) {
        uint32_t loc = locs[i];
        if (loc >= m->n_packs) continue;
        memcpy(dst + (size_t)i * pack_bytes, staging + (size_t)loc * pack_bytes, pack_bytes);
    }
    
    return ADAMAH_OK;
}

// ============================================
// Device-only async sparse I/O
// ============================================

uint64_t map_upload_dev(uint32_t handle, const void* data, uint32_t n_bytes) {
    if (!data || n_bytes == 0) return 0;
    if (!ctx.hot_pool || !ctx.cold_pool) {
        if (debug_enabled()) fprintf(stderr, "ADAMAH DEBUG: upload_dev missing cache pools\n");
        return 0;
    }

    uint32_t id = handle;
    if (id == 0) {
        if (res_alloc(RES_TYPE_CVAR, n_bytes, &id) != 0) {
            if (debug_enabled()) fprintf(stderr, "ADAMAH DEBUG: upload_dev res_alloc failed (bytes=%u)\n", n_bytes);
            return 0;
        }
    }
    ResEntry* r = res_get(id);
    if (!r) {
        if (debug_enabled()) fprintf(stderr, "ADAMAH DEBUG: upload_dev res_get failed (id=%u)\n", id);
        return 0;
    }
    if (r->size_bytes != n_bytes) {
        if (debug_enabled()) fprintf(stderr, "ADAMAH DEBUG: upload_dev size mismatch (id=%u size=%zu bytes=%u)\n",
                                     id, (size_t)r->size_bytes, n_bytes);
        return 0;
    }

    VkDeviceSize hot_off = 0;
    r->dirty = 1;
    if (res_require_hot(id, &hot_off) != 0) {
        if (debug_enabled()) fprintf(stderr, "ADAMAH DEBUG: upload_dev res_require_hot failed (id=%u)\n", id);
        return 0;
    }

    VkDeviceSize copy_size = align_up((VkDeviceSize)n_bytes, ctx.copy_align);
    if (copy_size > ctx.hot_pool->bytes_capacity) copy_size = (VkDeviceSize)n_bytes;
    GpuBuf* stage = get_or_create_buf_ex("_stage_upload", (uint32_t)copy_size, 1, 0,
                                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    if (!stage || !stage->ptr) {
        if (debug_enabled()) fprintf(stderr, "ADAMAH DEBUG: upload_dev staging alloc failed\n");
        return 0;
    }

    memcpy(stage->ptr, data, n_bytes);
    if (copy_size > n_bytes) {
        memset((char*)stage->ptr + n_bytes, 0, (size_t)(copy_size - n_bytes));
    }

    cmd_begin();
    VkBufferCopy copy = { .srcOffset = 0, .dstOffset = hot_off, .size = copy_size };
    vkCmdCopyBuffer(ctx.cmd, stage->buf, ctx.hot_pool->buf, 1, &copy);
    VkBuffer trans_bufs[1] = { ctx.hot_pool->buf };
    cmd_buffer_barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                       trans_bufs, 1);
    cmd_submit();

    return (uint64_t)id;
}

int map_download_dev(uint32_t handle, void* data, uint32_t n_bytes) {
    if (!data || n_bytes == 0) return ADAMAH_ERR_INVALID;
    if (!ctx.hot_pool || !ctx.cold_pool) return ADAMAH_ERR_INVALID;
    ResEntry* r = res_get(handle);
    if (!r) return ADAMAH_ERR_INVALID;
    if (n_bytes > r->size_bytes) return ADAMAH_ERR_INVALID;

    VkDeviceSize hot_off = 0;
    if (res_require_hot(handle, &hot_off) != 0) return ADAMAH_ERR_INVALID;

    VkDeviceSize copy_size = align_up((VkDeviceSize)n_bytes, ctx.copy_align);
    GpuBuf* stage = get_or_create_buf_ex("_stage_download", (uint32_t)copy_size, 1, 0,
                                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    if (!stage || !stage->ptr) return ADAMAH_ERR_MEMORY;

    cmd_begin();
    VkBuffer comp_bufs[1] = { ctx.hot_pool->buf };
    cmd_buffer_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                       comp_bufs, 1);
    VkBufferCopy copy = { .srcOffset = hot_off, .dstOffset = 0, .size = copy_size };
    vkCmdCopyBuffer(ctx.cmd, ctx.hot_pool->buf, stage->buf, 1, &copy);
    cmd_submit();

    memcpy(data, stage->ptr, n_bytes);
    return ADAMAH_OK;
}

uint64_t map_scatter_dev(uint32_t map_id, uint32_t locs_handle, uint32_t n_locs, uint32_t src_handle) {
    if (map_id >= MAX_MAPS || !ctx.maps[map_id].active) return 0;
    if (n_locs == 0) return 0;
    Map* m = &ctx.maps[map_id];
    if (!ctx.scatter_pipe.pipeline) init_pipelines();
    if (!ctx.scatter_pipe.pipeline) return 0;

    ResEntry* src_res = res_get(src_handle);
    ResEntry* locs_res = res_get(locs_handle);
    if (!src_res || !locs_res) return 0;
    if ((VkDeviceSize)n_locs * 4 > locs_res->size_bytes) return 0;
    VkDeviceSize needed_src = (VkDeviceSize)n_locs * m->pack_size * m->word_size;
    if (needed_src > src_res->size_bytes) return 0;

    res_pin(src_handle);
    res_pin(locs_handle);
    VkDeviceSize src_off = 0;
    VkDeviceSize locs_off = 0;
    if (res_require_hot(src_handle, &src_off) != 0 ||
        res_require_hot(locs_handle, &locs_off) != 0) {
        res_unpin(src_handle);
        res_unpin(locs_handle);
        return 0;
    }

    VkDescriptorSet ds = alloc_desc_set(&ctx.scatter_pipe);
    if (!ds) return 0;

    VkDescriptorBufferInfo buf_infos[3] = {
        { .buffer = m->buf, .range = VK_WHOLE_SIZE },
        { .buffer = ctx.hot_pool->buf, .offset = src_off, .range = src_res->size_bytes },
        { .buffer = ctx.hot_pool->buf, .offset = locs_off, .range = locs_res->size_bytes }
    };
    VkWriteDescriptorSet writes[3];
    for (int i = 0; i < 3; i++) {
        writes[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = ds, .dstBinding = i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &buf_infos[i] };
    }
    vkUpdateDescriptorSets(ctx.device, 3, writes, 0, NULL);

    uint32_t push[2] = { n_locs, m->pack_size };
    uint32_t total_threads = n_locs * m->pack_size;

    cmd_begin();
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.scatter_pipe.pipeline);
    vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.scatter_pipe.pipe_layout,
                            0, 1, &ds, 0, NULL);
    vkCmdPushConstants(ctx.cmd, ctx.scatter_pipe.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, push);
    vkCmdDispatch(ctx.cmd, (total_threads + 255) / 256, 1, 1);

    VkBuffer comp_bufs[1] = { m->buf };
    cmd_buffer_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                       comp_bufs, 1);

    cmd_submit();
    res_unpin(src_handle);
    res_unpin(locs_handle);
    return 0;
}

uint64_t map_gather_dev(uint32_t map_id, uint32_t locs_handle, uint32_t n_locs) {
    if (map_id >= MAX_MAPS || !ctx.maps[map_id].active) return 0;
    if (n_locs == 0) return 0;
    Map* m = &ctx.maps[map_id];
    if (!ctx.gather_pipe.pipeline) init_pipelines();
    if (!ctx.gather_pipe.pipeline) return 0;

    ResEntry* locs_res = res_get(locs_handle);
    if (!locs_res) return 0;
    if ((VkDeviceSize)n_locs * 4 > locs_res->size_bytes) return 0;

    res_pin(locs_handle);
    VkDeviceSize locs_off = 0;
    if (res_require_hot(locs_handle, &locs_off) != 0) {
        res_unpin(locs_handle);
        return 0;
    }

    uint32_t dst_id = 0;
    uint32_t dst_bytes = n_locs * m->pack_size * m->word_size;
    if (res_alloc(RES_TYPE_CVAR, dst_bytes, &dst_id) != 0) return 0;
    ResEntry* dst_res = res_get(dst_id);
    if (!dst_res) return 0;
    dst_res->dirty = 1;

    res_pin(dst_id);
    VkDeviceSize dst_off = 0;
    if (res_require_hot(dst_id, &dst_off) != 0) {
        res_unpin(locs_handle);
        res_unpin(dst_id);
        return 0;
    }

    VkDescriptorSet ds = alloc_desc_set(&ctx.gather_pipe);
    if (!ds) return 0;

    VkDescriptorBufferInfo buf_infos[3] = {
        { .buffer = m->buf, .range = VK_WHOLE_SIZE },
        { .buffer = ctx.hot_pool->buf, .offset = dst_off, .range = dst_res->size_bytes },
        { .buffer = ctx.hot_pool->buf, .offset = locs_off, .range = locs_res->size_bytes }
    };
    VkWriteDescriptorSet writes[3];
    for (int i = 0; i < 3; i++) {
        writes[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = ds, .dstBinding = i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &buf_infos[i] };
    }
    vkUpdateDescriptorSets(ctx.device, 3, writes, 0, NULL);

    uint32_t push[2] = { n_locs, m->pack_size };
    uint32_t total_threads = n_locs * m->pack_size;

    cmd_begin();
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.gather_pipe.pipeline);
    vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.gather_pipe.pipe_layout,
                            0, 1, &ds, 0, NULL);
    vkCmdPushConstants(ctx.cmd, ctx.gather_pipe.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, push);
    vkCmdDispatch(ctx.cmd, (total_threads + 255) / 256, 1, 1);

    VkBuffer comp_bufs[1] = { ctx.hot_pool->buf };
    cmd_buffer_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                       comp_bufs, 1);

    cmd_submit();
    res_unpin(locs_handle);
    res_unpin(dst_id);
    return (uint64_t)dst_id;
}

// Backward-compatible wrappers
int mscatter(uint32_t map_id, const uint32_t* locs, const void* data, uint32_t n_locs) {
    return map_scatter(map_id, locs, data, n_locs);
}

int mgather(uint32_t map_id, const uint32_t* locs, void* data, uint32_t n_locs) {
    return map_gather(map_id, locs, data, n_locs);
}

// ============================================
// ============================================
// Persistence
// ============================================

int map_save(uint32_t id, const char* path) {
    if (id >= MAX_MAPS || !ctx.maps[id].active) return ADAMAH_ERR_INVALID;
    Map* m = &ctx.maps[id];
    
    // Download
    cmd_begin();
    VkBufferCopy copy = { .size = m->total_bytes };
    vkCmdCopyBuffer(ctx.cmd, m->buf, m->staging, 1, &copy);
    cmd_submit();
    
    FILE* f = fopen(path, "wb");
    if (!f) return ADAMAH_ERR_INVALID;
    fwrite(&m->word_size, 4, 1, f);
    fwrite(&m->pack_size, 4, 1, f);
    fwrite(&m->n_packs, 4, 1, f);
    fwrite(m->staging_ptr, 1, m->total_bytes, f);
    fclose(f);
    return ADAMAH_OK;
}

int map_load(uint32_t id, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return ADAMAH_ERR_INVALID;
    
    uint32_t ws, ps, np;
    if (fread(&ws, 4, 1, f) != 1 ||
        fread(&ps, 4, 1, f) != 1 ||
        fread(&np, 4, 1, f) != 1) {
        fclose(f);
        return ADAMAH_ERR_INVALID;
    }
    
    int ret = map_init(id, ws, ps, np);
    if (ret != ADAMAH_OK) { fclose(f); return ret; }
    
    Map* m = &ctx.maps[id];
    if (m->total_bytes > 0) {
        size_t got = fread(m->staging_ptr, 1, m->total_bytes, f);
        if (got != (size_t)m->total_bytes) {
            fclose(f);
            return ADAMAH_ERR_INVALID;
        }
    }
    fclose(f);
    
    // Upload
    cmd_begin();
    VkBufferCopy copy = { .size = m->total_bytes };
    vkCmdCopyBuffer(ctx.cmd, m->staging, m->buf, 1, &copy);
    cmd_submit();
    
    return ADAMAH_OK;
}

// ============================================
// Shader Loading & Pipeline Creation
// ============================================

static uint32_t* load_spv(const char* name, size_t* size) {
    char path[600];
    snprintf(path, sizeof(path), "%s/%s", ctx.shader_path, name);
    
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    
    fseek(f, 0, SEEK_END);
    long end = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (end <= 0) { fclose(f); return NULL; }
    *size = (size_t)end;
    
    uint32_t* code = malloc(*size);
    if (!code) { fclose(f); return NULL; }
    size_t got = fread(code, 1, *size, f);
    if (got != *size) {
        free(code);
        fclose(f);
        return NULL;
    }
    fclose(f);
    return code;
}

static int create_pipeline(Pipeline* p, const char* shader_name, int num_bindings, size_t push_size) {
    size_t code_size;
    uint32_t* code = load_spv(shader_name, &code_size);
    if (!code) return ADAMAH_ERR_INVALID;
    
    // Create shader module
    VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code_size, .pCode = code };
    VkResult res = vkCreateShaderModule(ctx.device, &smci, NULL, &p->shader);
    free(code);
    if (res != VK_SUCCESS) return ADAMAH_ERR_VULKAN;
    
    // Descriptor set layout
    VkDescriptorSetLayoutBinding bindings[8];  // Max 8 bindings
    for (int i = 0; i < num_bindings && i < 8; i++) {
        bindings[i] = (VkDescriptorSetLayoutBinding){
            .binding = i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
    }
    VkDescriptorSetLayoutCreateInfo dslci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = num_bindings, .pBindings = bindings };
    vkCreateDescriptorSetLayout(ctx.device, &dslci, NULL, &p->desc_layout);
    
    // Pipeline layout
    VkPushConstantRange pcr = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = push_size };
    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &p->desc_layout,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
    vkCreatePipelineLayout(ctx.device, &plci, NULL, &p->pipe_layout);
    
    // Compute pipeline
    VkComputePipelineCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = p->shader, .pName = "main" },
        .layout = p->pipe_layout };
    vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpci, NULL, &p->pipeline);
    
    // Descriptor pool
    const uint32_t max_sets = 8192;
    VkDescriptorPoolSize pool_size = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = (uint32_t)num_bindings * max_sets };
    VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = max_sets, .poolSizeCount = 1, .pPoolSizes = &pool_size };
    vkCreateDescriptorPool(ctx.device, &dpci, NULL, &p->desc_pool);
    
    // Allocate descriptor set
    VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = p->desc_pool, .descriptorSetCount = 1, .pSetLayouts = &p->desc_layout };
    vkAllocateDescriptorSets(ctx.device, &dsai, &p->desc_set);
    
    return ADAMAH_OK;
}

static void destroy_pipeline(Pipeline* p) {
    if (p->pipeline) vkDestroyPipeline(ctx.device, p->pipeline, NULL);
    if (p->pipe_layout) vkDestroyPipelineLayout(ctx.device, p->pipe_layout, NULL);
    if (p->desc_pool) vkDestroyDescriptorPool(ctx.device, p->desc_pool, NULL);
    if (p->desc_layout) vkDestroyDescriptorSetLayout(ctx.device, p->desc_layout, NULL);
    if (p->shader) vkDestroyShaderModule(ctx.device, p->shader, NULL);
    memset(p, 0, sizeof(Pipeline));
}

static int init_pipelines(void) {
    // Find shader path - check compile-time define first, then runtime paths
    const char* paths[] = { 
#ifdef SHADER_PATH
        SHADER_PATH,
#endif
        "./shaders", 
        "./src/adamah/shaders", 
        "/usr/share/adamah/shaders",
        NULL
    };
    
    for (int i = 0; paths[i] != NULL; i++) {
        char test[600];
        snprintf(test, sizeof(test), "%s/map_op1.spv", paths[i]);
        FILE* f = fopen(test, "rb");
        if (f) { 
            fclose(f); 
            strncpy(ctx.shader_path, paths[i], 511); 
            break; 
        }
    }
    
    if (!ctx.shader_path[0]) {
        fprintf(stderr, "ADAMAH: Cannot find shaders\n");
        fprintf(stderr, "Searched in:\n");
        for (int i = 0; paths[i] != NULL; i++) {
            fprintf(stderr, "  - %s\n", paths[i]);
        }
        return ADAMAH_ERR_INVALID;
    }
    
    // Unary: op, n_locs, pack_size = 12 bytes, 3 bindings
    if (create_pipeline(&ctx.unary_pipe, "map_op1.spv", 3, 12) != 0) return ADAMAH_ERR_VULKAN;
    
    // Binary: op, n_locs, pack_size = 12 bytes, 4 bindings
    if (create_pipeline(&ctx.binary_pipe, "map_op2.spv", 4, 12) != 0) return ADAMAH_ERR_VULKAN;
    
    // Matmul: M, K, N, n_ops = 16 bytes, 4 bindings (map, locs_a, locs_b, locs_c)
    if (create_pipeline(&ctx.matmul_pipe, "map_matmul.spv", 4, 16) != 0) {
        fprintf(stderr, "ADAMAH: Warning - matmul shader not found\n");
        // Not fatal, continue without matmul
    }
    
    // Reduce: op, n_locs, pack_size = 12 bytes, 3 bindings
    if (create_pipeline(&ctx.reduce_pipe, "map_reduce.spv", 3, 12) != 0) {
        fprintf(stderr, "ADAMAH: Warning - reduce shader not found\n");
    }
    if (create_pipeline(&ctx.reduce_small_pipe, "map_reduce_small.spv", 3, 12) != 0) {
        fprintf(stderr, "ADAMAH: Warning - reduce_small shader not found\n");
    }
    
    // Broadcast: op, n_locs, pack_size = 12 bytes, 4 bindings
    if (create_pipeline(&ctx.broadcast_pipe, "map_broadcast.spv", 4, 12) != 0) {
        fprintf(stderr, "ADAMAH: Warning - broadcast shader not found\n");
    }
    
    // Softmax: n_rows, row_size = 8 bytes, 3 bindings
    if (create_pipeline(&ctx.softmax_pipe, "map_softmax.spv", 3, 8) != 0) {
        fprintf(stderr, "ADAMAH: Warning - softmax shader not found\n");
    }
    
    // LayerNorm: n_rows, dim, eps = 12 bytes, 5 bindings
    if (create_pipeline(&ctx.layernorm_pipe, "map_layernorm.spv", 5, 12) != 0) {
        fprintf(stderr, "ADAMAH: Warning - layernorm shader not found\n");
    }

    
    // Scatter: n_locs, pack_size = 8 bytes, 3 bindings (map, src, locs)
    if (create_pipeline(&ctx.scatter_pipe, "map_scatter.spv", 3, 8) != 0) {
        fprintf(stderr, "ADAMAH: Warning - scatter shader not found\n");
    }
    
    // Gather: n_locs, pack_size = 8 bytes, 3 bindings (map, dst, locs)
    if (create_pipeline(&ctx.gather_pipe, "map_gather.spv", 3, 8) != 0) {
        fprintf(stderr, "ADAMAH: Warning - gather shader not found\n");
    }
    
    return ADAMAH_OK;
}

// ============================================
// Internal GPU buffers for locations
// ============================================

static GpuBuf* get_or_create_buf_ex(const char* base_name, uint32_t n_elems, uint32_t elem_size,
                                    int device_local, VkBufferUsageFlags usage) {
    VkDeviceSize bytes_needed = (VkDeviceSize)n_elems * (VkDeviceSize)elem_size;
    if (bytes_needed == 0) bytes_needed = 1;
    // In batch mode, create unique buffer name for each op
    char name[64];
    if (batch_mode) {
        snprintf(name, sizeof(name), "%s_%d", base_name, batch_op_counter);
    } else {
        strncpy(name, base_name, 63);
        name[63] = 0;
    }
    
    VkDeviceSize aligned_needed = align_up(bytes_needed, ctx.copy_align);
    VkDeviceSize desired_capacity = bytes_needed;
    if (device_local) {
        desired_capacity = device_local_bucket(aligned_needed);
    } else if (is_stage_upload_name(name) || is_stage_download_name(name)) {
        desired_capacity = next_pow2(aligned_needed);
    }

    // Find existing
    for (int i = 0; i < ctx.buf_count; i++) {
        if (strcmp(ctx.bufs[i].name, name) == 0) {
            if (ctx.bufs[i].bytes_capacity >= desired_capacity &&
                ctx.bufs[i].device_local == device_local &&
                ctx.bufs[i].usage == usage) {
                return &ctx.bufs[i];
            }
            // Need to resize or mismatch - destroy old
            if (ctx.bufs[i].bytes_capacity < desired_capacity) {
                if (is_stage_upload_name(name)) ctx.stage_upload_grow_events++;
                if (is_stage_download_name(name)) ctx.stage_download_grow_events++;
            }
            ctx.num_buffer_recreates++;
            if (ctx.bufs[i].ptr) vkUnmapMemory(ctx.device, ctx.bufs[i].mem);
            vkDestroyBuffer(ctx.device, ctx.bufs[i].buf, NULL);
            vkFreeMemory(ctx.device, ctx.bufs[i].mem, NULL);
            ctx.bufs[i].buf = VK_NULL_HANDLE;
            ctx.bufs[i].ptr = NULL;
        }
    }
    
    // Find or create slot
    GpuBuf* b = NULL;
    for (int i = 0; i < ctx.buf_count; i++) {
        if (strcmp(ctx.bufs[i].name, name) == 0) { b = &ctx.bufs[i]; break; }
    }
    if (!b && ctx.buf_count < MAX_BUFS) {
        b = &ctx.bufs[ctx.buf_count++];
    }
    if (!b) return NULL;
    
    strncpy(b->name, name, 63);
    b->bytes_capacity = desired_capacity;
    b->elem_size = elem_size;
    b->device_local = device_local;
    b->usage = usage;
    
    if (create_buffer(&b->buf, &b->mem, desired_capacity, usage, device_local) != 0) return NULL;
    if (!device_local) {
        vkMapMemory(ctx.device, b->mem, 0, desired_capacity, 0, &b->ptr);
    } else {
        b->ptr = NULL;
    }
    
    return b;
}

static GpuBuf* get_or_create_buf(const char* base_name, uint32_t n_elems, uint32_t elem_size) {
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    return get_or_create_buf_ex(base_name, n_elems, elem_size, 0, usage);
}

// ============================================
// Map Operations (Pure GPU)
// ============================================

int map_op1(uint32_t map_id, uint32_t op, const uint32_t* locs_src, const uint32_t* locs_dst, uint32_t n) {
    // Legacy path removed: use map_op1_dev with cached locs.
    (void)map_id;
    (void)op;
    (void)locs_src;
    (void)locs_dst;
    (void)n;
    return ADAMAH_ERR_INVALID;
}

int map_op1_dev(uint32_t map_id, uint32_t op, uint32_t locs_src_handle, uint32_t locs_dst_handle, uint32_t n) {
    if (map_id >= MAX_MAPS || !ctx.maps[map_id].active) return ADAMAH_ERR_INVALID;
    if (n == 0) return ADAMAH_OK;
    if (!ctx.unary_pipe.pipeline && init_pipelines() != 0) return ADAMAH_ERR_VULKAN;

    Map* m = &ctx.maps[map_id];
    ResEntry* src_res = res_get(locs_src_handle);
    ResEntry* dst_res = res_get(locs_dst_handle);
    if (!src_res || !dst_res) return ADAMAH_ERR_INVALID;
    if ((VkDeviceSize)n * 4 > src_res->size_bytes) return ADAMAH_ERR_INVALID;
    if ((VkDeviceSize)n * 4 > dst_res->size_bytes) return ADAMAH_ERR_INVALID;
    res_pin(locs_src_handle);
    res_pin(locs_dst_handle);
    VkDeviceSize src_off = 0;
    VkDeviceSize dst_off = 0;
    if (res_require_hot(locs_src_handle, &src_off) != 0 ||
        res_require_hot(locs_dst_handle, &dst_off) != 0) {
        res_unpin(locs_src_handle);
        res_unpin(locs_dst_handle);
        return ADAMAH_ERR_INVALID;
    }

    VkDescriptorSet ds = alloc_desc_set(&ctx.unary_pipe);
    if (ds == VK_NULL_HANDLE) return ADAMAH_ERR_VULKAN;

    VkDescriptorBufferInfo buf_infos[3] = {
        { .buffer = m->buf, .range = VK_WHOLE_SIZE },
        { .buffer = ctx.hot_pool->buf, .offset = src_off, .range = src_res->size_bytes },
        { .buffer = ctx.hot_pool->buf, .offset = dst_off, .range = dst_res->size_bytes }
    };
    VkWriteDescriptorSet writes[3];
    for (int i = 0; i < 3; i++) {
        writes[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = ds, .dstBinding = i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &buf_infos[i] };
    }
    vkUpdateDescriptorSets(ctx.device, 3, writes, 0, NULL);

    uint32_t push[3] = { op, n, m->pack_size };
    uint32_t total_threads = n * m->pack_size;

    cmd_begin();
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.unary_pipe.pipeline);
    vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.unary_pipe.pipe_layout,
                            0, 1, &ds, 0, NULL);
    vkCmdPushConstants(ctx.cmd, ctx.unary_pipe.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, push);
    vkCmdDispatch(ctx.cmd, (total_threads + 255) / 256, 1, 1);
    cmd_barrier_after_dispatch();
    cmd_submit();
    res_unpin(locs_src_handle);
    res_unpin(locs_dst_handle);

    return ADAMAH_OK;
}

int map_op2(uint32_t map_id, uint32_t op,
            const uint32_t* locs_a, const uint32_t* locs_b, const uint32_t* locs_dst, uint32_t n) {
    // Legacy path removed: use map_op2_dev with cached locs.
    (void)map_id;
    (void)op;
    (void)locs_a;
    (void)locs_b;
    (void)locs_dst;
    (void)n;
    return ADAMAH_ERR_INVALID;
}

int map_op2_dev(uint32_t map_id, uint32_t op,
                uint32_t locs_a_handle, uint32_t locs_b_handle, uint32_t locs_dst_handle, uint32_t n) {
    if (map_id >= MAX_MAPS || !ctx.maps[map_id].active) return ADAMAH_ERR_INVALID;
    if (n == 0) return ADAMAH_OK;
    if (!ctx.binary_pipe.pipeline && init_pipelines() != 0) return ADAMAH_ERR_VULKAN;

    Map* m = &ctx.maps[map_id];
    ResEntry* a_res = res_get(locs_a_handle);
    ResEntry* b_res = res_get(locs_b_handle);
    ResEntry* dst_res = res_get(locs_dst_handle);
    if (!a_res || !b_res || !dst_res) return ADAMAH_ERR_INVALID;
    if ((VkDeviceSize)n * 4 > a_res->size_bytes) return ADAMAH_ERR_INVALID;
    if ((VkDeviceSize)n * 4 > b_res->size_bytes) return ADAMAH_ERR_INVALID;
    if ((VkDeviceSize)n * 4 > dst_res->size_bytes) return ADAMAH_ERR_INVALID;
    res_pin(locs_a_handle);
    res_pin(locs_b_handle);
    res_pin(locs_dst_handle);
    VkDeviceSize a_off = 0;
    VkDeviceSize b_off = 0;
    VkDeviceSize dst_off = 0;
    if (res_require_hot(locs_a_handle, &a_off) != 0 ||
        res_require_hot(locs_b_handle, &b_off) != 0 ||
        res_require_hot(locs_dst_handle, &dst_off) != 0) {
        res_unpin(locs_a_handle);
        res_unpin(locs_b_handle);
        res_unpin(locs_dst_handle);
        return ADAMAH_ERR_INVALID;
    }

    VkDescriptorSet ds = alloc_desc_set(&ctx.binary_pipe);
    if (ds == VK_NULL_HANDLE) return ADAMAH_ERR_VULKAN;

    VkDescriptorBufferInfo buf_infos[4] = {
        { .buffer = m->buf, .range = VK_WHOLE_SIZE },
        { .buffer = ctx.hot_pool->buf, .offset = a_off, .range = a_res->size_bytes },
        { .buffer = ctx.hot_pool->buf, .offset = b_off, .range = b_res->size_bytes },
        { .buffer = ctx.hot_pool->buf, .offset = dst_off, .range = dst_res->size_bytes }
    };
    VkWriteDescriptorSet writes[4];
    for (int i = 0; i < 4; i++) {
        writes[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = ds, .dstBinding = i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &buf_infos[i] };
    }
    vkUpdateDescriptorSets(ctx.device, 4, writes, 0, NULL);

    uint32_t push[3] = { op, n, m->pack_size };
    uint32_t total_threads = n * m->pack_size;

    cmd_begin();
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.binary_pipe.pipeline);
    vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.binary_pipe.pipe_layout,
                            0, 1, &ds, 0, NULL);
    vkCmdPushConstants(ctx.cmd, ctx.binary_pipe.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, push);
    vkCmdDispatch(ctx.cmd, (total_threads + 255) / 256, 1, 1);
    cmd_barrier_after_dispatch();
    cmd_submit();
    res_unpin(locs_a_handle);
    res_unpin(locs_b_handle);
    res_unpin(locs_dst_handle);

    return ADAMAH_OK;
}

// ============================================
// Matrix Multiplication: C = A @ B
// A: (M x K), B: (K x N), C: (M x N)
// ============================================
int map_matmul(uint32_t map_id,
               const uint32_t* locs_a, const uint32_t* locs_b, const uint32_t* locs_c,
               uint32_t M, uint32_t K, uint32_t N, uint32_t n_ops) {
    // Legacy path removed: use map_matmul_dev with cached locs.
    (void)map_id;
    (void)locs_a;
    (void)locs_b;
    (void)locs_c;
    (void)M;
    (void)K;
    (void)N;
    (void)n_ops;
    return ADAMAH_ERR_INVALID;
}

int map_matmul_dev(uint32_t map_id,
                   uint32_t locs_a_handle, uint32_t locs_b_handle, uint32_t locs_c_handle,
                   uint32_t M, uint32_t K, uint32_t N, uint32_t n_ops) {
    if (map_id >= MAX_MAPS || !ctx.maps[map_id].active) {
        if (debug_enabled()) fprintf(stderr, "ADAMAH DEBUG: matmul invalid map_id=%u\n", map_id);
        return ADAMAH_ERR_INVALID;
    }
    if (n_ops == 0) return ADAMAH_OK;
    if (!ctx.matmul_pipe.pipeline && init_pipelines() != 0) return ADAMAH_ERR_VULKAN;
    if (!ctx.matmul_pipe.pipeline) {
        if (debug_enabled()) fprintf(stderr, "ADAMAH DEBUG: matmul pipeline not available\n");
        return ADAMAH_ERR_INVALID;
    }

    Map* m = &ctx.maps[map_id];
    ResEntry* a_res = res_get(locs_a_handle);
    ResEntry* b_res = res_get(locs_b_handle);
    ResEntry* c_res = res_get(locs_c_handle);
    if (!a_res || !b_res || !c_res) {
        if (debug_enabled()) {
            fprintf(stderr, "ADAMAH DEBUG: matmul res missing a=%u b=%u c=%u\n",
                    locs_a_handle, locs_b_handle, locs_c_handle);
        }
        return ADAMAH_ERR_INVALID;
    }
    if ((VkDeviceSize)n_ops * 4 > a_res->size_bytes ||
        (VkDeviceSize)n_ops * 4 > b_res->size_bytes ||
        (VkDeviceSize)n_ops * 4 > c_res->size_bytes) {
        if (debug_enabled()) {
            fprintf(stderr, "ADAMAH DEBUG: matmul locs size mismatch n_ops=%u sizes=%zu,%zu,%zu\n",
                    n_ops, (size_t)a_res->size_bytes, (size_t)b_res->size_bytes, (size_t)c_res->size_bytes);
        }
        return ADAMAH_ERR_INVALID;
    }
    res_pin(locs_a_handle);
    res_pin(locs_b_handle);
    res_pin(locs_c_handle);
    VkDeviceSize a_off = 0;
    VkDeviceSize b_off = 0;
    VkDeviceSize c_off = 0;
    if (res_require_hot(locs_a_handle, &a_off) != 0 ||
        res_require_hot(locs_b_handle, &b_off) != 0 ||
        res_require_hot(locs_c_handle, &c_off) != 0) {
        if (debug_enabled()) fprintf(stderr, "ADAMAH DEBUG: matmul res_require_hot failed\n");
        res_unpin(locs_a_handle);
        res_unpin(locs_b_handle);
        res_unpin(locs_c_handle);
        return ADAMAH_ERR_INVALID;
    }

    VkDescriptorSet ds = alloc_desc_set(&ctx.matmul_pipe);
    if (ds == VK_NULL_HANDLE) return ADAMAH_ERR_VULKAN;

    VkDescriptorBufferInfo buf_infos[4] = {
        { .buffer = m->buf, .range = VK_WHOLE_SIZE },
        { .buffer = ctx.hot_pool->buf, .offset = a_off, .range = a_res->size_bytes },
        { .buffer = ctx.hot_pool->buf, .offset = b_off, .range = b_res->size_bytes },
        { .buffer = ctx.hot_pool->buf, .offset = c_off, .range = c_res->size_bytes }
    };
    VkWriteDescriptorSet writes[4];
    for (int i = 0; i < 4; i++) {
        writes[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = ds, .dstBinding = i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &buf_infos[i] };
    }
    vkUpdateDescriptorSets(ctx.device, 4, writes, 0, NULL);

    uint32_t push[4] = { M, K, N, n_ops };
    uint32_t grid_x = (M + 15) / 16;
    uint32_t grid_y = (N + 15) / 16;

    cmd_begin();
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.matmul_pipe.pipeline);
    vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.matmul_pipe.pipe_layout,
                            0, 1, &ds, 0, NULL);
    vkCmdPushConstants(ctx.cmd, ctx.matmul_pipe.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, push);
    vkCmdDispatch(ctx.cmd, grid_x, grid_y, n_ops);
    cmd_barrier_after_dispatch();
    cmd_submit();
    res_unpin(locs_a_handle);
    res_unpin(locs_b_handle);
    res_unpin(locs_c_handle);

    return ADAMAH_OK;
}

// ============================================
// Reduce: sum/max/min along pack dimension
// ============================================
#define REDUCE_SUM 0
#define REDUCE_MAX 1
#define REDUCE_MIN 2

int map_reduce(uint32_t map_id, uint32_t op,
               const uint32_t* locs_src, const uint32_t* locs_dst, uint32_t n) {
    // Legacy path removed: use map_reduce_dev with cached locs.
    (void)map_id;
    (void)op;
    (void)locs_src;
    (void)locs_dst;
    (void)n;
    return ADAMAH_ERR_INVALID;
}

int map_reduce_dev(uint32_t map_id, uint32_t op,
                   uint32_t locs_src_handle, uint32_t locs_dst_handle, uint32_t n) {
    if (map_id >= MAX_MAPS || !ctx.maps[map_id].active) return ADAMAH_ERR_INVALID;
    if (n == 0) return ADAMAH_OK;

    Map* m = &ctx.maps[map_id];
    if (m->pack_size == 1) {
        // Reduce of a single element is a copy; avoid 256-thread reduction.
        return map_op1_dev(map_id, 255, locs_src_handle, locs_dst_handle, n);
    }
    if (!ctx.reduce_pipe.pipeline && init_pipelines() != 0) return ADAMAH_ERR_VULKAN;
    Pipeline* reduce_pipe = &ctx.reduce_pipe;
    if (m->pack_size <= 64 && ctx.reduce_small_pipe.pipeline) {
        reduce_pipe = &ctx.reduce_small_pipe;
    }
    if (!reduce_pipe->pipeline) return ADAMAH_ERR_INVALID;
    ResEntry* src_res = res_get(locs_src_handle);
    ResEntry* dst_res = res_get(locs_dst_handle);
    if (!src_res || !dst_res) return ADAMAH_ERR_INVALID;
    if ((VkDeviceSize)n * 4 > src_res->size_bytes) return ADAMAH_ERR_INVALID;
    if ((VkDeviceSize)n * 4 > dst_res->size_bytes) return ADAMAH_ERR_INVALID;
    res_pin(locs_src_handle);
    res_pin(locs_dst_handle);
    VkDeviceSize src_off = 0;
    VkDeviceSize dst_off = 0;
    if (res_require_hot(locs_src_handle, &src_off) != 0 ||
        res_require_hot(locs_dst_handle, &dst_off) != 0) {
        res_unpin(locs_src_handle);
        res_unpin(locs_dst_handle);
        return ADAMAH_ERR_INVALID;
    }

    VkDescriptorSet ds = alloc_desc_set(reduce_pipe);
    if (ds == VK_NULL_HANDLE) return ADAMAH_ERR_VULKAN;

    VkDescriptorBufferInfo buf_infos[3] = {
        { .buffer = m->buf, .range = VK_WHOLE_SIZE },
        { .buffer = ctx.hot_pool->buf, .offset = src_off, .range = src_res->size_bytes },
        { .buffer = ctx.hot_pool->buf, .offset = dst_off, .range = dst_res->size_bytes }
    };
    VkWriteDescriptorSet writes[3];
    for (int i = 0; i < 3; i++) {
        writes[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = ds, .dstBinding = i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &buf_infos[i] };
    }
    vkUpdateDescriptorSets(ctx.device, 3, writes, 0, NULL);

    uint32_t push[3] = { op, n, m->pack_size };

    cmd_begin();
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, reduce_pipe->pipeline);
    vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, reduce_pipe->pipe_layout,
                            0, 1, &ds, 0, NULL);
    vkCmdPushConstants(ctx.cmd, reduce_pipe->pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, push);
    vkCmdDispatch(ctx.cmd, n, 1, 1);
    cmd_barrier_after_dispatch();
    cmd_submit();
    res_unpin(locs_src_handle);
    res_unpin(locs_dst_handle);

    return ADAMAH_OK;
}

// ============================================
// Broadcast: element-wise op with scalar
// dst = src op scalar (scalar broadcast to all elements)
// ============================================
#define BROADCAST_MUL 0
#define BROADCAST_DIV 1
#define BROADCAST_ADD 2
#define BROADCAST_SUB 3

int map_broadcast(uint32_t map_id, uint32_t op,
                  const uint32_t* locs_src, const uint32_t* locs_scalar, const uint32_t* locs_dst,
                  uint32_t n) {
    // Legacy path removed: use map_broadcast_dev with cached locs.
    (void)map_id;
    (void)op;
    (void)locs_src;
    (void)locs_scalar;
    (void)locs_dst;
    (void)n;
    return ADAMAH_ERR_INVALID;
}

int map_broadcast_dev(uint32_t map_id, uint32_t op,
                      uint32_t locs_src_handle, uint32_t locs_scalar_handle, uint32_t locs_dst_handle,
                      uint32_t n) {
    if (map_id >= MAX_MAPS || !ctx.maps[map_id].active) return ADAMAH_ERR_INVALID;
    if (n == 0) return ADAMAH_OK;
    if (!ctx.broadcast_pipe.pipeline && init_pipelines() != 0) return ADAMAH_ERR_VULKAN;
    if (!ctx.broadcast_pipe.pipeline) return ADAMAH_ERR_INVALID;

    Map* m = &ctx.maps[map_id];
    ResEntry* src_res = res_get(locs_src_handle);
    ResEntry* scalar_res = res_get(locs_scalar_handle);
    ResEntry* dst_res = res_get(locs_dst_handle);
    if (!src_res || !scalar_res || !dst_res) return ADAMAH_ERR_INVALID;
    if ((VkDeviceSize)n * 4 > src_res->size_bytes) return ADAMAH_ERR_INVALID;
    if ((VkDeviceSize)n * 4 > scalar_res->size_bytes) return ADAMAH_ERR_INVALID;
    if ((VkDeviceSize)n * 4 > dst_res->size_bytes) return ADAMAH_ERR_INVALID;
    res_pin(locs_src_handle);
    res_pin(locs_scalar_handle);
    res_pin(locs_dst_handle);
    VkDeviceSize src_off = 0;
    VkDeviceSize scalar_off = 0;
    VkDeviceSize dst_off = 0;
    if (res_require_hot(locs_src_handle, &src_off) != 0 ||
        res_require_hot(locs_scalar_handle, &scalar_off) != 0 ||
        res_require_hot(locs_dst_handle, &dst_off) != 0) {
        res_unpin(locs_src_handle);
        res_unpin(locs_scalar_handle);
        res_unpin(locs_dst_handle);
        return ADAMAH_ERR_INVALID;
    }

    VkDescriptorSet ds = alloc_desc_set(&ctx.broadcast_pipe);
    if (ds == VK_NULL_HANDLE) return ADAMAH_ERR_VULKAN;

    VkDescriptorBufferInfo buf_infos[4] = {
        { .buffer = m->buf, .range = VK_WHOLE_SIZE },
        { .buffer = ctx.hot_pool->buf, .offset = src_off, .range = src_res->size_bytes },
        { .buffer = ctx.hot_pool->buf, .offset = scalar_off, .range = scalar_res->size_bytes },
        { .buffer = ctx.hot_pool->buf, .offset = dst_off, .range = dst_res->size_bytes }
    };
    VkWriteDescriptorSet writes[4];
    for (int i = 0; i < 4; i++) {
        writes[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = ds, .dstBinding = i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &buf_infos[i] };
    }
    vkUpdateDescriptorSets(ctx.device, 4, writes, 0, NULL);

    uint32_t push[3] = { op, n, m->pack_size };
    uint32_t total_threads = n * m->pack_size;

    cmd_begin();
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.broadcast_pipe.pipeline);
    vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.broadcast_pipe.pipe_layout,
                            0, 1, &ds, 0, NULL);
    vkCmdPushConstants(ctx.cmd, ctx.broadcast_pipe.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, push);
    vkCmdDispatch(ctx.cmd, (total_threads + 255) / 256, 1, 1);
    cmd_barrier_after_dispatch();
    cmd_submit();
    res_unpin(locs_src_handle);
    res_unpin(locs_scalar_handle);
    res_unpin(locs_dst_handle);

    return ADAMAH_OK;
}

// ============================================
// Softmax: fused max-subtract-exp-sum-normalize
// ============================================
int map_softmax(uint32_t map_id,
                const uint32_t* locs_src, const uint32_t* locs_dst,
                uint32_t n_rows, uint32_t row_size) {
    // Legacy path removed: use map_softmax_dev with cached locs.
    (void)map_id;
    (void)locs_src;
    (void)locs_dst;
    (void)n_rows;
    (void)row_size;
    return ADAMAH_ERR_INVALID;
}

int map_softmax_dev(uint32_t map_id,
                    uint32_t locs_src_handle, uint32_t locs_dst_handle,
                    uint32_t n_rows, uint32_t row_size) {
    if (map_id >= MAX_MAPS || !ctx.maps[map_id].active) return ADAMAH_ERR_INVALID;
    if (n_rows == 0) return ADAMAH_OK;
    if (!ctx.softmax_pipe.pipeline && init_pipelines() != 0) return ADAMAH_ERR_VULKAN;
    if (!ctx.softmax_pipe.pipeline) return ADAMAH_ERR_INVALID;

    Map* m = &ctx.maps[map_id];
    ResEntry* src_res = res_get(locs_src_handle);
    ResEntry* dst_res = res_get(locs_dst_handle);
    if (!src_res || !dst_res) return ADAMAH_ERR_INVALID;
    if ((VkDeviceSize)n_rows * 4 > src_res->size_bytes) return ADAMAH_ERR_INVALID;
    if ((VkDeviceSize)n_rows * 4 > dst_res->size_bytes) return ADAMAH_ERR_INVALID;
    res_pin(locs_src_handle);
    res_pin(locs_dst_handle);
    VkDeviceSize src_off = 0;
    VkDeviceSize dst_off = 0;
    if (res_require_hot(locs_src_handle, &src_off) != 0 ||
        res_require_hot(locs_dst_handle, &dst_off) != 0) {
        res_unpin(locs_src_handle);
        res_unpin(locs_dst_handle);
        return ADAMAH_ERR_INVALID;
    }

    VkDescriptorSet ds = alloc_desc_set(&ctx.softmax_pipe);
    if (ds == VK_NULL_HANDLE) return ADAMAH_ERR_VULKAN;

    VkDescriptorBufferInfo buf_infos[3] = {
        { .buffer = m->buf, .range = VK_WHOLE_SIZE },
        { .buffer = ctx.hot_pool->buf, .offset = src_off, .range = src_res->size_bytes },
        { .buffer = ctx.hot_pool->buf, .offset = dst_off, .range = dst_res->size_bytes }
    };
    VkWriteDescriptorSet writes[3];
    for (int i = 0; i < 3; i++) {
        writes[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = ds, .dstBinding = i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &buf_infos[i] };
    }
    vkUpdateDescriptorSets(ctx.device, 3, writes, 0, NULL);

    uint32_t push[2] = { n_rows, row_size };

    cmd_begin();
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.softmax_pipe.pipeline);
    vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.softmax_pipe.pipe_layout,
                            0, 1, &ds, 0, NULL);
    vkCmdPushConstants(ctx.cmd, ctx.softmax_pipe.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, push);
    vkCmdDispatch(ctx.cmd, n_rows, 1, 1);
    cmd_barrier_after_dispatch();
    cmd_submit();
    res_unpin(locs_src_handle);
    res_unpin(locs_dst_handle);

    return ADAMAH_OK;
}

// ============================================
// LayerNorm: fused mean-var-normalize-scale-shift
// ============================================
int map_layernorm(uint32_t map_id,
                  const uint32_t* locs_src, const uint32_t* locs_dst,
                  const uint32_t* locs_gamma, const uint32_t* locs_beta,
                  uint32_t n_rows, uint32_t dim, float eps) {
    // Legacy path removed: use map_layernorm_dev with cached locs.
    (void)map_id;
    (void)locs_src;
    (void)locs_dst;
    (void)locs_gamma;
    (void)locs_beta;
    (void)n_rows;
    (void)dim;
    (void)eps;
    return ADAMAH_ERR_INVALID;
}

int map_layernorm_dev(uint32_t map_id,
                      uint32_t locs_src_handle, uint32_t locs_dst_handle,
                      uint32_t locs_gamma_handle, uint32_t locs_beta_handle,
                      uint32_t n_rows, uint32_t dim, float eps) {
    if (map_id >= MAX_MAPS || !ctx.maps[map_id].active) return ADAMAH_ERR_INVALID;
    if (n_rows == 0) return ADAMAH_OK;
    if (!ctx.layernorm_pipe.pipeline && init_pipelines() != 0) return ADAMAH_ERR_VULKAN;
    if (!ctx.layernorm_pipe.pipeline) return ADAMAH_ERR_INVALID;

    Map* m = &ctx.maps[map_id];
    ResEntry* src_res = res_get(locs_src_handle);
    ResEntry* dst_res = res_get(locs_dst_handle);
    ResEntry* gamma_res = res_get(locs_gamma_handle);
    ResEntry* beta_res = res_get(locs_beta_handle);
    if (!src_res || !dst_res || !gamma_res || !beta_res) return ADAMAH_ERR_INVALID;
    if ((VkDeviceSize)n_rows * 4 > src_res->size_bytes) return ADAMAH_ERR_INVALID;
    if ((VkDeviceSize)n_rows * 4 > dst_res->size_bytes) return ADAMAH_ERR_INVALID;
    if ((VkDeviceSize)n_rows * 4 > gamma_res->size_bytes) return ADAMAH_ERR_INVALID;
    if ((VkDeviceSize)n_rows * 4 > beta_res->size_bytes) return ADAMAH_ERR_INVALID;
    res_pin(locs_src_handle);
    res_pin(locs_dst_handle);
    res_pin(locs_gamma_handle);
    res_pin(locs_beta_handle);
    VkDeviceSize src_off = 0;
    VkDeviceSize dst_off = 0;
    VkDeviceSize gamma_off = 0;
    VkDeviceSize beta_off = 0;
    if (res_require_hot(locs_src_handle, &src_off) != 0 ||
        res_require_hot(locs_dst_handle, &dst_off) != 0 ||
        res_require_hot(locs_gamma_handle, &gamma_off) != 0 ||
        res_require_hot(locs_beta_handle, &beta_off) != 0) {
        res_unpin(locs_src_handle);
        res_unpin(locs_dst_handle);
        res_unpin(locs_gamma_handle);
        res_unpin(locs_beta_handle);
        return ADAMAH_ERR_INVALID;
    }

    VkDescriptorSet ds = alloc_desc_set(&ctx.layernorm_pipe);
    if (ds == VK_NULL_HANDLE) return ADAMAH_ERR_VULKAN;

    VkDescriptorBufferInfo buf_infos[5] = {
        { .buffer = m->buf, .range = VK_WHOLE_SIZE },
        { .buffer = ctx.hot_pool->buf, .offset = src_off, .range = src_res->size_bytes },
        { .buffer = ctx.hot_pool->buf, .offset = dst_off, .range = dst_res->size_bytes },
        { .buffer = ctx.hot_pool->buf, .offset = gamma_off, .range = gamma_res->size_bytes },
        { .buffer = ctx.hot_pool->buf, .offset = beta_off, .range = beta_res->size_bytes }
    };
    VkWriteDescriptorSet writes[5];
    for (int i = 0; i < 5; i++) {
        writes[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = ds, .dstBinding = i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &buf_infos[i] };
    }
    vkUpdateDescriptorSets(ctx.device, 5, writes, 0, NULL);

    uint32_t push[3];
    push[0] = n_rows;
    push[1] = dim;
    memcpy(&push[2], &eps, 4);

    cmd_begin();
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.layernorm_pipe.pipeline);
    vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.layernorm_pipe.pipe_layout,
                            0, 1, &ds, 0, NULL);
    vkCmdPushConstants(ctx.cmd, ctx.layernorm_pipe.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, push);
    vkCmdDispatch(ctx.cmd, n_rows, 1, 1);
    cmd_barrier_after_dispatch();
    cmd_submit();
    res_unpin(locs_src_handle);
    res_unpin(locs_dst_handle);
    res_unpin(locs_gamma_handle);
    res_unpin(locs_beta_handle);

    return ADAMAH_OK;
}

