/*
 * ADAMAH v4.0 - Map-Centric GPU Compute
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

// Error codes
#define ADAMAH_OK           0
#define ADAMAH_ERR_VULKAN  -1
#define ADAMAH_ERR_MEMORY  -2
#define ADAMAH_ERR_INVALID -3
#define ADAMAH_ERR_NOT_FOUND -4

#define MAX_MAPS 16
#define MAX_BUFS 64
#define LOCAL_SIZE 256

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
    uint32_t size;      // In elements
    uint32_t elem_size; // Bytes per element
    int device_local;   // 1 = VRAM, 0 = HOST_VISIBLE
} GpuBuf;

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
    
    Map maps[MAX_MAPS];
    GpuBuf bufs[MAX_BUFS];
    int buf_count;
    
    Pipeline unary_pipe;
    Pipeline binary_pipe;
    Pipeline matmul_pipe;
    Pipeline reduce_pipe;
    Pipeline broadcast_pipe;
    
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

static int create_buffer(VkBuffer* buf, VkDeviceMemory* mem, VkDeviceSize size, 
                         VkBufferUsageFlags usage, int device_local) {
    VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size, .usage = usage };
    if (vkCreateBuffer(ctx.device, &bci, NULL, buf) != VK_SUCCESS) return -1;
    
    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(ctx.device, *buf, &reqs);
    
    uint32_t mem_type = device_local ? 
        find_device_local(reqs.memoryTypeBits) :
        find_mem_type(reqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = reqs.size, .memoryTypeIndex = mem_type };
    if (vkAllocateMemory(ctx.device, &mai, NULL, mem) != VK_SUCCESS) {
        vkDestroyBuffer(ctx.device, *buf, NULL);
        return -1;
    }
    vkBindBufferMemory(ctx.device, *buf, *mem, 0);
    return 0;
}

static void cmd_begin(void) {
    vkWaitForFences(ctx.device, 1, &ctx.fence, VK_TRUE, UINT64_MAX);
    vkResetFences(ctx.device, 1, &ctx.fence);
    vkResetCommandBuffer(ctx.cmd, 0);
    VkCommandBufferBeginInfo cbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    vkBeginCommandBuffer(ctx.cmd, &cbi);
}

static void cmd_submit(void) {
    vkEndCommandBuffer(ctx.cmd);
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &ctx.cmd };
    vkQueueSubmit(ctx.queue, 1, &si, ctx.fence);
    vkWaitForFences(ctx.device, 1, &ctx.fence, VK_TRUE, UINT64_MAX);
}

// ============================================
// Init / Shutdown
// ============================================

int adamah_init(void) {
    if (ctx.initialized) return ADAMAH_OK;
    
    // Create instance
    VkApplicationInfo ai = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "ADAMAH", .apiVersion = VK_API_VERSION_1_0 };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &ai };
    if (vkCreateInstance(&ici, NULL, &ctx.instance) != VK_SUCCESS) return ADAMAH_ERR_VULKAN;
    
    // Get physical device
    uint32_t dc = 0;
    vkEnumeratePhysicalDevices(ctx.instance, &dc, NULL);
    if (!dc) return ADAMAH_ERR_VULKAN;
    VkPhysicalDevice devs[8];
    vkEnumeratePhysicalDevices(ctx.instance, &dc, devs);
    ctx.phys = devs[0];
    
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(ctx.phys, &props);
    printf("ADAMAH v4: %s\n", props.deviceName);
    
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
    
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT };
    vkCreateFence(ctx.device, &fci, NULL, &ctx.fence);
    
    ctx.initialized = 1;
    return ADAMAH_OK;
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
    
    // Staging buffer (HOST_VISIBLE)
    if (create_buffer(&m->staging, &m->staging_mem, m->total_bytes, usage, 0) != 0) {
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

// Scatter: write data to map at locations
// locs: array of pack indices (uint32)
// data: packed data (n_locs * pack_size * word_size bytes)
int mscatter(uint32_t map_id, const uint32_t* locs, const void* data, uint32_t n_locs) {
    if (map_id >= MAX_MAPS || !ctx.maps[map_id].active) return ADAMAH_ERR_INVALID;
    Map* m = &ctx.maps[map_id];
    uint32_t pack_bytes = m->word_size * m->pack_size;
    
    // Copy to staging at correct offsets
    const char* src = (const char*)data;
    char* staging = (char*)m->staging_ptr;
    for (uint32_t i = 0; i < n_locs; i++) {
        uint32_t loc = locs[i];
        if (loc >= m->n_packs) continue;
        memcpy(staging + loc * pack_bytes, src + i * pack_bytes, pack_bytes);
    }
    
    // Upload to GPU
    cmd_begin();
    VkBufferCopy copy = { .size = m->total_bytes };
    vkCmdCopyBuffer(ctx.cmd, m->staging, m->buf, 1, &copy);
    cmd_submit();
    
    return ADAMAH_OK;
}

// Gather: read data from map at locations
int mgather(uint32_t map_id, const uint32_t* locs, void* data, uint32_t n_locs) {
    if (map_id >= MAX_MAPS || !ctx.maps[map_id].active) return ADAMAH_ERR_INVALID;
    Map* m = &ctx.maps[map_id];
    uint32_t pack_bytes = m->word_size * m->pack_size;
    
    // Download from GPU
    cmd_begin();
    VkBufferCopy copy = { .size = m->total_bytes };
    vkCmdCopyBuffer(ctx.cmd, m->buf, m->staging, 1, &copy);
    cmd_submit();
    
    // Copy from staging
    char* dst = (char*)data;
    const char* staging = (const char*)m->staging_ptr;
    for (uint32_t i = 0; i < n_locs; i++) {
        uint32_t loc = locs[i];
        if (loc >= m->n_packs) continue;
        memcpy(dst + i * pack_bytes, staging + loc * pack_bytes, pack_bytes);
    }
    
    return ADAMAH_OK;
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
    fread(&ws, 4, 1, f);
    fread(&ps, 4, 1, f);
    fread(&np, 4, 1, f);
    
    int ret = map_init(id, ws, ps, np);
    if (ret != ADAMAH_OK) { fclose(f); return ret; }
    
    Map* m = &ctx.maps[id];
    fread(m->staging_ptr, 1, m->total_bytes, f);
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
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    uint32_t* code = malloc(*size);
    fread(code, 1, *size, f);
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
    VkDescriptorSetLayoutBinding bindings[4];
    for (int i = 0; i < num_bindings; i++) {
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
    VkDescriptorPoolSize pool_size = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = num_bindings };
    VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &pool_size };
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
    
    // Broadcast: op, n_locs, pack_size = 12 bytes, 4 bindings
    if (create_pipeline(&ctx.broadcast_pipe, "map_broadcast.spv", 4, 12) != 0) {
        fprintf(stderr, "ADAMAH: Warning - broadcast shader not found\n");
    }
    
    return ADAMAH_OK;
}

// ============================================
// Internal GPU buffers for locations
// ============================================

static GpuBuf* get_or_create_buf(const char* name, uint32_t n_elems, uint32_t elem_size) {
    // Find existing
    for (int i = 0; i < ctx.buf_count; i++) {
        if (strcmp(ctx.bufs[i].name, name) == 0) {
            if (ctx.bufs[i].size >= n_elems) return &ctx.bufs[i];
            // Need to resize - destroy old
            if (ctx.bufs[i].ptr) vkUnmapMemory(ctx.device, ctx.bufs[i].mem);
            vkDestroyBuffer(ctx.device, ctx.bufs[i].buf, NULL);
            vkFreeMemory(ctx.device, ctx.bufs[i].mem, NULL);
            ctx.bufs[i].buf = VK_NULL_HANDLE;
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
    b->size = n_elems;
    b->elem_size = elem_size;
    
    VkDeviceSize bytes = n_elems * elem_size;
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    
    if (create_buffer(&b->buf, &b->mem, bytes, usage, 0) != 0) return NULL;
    vkMapMemory(ctx.device, b->mem, 0, bytes, 0, &b->ptr);
    
    return b;
}

// ============================================
// Map Operations (Pure GPU)
// ============================================

int map_op1(uint32_t map_id, uint32_t op, const uint32_t* locs_src, const uint32_t* locs_dst, uint32_t n) {
    if (map_id >= MAX_MAPS || !ctx.maps[map_id].active) return ADAMAH_ERR_INVALID;
    if (!ctx.unary_pipe.pipeline && init_pipelines() != 0) return ADAMAH_ERR_VULKAN;
    
    Map* m = &ctx.maps[map_id];
    
    // Upload locations to GPU buffers
    GpuBuf* src_buf = get_or_create_buf("_locs_src", n, 4);
    GpuBuf* dst_buf = get_or_create_buf("_locs_dst", n, 4);
    if (!src_buf || !dst_buf) return ADAMAH_ERR_MEMORY;
    
    memcpy(src_buf->ptr, locs_src, n * 4);
    memcpy(dst_buf->ptr, locs_dst, n * 4);
    
    // Update descriptor set
    VkDescriptorBufferInfo buf_infos[3] = {
        { .buffer = m->buf, .range = VK_WHOLE_SIZE },
        { .buffer = src_buf->buf, .range = VK_WHOLE_SIZE },
        { .buffer = dst_buf->buf, .range = VK_WHOLE_SIZE }
    };
    VkWriteDescriptorSet writes[3];
    for (int i = 0; i < 3; i++) {
        writes[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = ctx.unary_pipe.desc_set, .dstBinding = i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &buf_infos[i] };
    }
    vkUpdateDescriptorSets(ctx.device, 3, writes, 0, NULL);
    
    // Push constants
    uint32_t push[3] = { op, n, m->pack_size };
    uint32_t total_threads = n * m->pack_size;
    
    // Dispatch
    cmd_begin();
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.unary_pipe.pipeline);
    vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.unary_pipe.pipe_layout,
                            0, 1, &ctx.unary_pipe.desc_set, 0, NULL);
    vkCmdPushConstants(ctx.cmd, ctx.unary_pipe.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, push);
    vkCmdDispatch(ctx.cmd, (total_threads + 255) / 256, 1, 1);
    cmd_submit();
    
    return ADAMAH_OK;
}

int map_op2(uint32_t map_id, uint32_t op, 
            const uint32_t* locs_a, const uint32_t* locs_b, const uint32_t* locs_dst, uint32_t n) {
    if (map_id >= MAX_MAPS || !ctx.maps[map_id].active) return ADAMAH_ERR_INVALID;
    if (!ctx.binary_pipe.pipeline && init_pipelines() != 0) return ADAMAH_ERR_VULKAN;
    
    Map* m = &ctx.maps[map_id];
    
    // Upload locations
    GpuBuf* a_buf = get_or_create_buf("_locs_a", n, 4);
    GpuBuf* b_buf = get_or_create_buf("_locs_b", n, 4);
    GpuBuf* dst_buf = get_or_create_buf("_locs_dst2", n, 4);
    if (!a_buf || !b_buf || !dst_buf) return ADAMAH_ERR_MEMORY;
    
    memcpy(a_buf->ptr, locs_a, n * 4);
    memcpy(b_buf->ptr, locs_b, n * 4);
    memcpy(dst_buf->ptr, locs_dst, n * 4);
    
    // Update descriptor set
    VkDescriptorBufferInfo buf_infos[4] = {
        { .buffer = m->buf, .range = VK_WHOLE_SIZE },
        { .buffer = a_buf->buf, .range = VK_WHOLE_SIZE },
        { .buffer = b_buf->buf, .range = VK_WHOLE_SIZE },
        { .buffer = dst_buf->buf, .range = VK_WHOLE_SIZE }
    };
    VkWriteDescriptorSet writes[4];
    for (int i = 0; i < 4; i++) {
        writes[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = ctx.binary_pipe.desc_set, .dstBinding = i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &buf_infos[i] };
    }
    vkUpdateDescriptorSets(ctx.device, 4, writes, 0, NULL);
    
    // Push constants
    uint32_t push[3] = { op, n, m->pack_size };
    uint32_t total_threads = n * m->pack_size;
    
    // Dispatch
    cmd_begin();
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.binary_pipe.pipeline);
    vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.binary_pipe.pipe_layout,
                            0, 1, &ctx.binary_pipe.desc_set, 0, NULL);
    vkCmdPushConstants(ctx.cmd, ctx.binary_pipe.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, push);
    vkCmdDispatch(ctx.cmd, (total_threads + 255) / 256, 1, 1);
    cmd_submit();
    
    return ADAMAH_OK;
}

// ============================================
// Matrix Multiplication: C = A @ B
// A: (M x K), B: (K x N), C: (M x N)
// ============================================
int map_matmul(uint32_t map_id, 
               const uint32_t* locs_a, const uint32_t* locs_b, const uint32_t* locs_c,
               uint32_t M, uint32_t K, uint32_t N, uint32_t n_ops) {
    if (map_id >= MAX_MAPS || !ctx.maps[map_id].active) return ADAMAH_ERR_INVALID;
    if (!ctx.matmul_pipe.pipeline && init_pipelines() != 0) return ADAMAH_ERR_VULKAN;
    if (!ctx.matmul_pipe.pipeline) return ADAMAH_ERR_INVALID;
    
    Map* m = &ctx.maps[map_id];
    
    // Upload locations
    GpuBuf* a_buf = get_or_create_buf("_matmul_a", n_ops, 4);
    GpuBuf* b_buf = get_or_create_buf("_matmul_b", n_ops, 4);
    GpuBuf* c_buf = get_or_create_buf("_matmul_c", n_ops, 4);
    if (!a_buf || !b_buf || !c_buf) return ADAMAH_ERR_MEMORY;
    
    memcpy(a_buf->ptr, locs_a, n_ops * 4);
    memcpy(b_buf->ptr, locs_b, n_ops * 4);
    memcpy(c_buf->ptr, locs_c, n_ops * 4);
    
    // Update descriptor set
    VkDescriptorBufferInfo buf_infos[4] = {
        { .buffer = m->buf, .range = VK_WHOLE_SIZE },
        { .buffer = a_buf->buf, .range = VK_WHOLE_SIZE },
        { .buffer = b_buf->buf, .range = VK_WHOLE_SIZE },
        { .buffer = c_buf->buf, .range = VK_WHOLE_SIZE }
    };
    VkWriteDescriptorSet writes[4];
    for (int i = 0; i < 4; i++) {
        writes[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = ctx.matmul_pipe.desc_set, .dstBinding = i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &buf_infos[i] };
    }
    vkUpdateDescriptorSets(ctx.device, 4, writes, 0, NULL);
    
    // Push constants: M, K, N, n_ops
    uint32_t push[4] = { M, K, N, n_ops };
    
    // Dispatch: workgroup per (16x16) tile of output, z for each operation
    uint32_t grid_x = (M + 15) / 16;
    uint32_t grid_y = (N + 15) / 16;
    
    cmd_begin();
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.matmul_pipe.pipeline);
    vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.matmul_pipe.pipe_layout,
                            0, 1, &ctx.matmul_pipe.desc_set, 0, NULL);
    vkCmdPushConstants(ctx.cmd, ctx.matmul_pipe.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, push);
    vkCmdDispatch(ctx.cmd, grid_x, grid_y, n_ops);
    cmd_submit();
    
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
    if (map_id >= MAX_MAPS || !ctx.maps[map_id].active) return ADAMAH_ERR_INVALID;
    if (!ctx.reduce_pipe.pipeline && init_pipelines() != 0) return ADAMAH_ERR_VULKAN;
    if (!ctx.reduce_pipe.pipeline) return ADAMAH_ERR_INVALID;
    
    Map* m = &ctx.maps[map_id];
    
    GpuBuf* src_buf = get_or_create_buf("_reduce_src", n, 4);
    GpuBuf* dst_buf = get_or_create_buf("_reduce_dst", n, 4);
    if (!src_buf || !dst_buf) return ADAMAH_ERR_MEMORY;
    
    memcpy(src_buf->ptr, locs_src, n * 4);
    memcpy(dst_buf->ptr, locs_dst, n * 4);
    
    VkDescriptorBufferInfo buf_infos[3] = {
        { .buffer = m->buf, .range = VK_WHOLE_SIZE },
        { .buffer = src_buf->buf, .range = VK_WHOLE_SIZE },
        { .buffer = dst_buf->buf, .range = VK_WHOLE_SIZE }
    };
    VkWriteDescriptorSet writes[3];
    for (int i = 0; i < 3; i++) {
        writes[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = ctx.reduce_pipe.desc_set, .dstBinding = i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &buf_infos[i] };
    }
    vkUpdateDescriptorSets(ctx.device, 3, writes, 0, NULL);
    
    // Push: op, n_locs, pack_size
    uint32_t push[3] = { op, n, m->pack_size };
    
    // One workgroup per pack (256 threads for reduction)
    cmd_begin();
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.reduce_pipe.pipeline);
    vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.reduce_pipe.pipe_layout,
                            0, 1, &ctx.reduce_pipe.desc_set, 0, NULL);
    vkCmdPushConstants(ctx.cmd, ctx.reduce_pipe.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, push);
    vkCmdDispatch(ctx.cmd, n, 1, 1);
    cmd_submit();
    
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
    if (map_id >= MAX_MAPS || !ctx.maps[map_id].active) return ADAMAH_ERR_INVALID;
    if (!ctx.broadcast_pipe.pipeline && init_pipelines() != 0) return ADAMAH_ERR_VULKAN;
    if (!ctx.broadcast_pipe.pipeline) return ADAMAH_ERR_INVALID;
    
    Map* m = &ctx.maps[map_id];
    
    GpuBuf* src_buf = get_or_create_buf("_bcast_src", n, 4);
    GpuBuf* scalar_buf = get_or_create_buf("_bcast_scalar", n, 4);
    GpuBuf* dst_buf = get_or_create_buf("_bcast_dst", n, 4);
    if (!src_buf || !scalar_buf || !dst_buf) return ADAMAH_ERR_MEMORY;
    
    memcpy(src_buf->ptr, locs_src, n * 4);
    memcpy(scalar_buf->ptr, locs_scalar, n * 4);
    memcpy(dst_buf->ptr, locs_dst, n * 4);
    
    VkDescriptorBufferInfo buf_infos[4] = {
        { .buffer = m->buf, .range = VK_WHOLE_SIZE },
        { .buffer = src_buf->buf, .range = VK_WHOLE_SIZE },
        { .buffer = scalar_buf->buf, .range = VK_WHOLE_SIZE },
        { .buffer = dst_buf->buf, .range = VK_WHOLE_SIZE }
    };
    VkWriteDescriptorSet writes[4];
    for (int i = 0; i < 4; i++) {
        writes[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = ctx.broadcast_pipe.desc_set, .dstBinding = i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &buf_infos[i] };
    }
    vkUpdateDescriptorSets(ctx.device, 4, writes, 0, NULL);
    
    // Push: op, n_locs, pack_size
    uint32_t push[3] = { op, n, m->pack_size };
    uint32_t total_threads = n * m->pack_size;
    
    cmd_begin();
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.broadcast_pipe.pipeline);
    vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.broadcast_pipe.pipe_layout,
                            0, 1, &ctx.broadcast_pipe.desc_set, 0, NULL);
    vkCmdPushConstants(ctx.cmd, ctx.broadcast_pipe.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, push);
    vkCmdDispatch(ctx.cmd, (total_threads + 255) / 256, 1, 1);
    cmd_submit();
    
    return ADAMAH_OK;
}
