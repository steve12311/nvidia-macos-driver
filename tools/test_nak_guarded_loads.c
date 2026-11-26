#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vulkan/vulkan.h>

#define CHECK_VK(call) do { \
    VkResult status = (call); \
    if (status != VK_SUCCESS) { \
        fprintf(stderr, "%s: %d\n", #call, status); \
        exit(1); \
    } \
} while (0)

enum { THREADS = 64, GROUPS = 40, COUNT = THREADS * GROUPS };

struct storage {
    VkBuffer buffer;
    VkDeviceMemory memory;
    void *data;
};

static struct storage make_storage(VkDevice device, VkPhysicalDevice physical, VkDeviceSize size)
{
    struct storage out = {0};
    VkBufferCreateInfo info = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT };
    CHECK_VK(vkCreateBuffer(device, &info, NULL, &out.buffer));

    VkPhysicalDeviceMemoryProperties props;
    VkMemoryRequirements req;
    vkGetPhysicalDeviceMemoryProperties(physical, &props);
    vkGetBufferMemoryRequirements(device, out.buffer, &req);
    unsigned index;
    VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (index = 0; index < props.memoryTypeCount; index++) {
        if ((req.memoryTypeBits & (1u << index)) &&
            (props.memoryTypes[index].propertyFlags & required) == required)
            break;
    }
    if (index == props.memoryTypeCount) {
        fprintf(stderr, "No host-coherent storage memory\n");
        exit(1);
    }
    VkMemoryAllocateInfo alloc = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size, .memoryTypeIndex = index };
    CHECK_VK(vkAllocateMemory(device, &alloc, NULL, &out.memory));
    CHECK_VK(vkBindBufferMemory(device, out.buffer, out.memory, 0));
    CHECK_VK(vkMapMemory(device, out.memory, 0, size, 0, &out.data));
    return out;
}

static VkShaderModule load_shader(VkDevice device, const char *path)
{
    FILE *file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END)) {
        perror(path);
        exit(1);
    }
    long size = ftell(file);
    if (size <= 0 || size % 4 || fseek(file, 0, SEEK_SET)) {
        fprintf(stderr, "Invalid SPIR-V size in %s\n", path);
        exit(1);
    }
    uint32_t *code = malloc((size_t)size);
    if (!code || fread(code, 1, (size_t)size, file) != (size_t)size) {
        fprintf(stderr, "Cannot read %s\n", path);
        exit(1);
    }
    fclose(file);
    VkShaderModuleCreateInfo info = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = (size_t)size, .pCode = code };
    VkShaderModule shader;
    CHECK_VK(vkCreateShaderModule(device, &info, NULL, &shader));
    free(code);
    return shader;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s shader.spv\n", argv[0]);
        return 2;
    }
    VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .apiVersion = VK_API_VERSION_1_2 };
    VkInstanceCreateInfo instance_info = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app };
    VkInstance instance;
    CHECK_VK(vkCreateInstance(&instance_info, NULL, &instance));
    VkPhysicalDevice physical;
    uint32_t count = 1;
    VkResult devices = vkEnumeratePhysicalDevices(instance, &count, &physical);
    if (!count || (devices != VK_SUCCESS && devices != VK_INCOMPLETE)) {
        fprintf(stderr, "No Vulkan physical device (%d)\n", devices);
        return 1;
    }
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical, &props);
    printf("Device: %s\n", props.deviceName);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, NULL);
    if (!count)
        return 1;
    VkQueueFamilyProperties queues[count];
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, queues);
    uint32_t family = 0;
    while (family < count && !(queues[family].queueFlags & VK_QUEUE_COMPUTE_BIT))
        family++;
    if (family == count) {
        fprintf(stderr, "No compute queue\n");
        return 1;
    }
    VkPhysicalDevice16BitStorageFeatures storage16 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES };
    VkPhysicalDeviceFeatures2 available = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &storage16 };
    vkGetPhysicalDeviceFeatures2(physical, &available);
    if (!storage16.storageBuffer16BitAccess) {
        fprintf(stderr, "No 16-bit storage buffer support\n");
        return 1;
    }
    storage16 = (VkPhysicalDevice16BitStorageFeatures){
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES, .storageBuffer16BitAccess = VK_TRUE };
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &priority };
    VkDeviceCreateInfo device_info = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &storage16,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &queue_info };
    VkDevice device;
    CHECK_VK(vkCreateDevice(physical, &device_info, NULL, &device));
    VkQueue queue;
    vkGetDeviceQueue(device, family, 0, &queue);

    const VkDeviceSize sizes[] = { sizeof(uint32_t), (COUNT * 8 + 1) * sizeof(uint16_t), COUNT * sizeof(uint32_t) };
    struct storage buffers[3];
    VkDescriptorSetLayoutBinding bindings[3];
    for (unsigned i = 0; i < 3; i++) {
        buffers[i] = make_storage(device, physical, sizes[i]);
        bindings[i] = (VkDescriptorSetLayoutBinding){ .binding = i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
    }
    uint16_t *input = buffers[1].data;
    for (unsigned i = 0; i < COUNT * 8 + 1; i++)
        input[i] = (uint16_t)(i * 91 + 47);

    VkDescriptorSetLayoutCreateInfo set_info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3, .pBindings = bindings };
    VkDescriptorSetLayout set_layout;
    CHECK_VK(vkCreateDescriptorSetLayout(device, &set_info, NULL, &set_layout));
    VkPipelineLayoutCreateInfo layout_info = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &set_layout };
    VkPipelineLayout layout;
    CHECK_VK(vkCreatePipelineLayout(device, &layout_info, NULL, &layout));
    VkShaderModule shader = load_shader(device, argv[1]);
    VkComputePipelineCreateInfo compute_info = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = shader, .pName = "main" }, .layout = layout };
    VkPipeline pipeline;
    CHECK_VK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &compute_info, NULL, &pipeline));

    VkDescriptorPoolSize pool_size = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 3 };
    VkDescriptorPoolCreateInfo pool_info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &pool_size };
    VkDescriptorPool pool;
    CHECK_VK(vkCreateDescriptorPool(device, &pool_info, NULL, &pool));
    VkDescriptorSetAllocateInfo set_alloc = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &set_layout };
    VkDescriptorSet set;
    CHECK_VK(vkAllocateDescriptorSets(device, &set_alloc, &set));
    VkDescriptorBufferInfo descriptors[3];
    VkWriteDescriptorSet writes[3];
    for (unsigned i = 0; i < 3; i++) {
        descriptors[i] = (VkDescriptorBufferInfo){ .buffer = buffers[i].buffer, .range = sizes[i] };
        writes[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set, .dstBinding = i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &descriptors[i] };
    }
    vkUpdateDescriptorSets(device, 3, writes, 0, NULL);

    VkCommandPoolCreateInfo commands_info = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = family };
    VkCommandPool commands;
    CHECK_VK(vkCreateCommandPool(device, &commands_info, NULL, &commands));
    VkCommandBufferAllocateInfo command_alloc = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = commands, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
    VkCommandBuffer command;
    CHECK_VK(vkAllocateCommandBuffers(device, &command_alloc, &command));
    VkCommandBufferBeginInfo begin = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    CHECK_VK(vkBeginCommandBuffer(command, &begin));
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, NULL);
    vkCmdDispatch(command, GROUPS, 1, 1);
    VkMemoryBarrier to_host = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT };
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &to_host, 0, NULL, 0, NULL);
    CHECK_VK(vkEndCommandBuffer(command));
    VkSubmitInfo submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &command };

    const unsigned weights[] = { 3, 5, 7, 11, 13, 17, 19, 23 };
    int result = 0;
    for (uint32_t mode = 0; mode < 3; mode++) {
        *(uint32_t *)buffers[0].data = mode;
        CHECK_VK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
        CHECK_VK(vkQueueWaitIdle(queue));
        const uint32_t *output = buffers[2].data;
        for (unsigned i = 0; i < COUNT; i++) {
            unsigned offset = mode == 2 ? (i & 1) : mode;
            uint32_t expected = 0;
            for (unsigned j = 0; j < 8; j++)
                expected += input[i * 8 + offset + j] * weights[j];
            if (output[i] != expected) {
                fprintf(stderr, "mode %u, lane %u: expected %u, got %u\n", mode, i, expected, output[i]);
                result = 1;
                break;
            }
        }
    }
    if (!result)
        printf("All u16 sums match for aligned, unaligned and mixed lanes\n");
    vkDestroyCommandPool(device, commands, NULL);
    vkDestroyDescriptorPool(device, pool, NULL);
    vkDestroyPipeline(device, pipeline, NULL);
    vkDestroyShaderModule(device, shader, NULL);
    vkDestroyPipelineLayout(device, layout, NULL);
    vkDestroyDescriptorSetLayout(device, set_layout, NULL);
    for (unsigned i = 0; i < 3; i++) {
        vkUnmapMemory(device, buffers[i].memory);
        vkDestroyBuffer(device, buffers[i].buffer, NULL);
        vkFreeMemory(device, buffers[i].memory, NULL);
    }
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    return result;
}
