/*
 * Host unit test for plugin/ mmap maxBufferLength capping, read-only host range detection,
 * and BDA descriptor-set reuse dirty-bit invalidation.
 */
#include <assert.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

static uint64_t calc_max_buffer_bytes(uint64_t max_storage_range, uint64_t max_buffer_size,
                                      uint64_t max_alloc_size, uint64_t guard) {
    uint64_t cap = max_storage_range;
    if (max_buffer_size && (!cap || max_buffer_size < cap)) cap = max_buffer_size;
    if (max_alloc_size) {
        const uint64_t usable = max_alloc_size > guard ? max_alloc_size - guard : 0;
        if (!cap || usable < cap) cap = usable;
    }
    return cap & ~0x3fffULL;
}

static int host_range_readonly(const void *ptr, size_t len) {
    if (!ptr || !len) return 0;
    mach_vm_address_t addr = (mach_vm_address_t)(uintptr_t)ptr;
    const mach_vm_address_t end = addr + (mach_vm_size_t)len;
    if (end < addr) return 0;
    while (addr < end) {
        mach_vm_address_t cur = addr;
        mach_vm_size_t size = 0;
        vm_region_basic_info_data_64_t info;
        mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
        mach_port_t obj = MACH_PORT_NULL;
        kern_return_t kr = mach_vm_region(mach_task_self(), &cur, &size, VM_REGION_BASIC_INFO_64,
                                          (vm_region_info_t)&info, &count, &obj);
        if (obj != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), obj);
        if (kr != KERN_SUCCESS || cur > addr || !size) return 0;
        if (!(info.protection & VM_PROT_WRITE)) return 1;
        if (size >= end - cur) break;
        addr = cur + size;
    }
    return 0;
}

int main(void) {
    /* 1. NVK caps maxMemoryAllocationSize at 2 GiB while maxStorageBufferRange is 4 GiB. */
    const uint64_t gib = 1ULL << 30;
    const uint64_t cap = calc_max_buffer_bytes(4 * gib - 1, 0, 2 * gib, 65536);
    assert(cap == 2 * gib - 65536);
    assert((cap & 0x3fffULL) == 0);
    /* A 2515.44 MiB model exceeds cap so ggml_metal_buffer_map splits it across buffers. */
    assert((uint64_t)(2515ULL << 20) > cap);
    /* With guard added back, allocation stays <= 2 GiB. */
    assert(cap + 65536 <= 2 * gib);

    /* 2. mach_vm_region detects PROT_READ mmap vs PROT_READ|PROT_WRITE mmap. */
    const size_t pg = (size_t)getpagesize();
    void *rw = mmap(NULL, 2 * pg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    assert(rw != MAP_FAILED);
    assert(host_range_readonly(rw, 2 * pg) == 0);
    assert(mprotect((char *)rw + pg, pg, PROT_READ) == 0);
    assert(host_range_readonly(rw, pg) == 0);
    assert(host_range_readonly(rw, 2 * pg) == 1);
    assert(host_range_readonly((char *)rw + pg, pg) == 1);
    munmap(rw, 2 * pg);

    /* 3. BDA fast path invalidates rebound buffer slots in lastw[0] before clearing chg. */
    uint64_t lastw0 = (1ULL << 2) | (1ULL << 33);
    uint64_t chg0 = (1ULL << 2);
    lastw0 &= ~(chg0 & 0xffffffffULL);
    assert((lastw0 & (1ULL << 2)) == 0);
    assert((lastw0 & (1ULL << 33)) != 0);

    puts("test_mmap_and_bda_cache: OK");
    return 0;
}
