using CUDA

function print_device_info()
    dev = CUDA.device()
    attr = (a) -> CUDA.attribute(dev, a)
    println("Device: $(CUDA.name(dev))")
    println("Compute Capability: $(CUDA.capability(dev))")
    println("  SMs:               ",
            attr(CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
    println("  Max threads/block: ",
            attr(CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK))
    println("  Warp size:         ",
            attr(CUDA.DEVICE_ATTRIBUTE_WARP_SIZE))
    println("  Shared mem/block:  ",
            attr(CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK),
            " bytes")
    println("  L2 cache:          ",
            attr(CUDA.DEVICE_ATTRIBUTE_L2_CACHE_SIZE), " bytes")
    println("  Total memory:      ",
            round(CUDA.totalmem(dev) / 2^30; digits=1), " GiB")
end
