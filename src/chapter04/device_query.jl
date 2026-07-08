# Querying GPU device capabilities

using CUDA

# --- begin:gpu_query ---
using CUDA

dev = device()

println("Device: ", name(dev))
println("Compute capability: ", capability(dev))
println("Total global memory: ", totalmem(dev) / 1024^3, " GB")

attr = CUDA.attribute
println("Max threads per block: ", 
    attr(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK))
println("Max shared memory per block: ",
    attr(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK) / 1024, " KB")
println("Warp size: ", 
    attr(dev, CUDA.DEVICE_ATTRIBUTE_WARP_SIZE))
println("Number of SMs: ", 
    attr(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
println("Max registers per block: ",
    attr(dev, CUDA.DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK))
# --- end:gpu_query ---
