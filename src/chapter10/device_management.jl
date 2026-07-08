# Device management for multi-GPU programming

using CUDA

# --- begin:enumerate_gpus ---
using CUDA

# Enumerate all available GPUs
for dev in CUDA.devices()
    println("Device $(CUDA.deviceid(dev)): $(CUDA.name(dev))")
    println("  Memory: $(CUDA.totalmem(dev) / 1024^3) GB")
    println("  Compute capability: $(CUDA.capability(dev))")
    println("  Multiprocessors: $(CUDA.attribute(dev,
              CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))")
end

# Count available GPUs
ngpus = length(CUDA.devices())
println("\nTotal GPUs: $ngpus")
# --- end:enumerate_gpus ---

# --- begin:select_device ---
# Query the currently active device
current = CUDA.device()
println("Active device: $(CUDA.name(current))")

# Switch to a specific device (0-based integer or CuDevice)
CUDA.device!(1)

# Do-block syntax: temporarily switch, then restore
CUDA.device!(0) do
    a = CUDA.rand(Float32, 1000)    # Allocated on GPU 0
    # ... computations on GPU 0 ...
end
# Automatically restored to previous device
# --- end:select_device ---

# --- begin:task_local_state ---
using CUDA

# Two concurrent tasks, each on a different GPU
@sync begin
    @async begin
        CUDA.device!(0)                    # Task 1 -> GPU 0
        a = CUDA.rand(Float32, 10_000)
        b = a .* 2.0f0
        CUDA.synchronize()
        println("GPU 0 done: sum = $(sum(b))")
    end
    @async begin
        CUDA.device!(1)                    # Task 2 -> GPU 1
        c = CUDA.rand(Float32, 10_000)
        d = c .+ 1.0f0
        CUDA.synchronize()
        println("GPU 1 done: sum = $(sum(d))")
    end
end
# --- end:task_local_state ---
