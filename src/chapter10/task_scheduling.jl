# Task-based GPU scheduling with Dagger.jl

using Dagger, CUDA

# --- begin:dagger_gpu_scope ---
using Dagger, CUDA

# Execute tasks on a specific CUDA GPU
Dagger.with_options(; scope=Dagger.scope(cuda_gpu=1)) do
    DA = rand(Blocks(32, 32), Float32, 128, 128)
    DB = rand(Blocks(32, 32), Float32, 128, 128)
    DC = DA * DB                # Scheduled on GPU 1
    result = collect(DC)        # Fetch to CPU
end

# Distribute tasks across all CUDA GPUs
Dagger.with_options(; scope=Dagger.scope(cuda_gpus=:)) do
    # Tasks automatically load-balanced across all GPUs
end
# --- end:dagger_gpu_scope ---

# --- begin:dagger_ka ---
using KernelAbstractions, Dagger, CUDA

@kernel function saxpy_kernel!(y, α, x)
    i = @index(Global, Linear)
    y[i] = α * x[i] + y[i]
end

# Dagger schedules the KA kernel on an available GPU
arr_x = Dagger.@mutable CUDA.rand(Float32, 10_000)
arr_y = Dagger.@mutable CUDA.rand(Float32, 10_000)

fetch(Dagger.@spawn Dagger.Kernel(saxpy_kernel!)(
    arr_y, 2.0f0, arr_x; ndrange=10_000
))
# --- end:dagger_ka ---
