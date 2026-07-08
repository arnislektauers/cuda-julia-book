# Multi-GPU Monte Carlo estimation of π
# Builds on the single-GPU kernel from monte_carlo.jl

using CUDA

# Reuse the kernel from single-GPU example
function mc_pi_kernel!(count, N)
    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    local_count = Int32(0)

    i = tid
    while i <= N
        x = rand(Float32)
        y = rand(Float32)
        if x * x + y * y <= 1.0f0
            local_count += Int32(1)
        end
        i += stride
    end

    CUDA.@atomic count[1] += local_count
    return nothing
end

# --- begin:multi_gpu_mc ---
using CUDA

function multi_gpu_monte_carlo_pi(N_per_gpu::Int)
    ngpus = length(CUDA.devices())
    results = zeros(Float64, ngpus)

    @sync for (idx, dev) in enumerate(CUDA.devices())
        @async begin
            CUDA.device!(dev)

            d_count = CUDA.zeros(Int32, 1)
            threads = 256
            sms = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            blocks = min(cld(N_per_gpu, threads), 32 * sms)
            @cuda threads=threads blocks=blocks mc_pi_kernel!(
                d_count, N_per_gpu)
            CUDA.synchronize()

            results[idx] = 4.0 * Float64(Array(d_count)[1]) / Float64(N_per_gpu)
        end
    end

    return sum(results) / ngpus
end
# --- end:multi_gpu_mc ---
