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

# ---------------------------------------------------------------------------
# Driver, outside the tagged region so the book is unaffected.
#
# On a single-GPU host this exercises the one-device path: the loop runs once,
# ngpus == 1, and the average is over a single result. The multi-device
# behaviour -- work actually split across devices -- can only be checked where
# there is more than one, so what this asserts is the estimator, not the split.
#
# The tolerance is derived, not guessed: the estimate has standard error
# 4*sqrt(p(1-p)/N) with p = pi/4, which is 5.2e-4 at N = 1e7. 0.01 is ~19
# sigma, so a passing run is not luck and a failing one is a real defect.
let N = 10_000_000
    est = multi_gpu_monte_carlo_pi(N)
    ngpus = length(CUDA.devices())
    @assert isapprox(est, pi; atol = 0.01) "estimate $est is not pi within 0.01"
    println("multi_gpu_monte_carlo_pi: $(round(est, digits=5)) over $ngpus GPU(s) CORRECT",
            ngpus == 1 ? " (single-device path only)" : "")
end
