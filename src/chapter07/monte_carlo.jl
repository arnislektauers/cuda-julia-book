# Adaptive Monte Carlo integration on the GPU

using CUDA

@inline function warp_reduce_sum(val::Float32)
    mask = 0xffffffff
    val += CUDA.shfl_down_sync(mask, val, 16)
    val += CUDA.shfl_down_sync(mask, val, 8)
    val += CUDA.shfl_down_sync(mask, val, 4)
    val += CUDA.shfl_down_sync(mask, val, 2)
    val += CUDA.shfl_down_sync(mask, val, 1)
    return val
end

# --- begin:monte_carlo_integration ---
function monte_carlo_kernel!(partial_sums, N, a, b)
    sdata = CuDynamicSharedArray(Float32, 32)
    tid = threadIdx().x
    i = (blockIdx().x - 1) * blockDim().x + tid
    stride = blockDim().x * gridDim().x

    # Each thread accumulates samples via grid-stride loop
    local_sum = 0.0f0
    # WARNING: toy LCG for illustration only. Correlated streams and short
    # low-bit periods make it unsuitable for production Monte Carlo; use the
    # device RNG rand(Float32) inside kernels, or CURAND, instead.
    seed = UInt32(i) * UInt32(1103515245) + UInt32(12345)
    while i <= N
        # Generate pseudo-random x in [a, b]
        seed = seed * UInt32(1103515245) + UInt32(12345)
        x = a + (b - a) * Float32(seed) / Float32(typemax(UInt32))
        # Evaluate function: f(x) = sin(x)^2
        local_sum += sin(x)^2
        i += stride
    end

    # Warp reduction
    local_sum = warp_reduce_sum(local_sum)

    # Block reduction via shared memory
    lane = (tid - 1) % 32 + 1
    warp_id = (tid - 1) ÷ 32 + 1
    if lane == 1
        @inbounds sdata[warp_id] = local_sum
    end
    sync_threads()

    local_sum = (tid <= blockDim().x ÷ 32) ? @inbounds(sdata[tid]) : 0.0f0
    if warp_id == 1
        local_sum = warp_reduce_sum(local_sum)
    end

    if tid == 1
        @inbounds partial_sums[blockIdx().x] = local_sum
    end
    return nothing
end

# Driver code
N = 100_000_000
a, b = 0.0f0, Float32(π)
threads = 256
blocks = 256
partial_sums = CuArray{Float32}(undef, blocks)

@cuda(threads=threads, blocks=blocks, shmem=32*sizeof(Float32),
      monte_carlo_kernel!(partial_sums, N, a, b))

# Final reduction on partial sums
result = (b - a) / N * sum(partial_sums)
println("integral of sin^2(x) on [0, pi] = $result (exact: $(pi/2))")
# --- end:monte_carlo_integration ---
