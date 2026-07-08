# Grid-stride reduction with warp shuffle; produces one partial sum per
# block, completed on the host with sum(output) or a second kernel pass

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

# --- begin:full_reduce ---
function full_reduce_sum!(output, input)
    sdata = CuDynamicSharedArray(Float32, 32)
    tid = threadIdx().x
    i = (blockIdx().x - 1) * blockDim().x + tid
    stride = blockDim().x * gridDim().x

    # Grid-stride loop: each thread accumulates multiple elements
    val = 0.0f0
    while i <= length(input)
        val += @inbounds input[i]
        i += stride
    end

    # Warp reduction
    val = warp_reduce_sum(val)

    # Block reduction via shared memory
    lane = (tid - 1) % 32 + 1
    warp_id = (tid - 1) ÷ 32 + 1
    if lane == 1
        @inbounds sdata[warp_id] = val
    end
    sync_threads()

    # First warp reduces the partial sums
    # (assumes blockDim().x is a multiple of 32, i.e. whole warps only)
    val = (tid <= blockDim().x ÷ 32) ? @inbounds(sdata[tid]) : 0.0f0
    if warp_id == 1
        val = warp_reduce_sum(val)
    end

    if tid == 1
        @inbounds output[blockIdx().x] = val
    end
    return nothing
end
# --- end:full_reduce ---
