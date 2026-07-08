# Warp-level programming: shuffle reductions and scans

using CUDA

# --- begin:warp_reduce_sum ---
@inline function warp_reduce_sum(val::Float32)
    mask = 0xffffffff
    val += CUDA.shfl_down_sync(mask, val, 16)
    val += CUDA.shfl_down_sync(mask, val, 8)
    val += CUDA.shfl_down_sync(mask, val, 4)
    val += CUDA.shfl_down_sync(mask, val, 2)
    val += CUDA.shfl_down_sync(mask, val, 1)
    return val
end
# --- end:warp_reduce_sum ---

# --- begin:warp_block_reduce ---
function block_reduce_sum!(output, input)
    sdata = CuDynamicSharedArray(Float32, 32)  # one slot per warp
    tid = threadIdx().x
    i = (blockIdx().x - 1) * blockDim().x + tid

    # Each thread loads one element
    val = (i <= length(input)) ? @inbounds(input[i]) : 0.0f0

    # Warp-level reduction (no shared memory needed)
    val = warp_reduce_sum(val)

    # Warp leaders write to shared memory
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

    # Thread 1 writes the block result
    if tid == 1
        @inbounds output[blockIdx().x] = val
    end
    return nothing
end
# --- end:warp_block_reduce ---

# --- begin:warp_scan ---
@inline function warp_inclusive_scan(val::Float32)
    mask = 0xffffffff
    lane = (threadIdx().x - Int32(1)) % Int32(32) + Int32(1)
    n = CUDA.shfl_up_sync(mask, val, 1)
    val += ifelse(lane > Int32(1), n, 0.0f0)
    n = CUDA.shfl_up_sync(mask, val, 2)
    val += ifelse(lane > Int32(2), n, 0.0f0)
    n = CUDA.shfl_up_sync(mask, val, 4)
    val += ifelse(lane > Int32(4), n, 0.0f0)
    n = CUDA.shfl_up_sync(mask, val, 8)
    val += ifelse(lane > Int32(8), n, 0.0f0)
    n = CUDA.shfl_up_sync(mask, val, 16)
    val += ifelse(lane > Int32(16), n, 0.0f0)
    return val
end
# --- end:warp_scan ---
