# Cooperative groups: grid-wide synchronization example

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

# --- begin:cooperative_reduce ---
function cooperative_reduce!(output, input, partial)
    tid = threadIdx().x
    bid = blockIdx().x
    i = (bid - 1) * blockDim().x + tid
    stride = blockDim().x * gridDim().x

    # Phase 1: each block reduces its portion
    val = 0.0f0
    while i <= length(input)
        val += @inbounds input[i]
        i += stride
    end
    val = warp_reduce_sum(val)

    # Warp leaders write partial sums
    lane = (tid - 1) % UInt32(32) + UInt32(1)
    warp_id = (tid - 1) ÷ 32 + 1
    sdata = CuDynamicSharedArray(Float32, 32)
    if lane == UInt32(1)
        @inbounds sdata[warp_id] = val
    end
    sync_threads()

    # Block-level final reduction
    val = tid <= blockDim().x ÷ 32 ? @inbounds(sdata[tid]) : 0.0f0
    if warp_id == 1
        val = warp_reduce_sum(val)
    end
    if tid == 1
        @inbounds partial[bid] = val
    end

    # Grid-wide synchronization: wait for ALL blocks
    grid = CG.this_grid()
    CG.sync(grid)

    # Phase 2: thread 1 of block 1 sums all per-block partials.
    # A sequential loop over at most gridDim().x values is
    # negligible compared to phase 1 and trivially correct.
    if bid == 1 && tid == 1
        total = 0.0f0
        for j in 1:gridDim().x
            total += @inbounds partial[j]
        end
        output[1] = total
    end
    return nothing
end

# A cooperative launch must fit all blocks resident at once. One block is
# valid on every supported device; larger grids require a device-specific
# occupancy calculation.
N = 10_000_000
input = CUDA.rand(Float32, N)
partial = CuArray{Float32}(undef, 256)
output = CuArray{Float32}(undef, 1)

@cuda(cooperative=true, threads=256, blocks=1,
      shmem=32*sizeof(Float32),
      cooperative_reduce!(output, input, partial))
# --- end:cooperative_reduce ---
