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

# ---------------------------------------------------------------------------
# Driver, outside the tagged region so the book is unaffected. Sums a vector of
# ones, where the expected total is exactly representable in Float32 and any
# lost or double-counted element shows up as an integer discrepancy.
#
# Two details the kernel depends on and a careless caller would get wrong: the
# shared array is dynamic, so the launch must pass shmem for 32 Float32s, and
# the block reduction assumes whole warps, so threads must be a multiple of 32.
let N = 1 << 20, threads = 256, blocks = 64
    input = CUDA.ones(Float32, N)
    output = CUDA.zeros(Float32, blocks)
    @cuda threads=threads blocks=blocks shmem=32*sizeof(Float32) full_reduce_sum!(output, input)
    CUDA.synchronize()
    total = sum(Array(output))
    @assert total == Float32(N) "grid-stride reduction: got $total, expected $N"
    println("full_reduce_sum!: $(Int(total)) == $N CORRECT")
end
