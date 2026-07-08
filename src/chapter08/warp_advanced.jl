# Advanced warp-level programming: generic reduction and stream compaction

using CUDA

# --- begin:warp_reduce_generic ---
@inline function warp_reduce(op, val::T) where T
    mask = 0xffffffff
    val = op(val, CUDA.shfl_down_sync(mask, val, 16))
    val = op(val, CUDA.shfl_down_sync(mask, val, 8))
    val = op(val, CUDA.shfl_down_sync(mask, val, 4))
    val = op(val, CUDA.shfl_down_sync(mask, val, 2))
    val = op(val, CUDA.shfl_down_sync(mask, val, 1))
    return val
end

# Usage: warp_reduce(min, val), warp_reduce(max, val), warp_reduce(+, val)
# --- end:warp_reduce_generic ---

# --- begin:stream_compaction ---
function compact_kernel!(output, input, count, threshold)
    tid = threadIdx().x
    i = (blockIdx().x - 1) * blockDim().x + tid

    # Each thread evaluates the predicate
    pred = i <= length(input) && @inbounds(input[i]) > threshold
    mask = 0xffffffff

    # Ballot: get bitmask of which lanes satisfy the predicate
    ballot = CUDA.vote_ballot_sync(mask, pred)

    # Count how many lanes before this one satisfy the predicate
    lane = (tid - 1) % UInt32(32)
    # Mask off lanes >= current lane
    prior_mask = (UInt32(1) << lane) - UInt32(1)
    local_offset = count_ones(ballot & prior_mask)

    # Warp leader atomically reserves space in output
    warp_count = count_ones(ballot)
    base = Int32(0)
    if lane == UInt32(0) && warp_count > 0
        base = CUDA.atomic_add!(pointer(count, 1), Int32(warp_count))
    end
    # Broadcast base offset from the warp leader (lane 1, 1-based) to all lanes
    base = CUDA.shfl_sync(mask, base, 1)

    # Write compacted output
    if pred
        @inbounds output[base + local_offset + 1] = input[i]
    end
    return nothing
end
# --- end:stream_compaction ---
