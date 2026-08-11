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

# ---------------------------------------------------------------------------
# Driver, outside the tagged region so the book is unaffected.
#
# Compaction writes in whatever order warps reserve their slots, so the output
# is a permutation of the expected values, not a sequence: sort before
# comparing. The count is the stricter check -- a ballot or prefix-count that
# is off by one produces a plausible list with the wrong length, or silently
# overlaps two warps' slots.
let N = 1 << 16, threshold = 0.5f0
    input = CUDA.rand(Float32, N)
    h = Array(input)
    want = sort(h[h .> threshold])

    output = CUDA.zeros(Float32, N)
    count = CUDA.zeros(Int32, 1)
    @cuda threads=256 blocks=cld(N, 256) compact_kernel!(output, input, count, threshold)
    CUDA.synchronize()
    n = CUDA.@allowscalar Int(count[1])
    @assert n == length(want) "compacted $n elements, expected $(length(want))"
    @assert sort(Array(output)[1:n]) == want "compacted values differ from the predicate's"

    # warp_reduce is generic over the operator, which the compaction kernel
    # does not exercise; check all three the comment advertises.
    function warp_reduce_probe!(out, x)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        v = @inbounds x[i]
        s = warp_reduce(+, v)
        mn = warp_reduce(min, v)
        mx = warp_reduce(max, v)
        if (threadIdx().x - 1) % 32 == 0
            w = (i - 1) ÷ 32 + 1
            @inbounds out[1, w] = s
            @inbounds out[2, w] = mn
            @inbounds out[3, w] = mx
        end
        return nothing
    end
    M = 256
    x = CuArray(Float32.(1:M))
    out = CUDA.zeros(Float32, 3, M ÷ 32)
    @cuda threads=M blocks=1 warp_reduce_probe!(out, x)
    CUDA.synchronize()
    got = Array(out)
    for w in 1:(M ÷ 32)
        seg = Float32.((w-1)*32+1 : w*32)
        @assert got[1, w] == sum(seg) "warp $w sum: $(got[1,w]) vs $(sum(seg))"
        @assert got[2, w] == minimum(seg) "warp $w min: $(got[2,w])"
        @assert got[3, w] == maximum(seg) "warp $w max: $(got[3,w])"
    end
    println("compact_kernel!: $n elements CORRECT; warp_reduce +/min/max CORRECT")
end
