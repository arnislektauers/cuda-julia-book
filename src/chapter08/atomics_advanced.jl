# Advanced atomic operations: CAS-based custom atomics, warp pre-reduction, data scattering

using CUDA

# --- begin:atomic_min_cas ---
using CUDA
using Core: LLVMPtr   # raw device pointer type used in the signature
using CUDA: AS        # address space tags (AS.Global, AS.Shared, ...)

# Note: CUDA.@atomic a[i] = min(a[i], v) already provides Float64 min;
# this listing shows the underlying technique.
@inline function atomic_min_float64!(ptr::LLVMPtr{Float64,AS.Global}, val::Float64)
    ptr_uint = reinterpret(LLVMPtr{UInt64,AS.Global}, ptr)
    old = unsafe_load(ptr_uint)
    while true
        assumed = old
        new_val = min(val, reinterpret(Float64, assumed))
        old = CUDA.atomic_cas!(ptr_uint, assumed, reinterpret(UInt64, new_val))
        old == assumed && break
    end
    return reinterpret(Float64, old)
end
# --- end:atomic_min_cas ---

# --- begin:warp_atomic_preadd ---
@inline function warp_aggregated_atomic_add!(ptr, val::Int32)
    mask = 0xffffffff
    # Warp-level reduction of identical contributions
    val = warp_reduce(+, val)
    # Only lane 1 performs the atomic
    lane = (threadIdx().x - 1) % UInt32(32) + UInt32(1)
    if lane == UInt32(1) && val != Int32(0)
        CUDA.atomic_add!(ptr, val)
    end
    return nothing
end
# --- end:warp_atomic_preadd ---

# --- begin:histogram_scattered ---
# Use NCOPIES independent histograms to spread contention
const NCOPIES = 4
function histogram_scattered!(hists, data, nbins)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    copy = (blockIdx().x - 1) % NCOPIES + 1  # each block targets a different copy

    while i <= length(data)
        bin = clamp(floor(Int32, @inbounds(data[i]) * nbins) + 1, 1, nbins)
        CUDA.atomic_add!(pointer(hists, (copy - 1) * nbins + bin), Int32(1))
        i += stride
    end
    return nothing
end
# --- end:histogram_scattered ---

# ---------------------------------------------------------------------------
# Driver, outside the tagged region so the book is unaffected.
#
# This file cannot run on its own: warp_aggregated_atomic_add! calls
# warp_reduce, which is defined in warp_advanced.jl and never imported here.
# Including that file is honest about the dependency rather than duplicating
# seven lines that would then drift; the side effect is that its own driver
# runs too.
include(joinpath(@__DIR__, "warp_advanced.jl"))

let N = 1 << 16, nbins = 64
    # CAS-based float64 min. Every thread proposes a value; the slot must end
    # up holding the global minimum. Seeded above every proposal so a kernel
    # that never wins its CAS loop leaves the seed behind and is caught.
    function min_probe!(a, vals)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        if i <= length(vals)
            atomic_min_float64!(pointer(a, 1), @inbounds vals[i])
        end
        return nothing
    end
    vals = CuArray(rand(Float64, N) .+ 1.0)
    a = CuArray([1e9])
    @cuda threads=256 blocks=cld(N, 256) min_probe!(a, vals)
    CUDA.synchronize()
    @assert CUDA.@allowscalar(a[1]) == minimum(Array(vals)) "CAS min: $(CUDA.@allowscalar a[1])"

    # Warp-aggregated add: every thread contributes 1, so the total is the
    # thread count. A broken lane predicate shows up as a multiple or a
    # fraction of it rather than as a crash.
    function preadd_probe!(counter)
        warp_aggregated_atomic_add!(pointer(counter, 1), Int32(1))
        return nothing
    end
    counter = CUDA.zeros(Int32, 1)
    @cuda threads=256 blocks=32 preadd_probe!(counter)
    CUDA.synchronize()
    @assert CUDA.@allowscalar(counter[1]) == 256 * 32 "aggregated add: $(CUDA.@allowscalar counter[1])"

    # Scattered histogram: NCOPIES independent copies, summed to compare.
    data = CUDA.rand(Float32, N)
    h = Array(data)
    want = zeros(Int32, nbins)
    for v in h
        want[clamp(floor(Int, v * nbins) + 1, 1, nbins)] += 1
    end
    hists = CUDA.zeros(Int32, NCOPIES * nbins)
    @cuda threads=256 blocks=64 histogram_scattered!(hists, data, nbins)
    CUDA.synchronize()
    merged = sum(reshape(Array(hists), nbins, NCOPIES); dims = 2)[:, 1]
    @assert merged == want "scattered histogram differs from the CPU count"
    println("atomic_min_float64!, warp_aggregated_atomic_add!, histogram_scattered!: all CORRECT")
end
