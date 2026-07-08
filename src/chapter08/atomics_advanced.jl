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
