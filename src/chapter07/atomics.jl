# Atomic operations: basic usage and histogram privatization

using CUDA

# --- begin:atomics_basic ---
# High-level macro
function atomic_increment!(counter)
    CUDA.@atomic counter[1] += 1
    return nothing
end

# Lower-level function
function atomic_add_example!(hist, data, nbins)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    while i <= length(data)
        bin = clamp(floor(Int32, @inbounds(data[i]) * nbins) + 1, 1, nbins)
        CUDA.atomic_add!(pointer(hist, bin), Int32(1))
        i += stride
    end
    return nothing
end
# --- end:atomics_basic ---

# --- begin:histogram_privatized ---
function histogram_privatized!(hist, data, nbins)
    # Block-private histogram in shared memory
    local_hist = CuDynamicSharedArray(Int32, nbins)

    tid = threadIdx().x
    # Initialize shared memory histogram
    i = tid
    while i <= nbins
        @inbounds local_hist[i] = Int32(0)
        i += blockDim().x
    end
    sync_threads()

    # Accumulate into shared memory (lower contention)
    i = (blockIdx().x - 1) * blockDim().x + tid
    stride = blockDim().x * gridDim().x
    while i <= length(data)
        bin = clamp(floor(Int32, @inbounds(data[i]) * nbins) + 1, 1, nbins)
        CUDA.atomic_add!(pointer(local_hist, bin), Int32(1))
        i += stride
    end
    sync_threads()

    # Merge shared histogram into global histogram
    i = tid
    while i <= nbins
        if @inbounds(local_hist[i]) > 0
            CUDA.atomic_add!(pointer(hist, i), @inbounds(local_hist[i]))
        end
        i += blockDim().x
    end
    return nothing
end
# --- end:histogram_privatized ---

# ---------------------------------------------------------------------------
# Driver, outside the tagged region so the book is unaffected. Histograms are
# checked against a CPU count of the same data: an atomic that drops updates
# under contention still produces a plausible-looking histogram, so only the
# exact bin counts distinguish a correct kernel from a lossy one.
let N = 1 << 18, nbins = 64
    counter = CUDA.zeros(Int32, 1)
    @cuda threads=256 blocks=16 atomic_increment!(counter)
    CUDA.synchronize()
    @assert CUDA.@allowscalar(counter[1]) == 256 * 16 "counter: $(CUDA.@allowscalar counter[1])"

    data = CUDA.rand(Float32, N)
    h = Array(data)
    want = zeros(Int32, nbins)
    for v in h
        want[clamp(floor(Int, v * nbins) + 1, 1, nbins)] += 1
    end

    hist = CUDA.zeros(Int32, nbins)
    @cuda threads=256 blocks=64 atomic_add_example!(hist, data, nbins)
    CUDA.synchronize()
    @assert Array(hist) == want "global-atomic histogram differs from the CPU count"

    # Shared-memory privatization: the launch must size the dynamic shared
    # array for nbins Int32s, or the kernel writes past the end of it.
    hist2 = CUDA.zeros(Int32, nbins)
    @cuda threads=256 blocks=64 shmem=nbins*sizeof(Int32) histogram_privatized!(hist2, data, nbins)
    CUDA.synchronize()
    @assert Array(hist2) == want "privatized histogram differs from the CPU count"
    println("atomics: counter, global histogram and privatized histogram all CORRECT")
end
