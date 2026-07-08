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
