# Parallel histogram with atomic operations
# Demonstrates CUDA.@atomic and the grid-stride pattern

using CUDA

# --- begin:histogram_kernel ---
function histogram_kernel!(hist, data, num_bins)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    while i <= length(data)
        bin = clamp(floor(Int32, data[i] * num_bins) + 1, 1, num_bins)
        CUDA.@atomic hist[bin] += Int32(1)
        i += stride
    end
    return nothing
end
# --- end:histogram_kernel ---

# Driver code
N = 1_000_000
num_bins = 256
data = CUDA.rand(Float32, N)
hist = CUDA.zeros(Int32, num_bins)

@cuda threads=256 blocks=min(cld(N, 256), 256) histogram_kernel!(hist, data, Int32(num_bins))
CUDA.synchronize()

@assert sum(Array(hist)) == N
