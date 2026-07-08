# Thread indexing patterns: 1D grid-stride, 1D and 2D indexing

using CUDA

# --- begin:grid_stride ---
function grid_stride_kernel!(A, val)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while i <= length(A)
        @inbounds A[i] *= val
        i += stride
    end
    return nothing
end
# --- end:grid_stride ---

# --- begin:index_kernel ---
function index_kernel!(output)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(output)
        @inbounds output[i] = i
    end
    return nothing
end

N = 16
output = CUDA.zeros(Int32, N)
@cuda threads=4 blocks=4 index_kernel!(output)
CUDA.synchronize()
println(Array(output))
# Output: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# --- end:index_kernel ---

# --- begin:index_2d_kernel ---
function index_2d_kernel!(out_x, out_y, M, N)
    ix = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    if ix <= M && iy <= N
        idx = ix + (iy - Int32(1)) * M  # column-major linear index
        @inbounds out_x[idx] = ix
        @inbounds out_y[idx] = iy
    end
    return nothing
end
# --- end:index_2d_kernel ---
