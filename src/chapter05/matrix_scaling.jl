# Row and column scaling kernels
# Demonstrates the impact of memory access patterns in column-major layout

using CUDA

# --- begin:rowscale ---
function rowscale_kernel!(A, scale)
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if row <= size(A, 1)
        for j in 1:size(A, 2)
            @inbounds A[row, j] *= scale[row]
        end
    end
    return nothing
end
# --- end:rowscale ---

# --- begin:colscale ---
function colscale_kernel!(A, scale)
    col = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if col <= size(A, 2)
        for i in 1:size(A, 1)
            @inbounds A[i, col] *= scale[col]
        end
    end
    return nothing
end
# --- end:colscale ---

# Demo
M, N = 1024, 512
A = CUDA.rand(Float32, M, N)
row_scale = CUDA.rand(Float32, M)
col_scale = CUDA.rand(Float32, N)

@cuda threads=256 blocks=cld(M, 256) rowscale_kernel!(A, row_scale)
@cuda threads=256 blocks=cld(N, 256) colscale_kernel!(A, col_scale)
