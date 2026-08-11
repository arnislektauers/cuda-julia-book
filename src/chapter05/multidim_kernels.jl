# Multi-dimensional kernel indexing: 2D matrix addition and 3D Laplacian
# Demonstrates 2D/3D thread grids with column-major memory access

using CUDA

# --- begin:matadd_2d ---
function matadd_kernel!(C, A, B)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if ix <= size(A, 1) && iy <= size(A, 2)
        @inbounds C[ix, iy] = A[ix, iy] + B[ix, iy]
    end
    return nothing
end

M, N_cols = 1024, 2048
A = CUDA.rand(Float32, M, N_cols)
B = CUDA.rand(Float32, M, N_cols)
C = similar(A)

threads = (16, 16)                         # 256 threads per block
blocks = (cld(M, 16), cld(N_cols, 16))

@cuda threads=threads blocks=blocks matadd_kernel!(C, A, B)
# --- end:matadd_2d ---

# --- begin:laplacian_3d ---
function laplacian_3d_kernel!(out, u, dx2, dy2, dz2)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    iz = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    nx, ny, nz = size(u)
    if 2 <= ix <= nx-1 && 2 <= iy <= ny-1 && 2 <= iz <= nz-1
        @inbounds out[ix,iy,iz] =
            (u[ix-1,iy,iz] - 2*u[ix,iy,iz] + u[ix+1,iy,iz]) / dx2 +
            (u[ix,iy-1,iz] - 2*u[ix,iy,iz] + u[ix,iy+1,iz]) / dy2 +
            (u[ix,iy,iz-1] - 2*u[ix,iy,iz] + u[ix,iy,iz+1]) / dz2
    end
    return nothing
end
# --- end:laplacian_3d ---
