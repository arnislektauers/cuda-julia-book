# Comparing high-level abstractions vs explicit CUDA kernels

using CUDA

# High-level version using broadcast
A = CUDA.rand(Float32, 1_000_000)
B_broadcast = @. A^2 + 3A + 1       # Single fused GPU kernel

# --- begin:poly_kernel ---
function poly_kernel!(B, A)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(A)
        a = A[i]
        @inbounds B[i] = a * a + 3.0f0 * a + 1.0f0
    end
    return nothing
end

N = length(A)
B = similar(A)
@cuda threads=256 blocks=cld(N, 256) poly_kernel!(B, A)
# --- end:poly_kernel ---
