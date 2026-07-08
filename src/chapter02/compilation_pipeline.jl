# GPU compilation pipeline -- introspection and kernel examples

using CUDA

# --- begin:gpu_introspection ---
using CUDA

function add_kernel!(C, A, B)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(C)
        @inbounds C[i] = A[i] + B[i]
    end
    return nothing
end

N = 1024
A = CUDA.rand(Float32, N)
B = CUDA.rand(Float32, N)
C = similar(A)

@device_code_typed @cuda threads=256 blocks=1 add_kernel!(C, A, B)
@device_code_llvm @cuda threads=256 blocks=1 add_kernel!(C, A, B)
@device_code_ptx @cuda threads=256 blocks=1 add_kernel!(C, A, B)
@device_code_sass @cuda threads=256 blocks=1 add_kernel!(C, A, B)
# --- end:gpu_introspection ---
