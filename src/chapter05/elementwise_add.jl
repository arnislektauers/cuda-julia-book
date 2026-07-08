# Elementwise addition kernels: basic, grid-stride, and vector scaling
# Demonstrates kernel anatomy, launch configuration, and scalar parameters

using CUDA

# --- begin:add_complete ---
using CUDA

function add_kernel!(c, a, b)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(a)
        @inbounds c[i] = a[i] + b[i]
    end
    return nothing
end

N = 10_000
a = CUDA.rand(Float32, N)
b = CUDA.rand(Float32, N)
c = similar(a)

threads = 256
blocks = cld(N, threads)     # ceiling division: cld(N, threads)

@cuda threads=threads blocks=blocks add_kernel!(c, a, b)
CUDA.synchronize()

# Verify correctness against CPU computation
@assert Array(c) ≈ Array(a) .+ Array(b)
# --- end:add_complete ---

# --- begin:add_stride ---
function add_kernel_stride!(c, a, b)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    i = index
    while i <= length(a)
        @inbounds c[i] = a[i] + b[i]
        i += stride
    end
    return nothing
end
# --- end:add_stride ---

# --- begin:scale_kernel ---
function scale_kernel!(a, α)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(a)
        @inbounds a[i] *= α
    end
    return nothing
end

a = CUDA.rand(Float32, 10_000)
@cuda threads=256 blocks=cld(10_000, 256) scale_kernel!(a, 2.0f0)
CUDA.synchronize()
# --- end:scale_kernel ---
