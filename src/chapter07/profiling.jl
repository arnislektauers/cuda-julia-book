# Profiling GPU kernels: timing and bandwidth analysis

using CUDA

# --- begin:profiling_benchmark ---
using CUDA

N = 10_000_000
a = CUDA.rand(Float32, N)
b = CUDA.rand(Float32, N)
c = similar(a)

function add_kernel!(c, a, b)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    while i <= length(a)
        @inbounds c[i] = a[i] + b[i]
        i += stride
    end
    return nothing
end

# Warm up (trigger compilation)
@cuda threads=256 blocks=256 add_kernel!(c, a, b)
CUDA.synchronize()

# Measure
t = CUDA.@elapsed begin
    @cuda threads=256 blocks=256 add_kernel!(c, a, b)
end
bytes = 3 * N * sizeof(Float32)  # 2 reads + 1 write
bw = bytes / t / 1e9
println("Time: $(round(t * 1000, digits=3)) ms")
println("Effective bandwidth: $(round(bw, digits=1)) GB/s")
# --- end:profiling_benchmark ---

# --- begin:stencil_profiling ---
function stencil_naive!(out, u, c0, c1)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if 2 <= i <= length(u) - 1
        @inbounds out[i] = c0 * u[i] + c1 * (u[i-1] + u[i+1])
    end
    return nothing
end

N = 10_000_000
u = CUDA.rand(Float32, N)
out = similar(u)

# Warm up
@cuda threads=256 blocks=cld(N, 256) stencil_naive!(out, u, 0.5f0, 0.25f0)
CUDA.synchronize()

# Profile
t = CUDA.@elapsed begin
    @cuda threads=256 blocks=cld(N, 256) stencil_naive!(out, u, 0.5f0, 0.25f0)
end

# Analysis
flops = 4 * N         # 2 adds + 2 multiplies per element
bytes = 2N * 4        # read u (with overlap) + write out
ai = flops / bytes
println("Arithmetic intensity: $(round(ai, digits=2)) FLOP/byte")
println("Time: $(round(t * 1000, digits=3)) ms")
println("Achieved bandwidth: $(round(bytes / t / 1e9, digits=1)) GB/s")
println("Achieved GFLOP/s: $(round(flops / t / 1e9, digits=1))")
# --- end:stencil_profiling ---
