# Complete vector scaling example: kernel, launch, verify

using CUDA

function scale_kernel!(A, factor)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(A)
        @inbounds A[i] = A[i] * factor
    end
    return nothing
end

# --- begin:scale_complete ---
using CUDA

# 1. Define kernel
function scale_kernel!(A, factor)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(A)
        @inbounds A[i] = A[i] * factor
    end
    return nothing
end

# 2. Allocate and initialize data on GPU
N = 1_000_000
A = CUDA.ones(Float32, N)        # All elements = 1.0

# 3. Configure and launch kernel
threads = 256
blocks = cld(N, threads)
@cuda threads=threads blocks=blocks scale_kernel!(A, 3.0f0)

# 4. Synchronize — wait for GPU to finish
CUDA.synchronize()

# 5. Verify results on CPU
A_host = Array(A)
@assert all(x -> x ≈ 3.0f0, A_host) "Scaling failed!"
println("All $(N) elements correctly scaled to 3.0")
# --- end:scale_complete ---

# --- begin:debug_kernel ---
# Enable bounds checking (slower, for debugging only)
# Launch without @inbounds to enable runtime checks
function debug_kernel!(A)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(A)
        A[i] = A[i] * 2.0f0   # No @inbounds — bounds are checked
    end
    return nothing
end
# --- end:debug_kernel ---

# --- begin:test_correctness ---
function test_kernel_correctness(N=10_000)
    A_cpu = rand(Float32, N)
    A_gpu = CuArray(copy(A_cpu))

    # CPU reference
    expected = A_cpu .* 3.0f0

    # GPU kernel
    @cuda threads=256 blocks=cld(N, 256) scale_kernel!(A_gpu, 3.0f0)
    CUDA.synchronize()

    # Compare
    @assert Array(A_gpu) ≈ expected "Results differ!"
end
# --- end:test_correctness ---
