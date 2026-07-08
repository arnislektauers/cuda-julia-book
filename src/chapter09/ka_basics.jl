# KernelAbstractions.jl basics: indexing, launching, shared memory

using KernelAbstractions
using CUDA

# --- begin:indexing_demo ---
@kernel function indexing_demo_kernel!(output)
    # Global indices
    gi = @index(Global)                  # Global linear index
    gij = @index(Global, Cartesian)      # Global CartesianIndex
    g_tuple = @index(Global, NTuple)     # Global (i, j, ...) tuple

    # Local (workgroup) indices
    li = @index(Local)                   # Local linear index
    lij = @index(Local, Cartesian)       # Local CartesianIndex

    # Group (workgroup) index
    grp = @index(Group)                  # Workgroup linear index
    grp_tuple = @index(Group, NTuple)    # Workgroup (i, j) tuple
end
# --- end:indexing_demo ---

# --- begin:kernel_launch ---
using KernelAbstractions
using CUDA  # Provides CUDABackend

# Create GPU arrays
N = 1_000_000
X = CUDA.rand(Float32, N)
Y = CUDA.rand(Float32, N)
α = 2.0f0

# Determine backend from array type
backend = get_backend(Y)  # Returns CUDABackend()

# Launch kernel
saxpy_kernel!(backend, 256)(Y, α, X; ndrange=N)
# Qualified: both KernelAbstractions and CUDA export `synchronize`
KernelAbstractions.synchronize(backend)
# --- end:kernel_launch ---

# --- begin:tiled_matmul_kernel ---
@kernel function tiled_matmul_kernel!(C, @Const(A), @Const(B),
                                       ::Val{TILE}) where TILE
    gi, gj = @index(Global, NTuple)
    li, lj = @index(Local, NTuple)

    # Allocate workgroup-local (shared) memory
    tile_A = @localmem eltype(A) (TILE, TILE)
    tile_B = @localmem eltype(B) (TILE, TILE)

    # Per-workitem accumulator in @private storage: plain locals are
    # not guaranteed to survive across @synchronize points
    acc = @private eltype(C) 1
    @inbounds acc[1] = zero(eltype(C))
    # Values reused across @synchronize must be @uniform
    @uniform N = size(A, 2)

    for t in 0:TILE:(N - 1)
        # Collaborative load into shared memory
        @inbounds tile_A[li, lj] = A[gi, t + lj]
        @inbounds tile_B[li, lj] = B[t + li, gj]
        @synchronize()

        # Compute partial dot product from shared memory tile
        for k in 1:TILE
            @inbounds acc[1] += tile_A[li, k] * tile_B[k, lj]
        end
        @synchronize()
    end

    @inbounds C[gi, gj] = acc[1]
end
# --- end:tiled_matmul_kernel ---

# --- begin:tiled_matmul_launch ---
M, N, K = 1024, 1024, 1024
TILE = 16
C = KernelAbstractions.zeros(backend, Float32, M, N)
A = KernelAbstractions.ones(backend, Float32, M, K)
B = KernelAbstractions.ones(backend, Float32, K, N)

tiled_matmul_kernel!(backend, (TILE, TILE))(
    C, A, B, Val(TILE); ndrange=(M, N)
)
KernelAbstractions.synchronize(backend)
# --- end:tiled_matmul_launch ---

# --- begin:ka_fallback_pattern ---
using KernelAbstractions

# Generic portable kernel (works on all backends)
@kernel function generic_stencil_kernel!(output, input, coeffs)
    i = @index(Global)
    if 1 < i < length(input)  # Interior points only: guard boundaries
        @inbounds begin
            output[i] = coeffs[1] * input[i-1] +
                         coeffs[2] * input[i]   +
                         coeffs[3] * input[i+1]
        end
    end
end

# Portable entry point
function apply_stencil!(output, input, coeffs)
    backend = get_backend(output)
    generic_stencil_kernel!(backend, 256)(
        output, input, coeffs; ndrange=length(output)
    )
    KernelAbstractions.synchronize(backend)
end

# CUDA-optimized path using shared memory tiling
function apply_stencil!(output::CuArray, input::CuArray, coeffs)
    blocks = cld(length(output), 256)
    shmem = 258 * sizeof(Float32)
    @cuda(threads=256, blocks=blocks, shmem=shmem,
          cuda_stencil_kernel!(output, input, coeffs))
    # Match the generic path: both methods return only
    # after the kernel has completed.
    CUDA.synchronize()
end
# --- end:ka_fallback_pattern ---
