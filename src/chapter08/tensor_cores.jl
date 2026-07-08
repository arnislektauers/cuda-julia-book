# Tensor Core programming: WMMA and cuBLAS usage

using CUDA
using CUDA.WMMA
using LinearAlgebra

# --- begin:wmma_gemm ---
using CUDA
using CUDA.WMMA

# Configuration: 16×16×16 MMA with FP16 inputs, FP32 accumulator
const M_TILE = 16
const N_TILE = 16
const K_TILE = 16

function wmma_gemm_kernel!(D, A, B, C, M, N, K)
    # Every WMMA call takes the shape/precision configuration
    conf = WMMA.Config{M_TILE, N_TILE, K_TILE, Float32}

    # Warp-level: each warp computes one 16×16 output tile
    warp_row = ((blockIdx().x - 1) * blockDim().y + threadIdx().y - 1) * M_TILE + 1
    warp_col = ((blockIdx().y - 1) * blockDim().x + threadIdx().x - 1) ÷ 32 * N_TILE + 1

    # Column-major: element (row, col) has 1-based linear index
    # row + (col - 1) * ld, with leading dimension ld = M for A and C/D,
    # ld = K for B.
    # Load the accumulator with C so the kernel computes D = A * B + C
    c_frag = WMMA.load_c(pointer(C, warp_row + (warp_col - 1) * M),
                         M, WMMA.ColMajor, conf)

    # Iterate over K dimension in tiles
    for k_step in 0:K_TILE:(K - 1)
        # A tile starts at (warp_row, k_step + 1), B tile at (k_step + 1, warp_col)
        a_frag = WMMA.load_a(pointer(A, warp_row + k_step * M),
                             M, WMMA.ColMajor, conf)
        b_frag = WMMA.load_b(pointer(B, (k_step + 1) + (warp_col - 1) * K),
                             K, WMMA.ColMajor, conf)

        # Tensor Core MMA: c_frag = a_frag * b_frag + c_frag
        c_frag = WMMA.mma(a_frag, b_frag, c_frag, conf)
    end

    # Store result tile at (warp_row, warp_col)
    WMMA.store_d(pointer(D, warp_row + (warp_col - 1) * M),
                 c_frag, M, WMMA.ColMajor, conf)
    return nothing
end
# --- end:wmma_gemm ---

# --- begin:cublas_tensor ---
using CUDA, LinearAlgebra

# FP16 inputs: cuBLAS uses Tensor Cores automatically
A = CUDA.rand(Float16, 4096, 4096)
B = CUDA.rand(Float16, 4096, 4096)
C = A * B  # dispatches to cuBLAS GEMM with Tensor Cores

# FP32 inputs with TF32 Tensor Core math
A32 = CUDA.rand(Float32, 4096, 4096)
B32 = CUDA.rand(Float32, 4096, 4096)

# Enable TF32 mode for FP32 inputs (uses Tensor Cores with reduced precision)
CUDA.math_mode!(CUDA.FAST_MATH)
C32 = A32 * B32  # uses TF32 Tensor Core path

# Restore default precision
CUDA.math_mode!(CUDA.DEFAULT_MATH)
# --- end:cublas_tensor ---

# --- begin:iterative_refinement ---
using CUDA, LinearAlgebra

function iterative_refinement(A_f64, b_f64; maxiter=10, tol=1e-12)
    # Low-precision factorization in FP32 (cuSOLVER provides no FP16 LU).
    # On Ampere and later, CUDA.math_mode!(CUDA.FAST_MATH) lets the FP32
    # factorization use TF32 Tensor Cores; cuSOLVER also offers dedicated
    # mixed-precision refinement solvers (cusolverDnIRSXgesv).
    F = lu(CuArray{Float32}(A_f64))

    # Initial solve in FP32, promoted to FP64
    x = CuArray{Float64}(F \ CuArray{Float32}(b_f64))

    for iter in 1:maxiter
        # Compute residual in FP64
        r = b_f64 - A_f64 * x

        # Check convergence
        norm_r = norm(r)
        norm_r < tol * norm(b_f64) && break

        # Solve correction in FP32, accumulate update in FP64
        Δx = CuArray{Float64}(F \ CuArray{Float32}(r))
        x .+= Δx
    end
    return x
end
# --- end:iterative_refinement ---
