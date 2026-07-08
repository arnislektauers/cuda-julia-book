# CUDA library integration: cuBLAS, cuFFT, cuSOLVER, cuSPARSE, cuDNN

using CUDA
using LinearAlgebra

# --- begin:cublas_ops ---
using CUDA, LinearAlgebra

A = CUDA.rand(Float32, 2048, 2048)
B = CUDA.rand(Float32, 2048, 2048)
x = CUDA.rand(Float32, 2048)

# BLAS Level 3: matrix-matrix multiply (dispatches to cuBLAS GEMM)
C = A * B

# BLAS Level 2: matrix-vector multiply (dispatches to cuBLAS GEMV)
y = A * x

# In-place GEMM with explicit α, β
α = 1.0f0; β = 0.0f0
C_out = similar(C)
mul!(C_out, A, B, α, β)  # C_out = α * A * B + β * C_out

# Direct cuBLAS gemm! call, avoiding the generic dispatch layer
CUDA.CUBLAS.gemm!('N', 'N', α, A, B, β, C_out)
# --- end:cublas_ops ---

# --- begin:cufft_ops ---
using CUDA, CUDA.CUFFT

# 1D FFT
x = CUDA.rand(Float32, 1024)
X = fft(x)           # returns CuArray{ComplexF32}

# Inverse FFT
x_back = real(ifft(X))

# 2D FFT (e.g., for image processing or spectral methods)
img = CUDA.rand(Float32, 512, 512)
IMG = fft(img)

# Plan-based FFT for repeated transforms (avoids plan recomputation)
plan = plan_fft(x)
X = plan * x         # execute forward FFT
x_back = plan \ X    # execute inverse FFT
# --- end:cufft_ops ---

# --- begin:cusolver_ops ---
using CUDA, LinearAlgebra

A = CUDA.rand(Float32, 2048, 2048)
A = A * A' + 100I  # make positive definite
b = CUDA.rand(Float32, 2048)

# LU factorization (dispatches to cuSOLVER)
F = lu(A)
x = F \ b

# Cholesky factorization for positive definite matrices
C = cholesky(A)
x = C \ b

# QR factorization
Q, R = qr(A)

# Eigenvalue decomposition (A is symmetric by construction;
# the symmetric path is the robust, widely supported one)
eigenvalues = eigvals(Hermitian(A))

# Singular value decomposition
U, S, V = svd(A)
# --- end:cusolver_ops ---

# --- begin:cusparse_ops ---
using CUDA, CUDA.CUSPARSE, SparseArrays

# Create sparse matrix on CPU and transfer to GPU
A_cpu = sprand(Float32, 10000, 10000, 0.01)  # 1% density
A_gpu = CuSparseMatrixCSR(A_cpu)              # CSR format on GPU

# Sparse matrix-vector multiply
x = CUDA.rand(Float32, 10000)
y = A_gpu * x  # dispatches to cuSPARSE SpMV

# Sparse matrix-matrix multiply
B_gpu = CuSparseMatrixCSR(sprand(Float32, 10000, 10000, 0.01))
C = A_gpu * B_gpu  # cuSPARSE SpGEMM

# Convert between formats
A_csc = CuSparseMatrixCSC(A_gpu)  # CSR -> CSC
# --- end:cusparse_ops ---

# --- begin:cudnn_ops ---
using CUDA, cuDNN

# Direct cuDNN usage (rarely needed; use Flux.jl instead)
# Convolution, pooling, and normalization are wrapped

# High-level usage via Flux.jl (recommended)
# using Flux
# model = Chain(Conv((3,3), 1=>16, relu), MaxPool((2,2)), ...)
# gpu_model = model |> gpu
# gpu_data = data |> gpu
# output = gpu_model(gpu_data)
# --- end:cudnn_ops ---
