# cuBLAS matrix multiplication benchmark
# Demonstrates automatic dispatch to cuBLAS SGEMM and Tensor Core usage

# --- begin:cublas_benchmark ---
using CUDA, LinearAlgebra, BenchmarkTools

# Create random matrices on the GPU
N = 4096
A = CUDA.rand(Float32, N, N)
B = CUDA.rand(Float32, N, N)

# Matrix multiplication: dispatches to cuBLAS SGEMM
C = A * B

# Benchmark GPU vs CPU
A_cpu = Array(A)
B_cpu = Array(B)

gpu_time = CUDA.@elapsed C = A * B
cpu_time = @elapsed C_cpu = A_cpu * B_cpu

println("GPU (cuBLAS): $(gpu_time) s")
println("CPU (OpenBLAS): $(cpu_time) s")
println("Speedup: $(cpu_time / gpu_time)x")
# --- end:cublas_benchmark ---
