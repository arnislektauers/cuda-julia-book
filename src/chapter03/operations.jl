# GPU array operations: reductions, numerical patterns, statistics

using CUDA, Statistics

# --- begin:reduction_ops ---
A = CUDA.rand(Float32, 1_000_000)

s = sum(A)                           # GPU reduction -> scalar
p = prod(A[1:100])                   # Product (on a view)
mn = minimum(A)                      # Global minimum
mx = maximum(A)                      # Global maximum

# Dimensional reductions
B = CUDA.rand(Float32, 1024, 512)
col_sums = sum(B, dims=1)            # 1×512 CuArray
row_sums = sum(B, dims=2)            # 1024×1 CuArray
# --- end:reduction_ops ---

# --- begin:numerical_patterns ---
A = CUDA.rand(Float32, 1_000_000)

# L2 norm — single kernel, no temporaries
norm_l2 = sqrt(mapreduce(x -> x^2, +, A))

# Mean absolute error — single kernel
B = CUDA.rand(Float32, 1_000_000)
mae = mapreduce((a, b) -> abs(a - b), +, A, B; init = 0.0f0) / length(A)

# Log-sum-exp (numerically stable): shift by the max, exponentiate with a
# fused broadcast, then reduce
m = maximum(A)
lse = m + log(sum(exp.(A .- m)))
# --- end:numerical_patterns ---

# --- begin:statistics_rng ---
using Statistics

A = CUDA.rand(Float32, 10_000)

# Statistics stdlib
m = mean(A)       # GPU reduction
v = var(A)        # GPU reduction
s = std(A)        # GPU reduction

# Random number generation
R = CUDA.randn(Float32, 1024, 1024)  # cuRAND on GPU

# Sorting
# CUDA.jl parallel sort (quicksort/bitonic)
sorted = sort(A)  

# Finding elements
idx = findmax(A)   # Returns (value, index)
# --- end:statistics_rng ---
