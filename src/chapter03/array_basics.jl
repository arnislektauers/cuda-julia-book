# CuArray creation and data transfer patterns

using CUDA

# --- begin:cuarray_allocation ---
using CUDA

# Uninitialized allocation (fastest: no fill operation)
A = CuArray{Float32}(undef, 1024, 1024)

# Zero-initialized arrays
B = CUDA.zeros(Float32, 1024, 1024)
C = CUDA.ones(Float32, 512)

# Random arrays (uses cuRAND library)
D = CUDA.rand(Float32, 1024)         # Uniform [0, 1)
E = CUDA.randn(Float32, 1024)        # Normal distribution N(0, 1)

# Fill with a specific value
F = CUDA.fill(3.14f0, 256, 256)
# --- end:cuarray_allocation ---

# --- begin:adapt_pattern ---
A_gpu = CUDA.rand(Float32, 1024)

# GPU -> CPU
A_cpu = Array(A_gpu)                 # Explicit copy to CPU
A_cpu = collect(A_gpu)               # Equivalent

# CPU -> GPU
B_gpu = CuArray(A_cpu)               # Explicit copy to GPU

# Adapt.jl for generic code
using Adapt
A_adapted = adapt(CuArray, A_cpu)    # Works for nested structures too
# --- end:adapt_pattern ---
