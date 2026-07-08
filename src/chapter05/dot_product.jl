# Naive dot product kernel (elementwise multiply + reduce)
# Demonstrates the two-phase reduction pattern

using CUDA

# --- begin:dot_naive ---
function dot_naive_kernel!(tmp, a, b)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(a)
        @inbounds tmp[i] = a[i] * b[i]
    end
    return nothing
end

N = 1_000_000
a = CUDA.rand(Float32, N)
b = CUDA.rand(Float32, N)
tmp = similar(a)             # temporary storage for elementwise products

# Launch elementwise product kernel
@cuda threads=256 blocks=cld(N, 256) dot_naive_kernel!(tmp, a, b)
# Reduce on device using built-in reduction
result = sum(tmp)
# --- end:dot_naive ---
