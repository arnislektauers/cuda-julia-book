# Multiple dispatch and GPU kernel dispatch

using CUDA

# --- begin:dispatch_kernel ---
# Generic inner function: works for any AbstractFloat
function compute(x::T, y::T) where T <: AbstractFloat
    return x * y + x
end

# GPU kernel: dispatch resolves at compile time
function my_kernel!(out, a, b)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(out)
        @inbounds out[i] = compute(a[i], b[i])
    end
    return nothing
end
# --- end:dispatch_kernel ---

# --- begin:normalize_dispatch ---
function normalize_columns!(out::AbstractArray, x::AbstractArray)
    s = sum(x)
    out .= x ./ s
    return nothing
end

# CPU path: dispatches to CPU sum and CPU broadcast
x_cpu = rand(Float32, 1000)
out_cpu = similar(x_cpu)
normalize_columns!(out_cpu, x_cpu)

# GPU path: dispatches to CUDA reduction and CUDA broadcast, same source code
x_gpu = CuArray(x_cpu)
out_gpu = similar(x_gpu)
normalize_columns!(out_gpu, x_gpu)
# --- end:normalize_dispatch ---
