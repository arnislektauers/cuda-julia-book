# Register usage example and CUDA.Const lookup kernel

using CUDA

# --- begin:register_example ---
using CUDA

function register_example!(A)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(A)
        val = @inbounds A[i]         # loaded into a register
        val = val * val + 1.0f0      # register arithmetic
        @inbounds A[i] = val
    end
    return nothing
end

A = CUDA.fill(1.0f0, 128)
@cuda threads=128 register_example!(A)
# --- end:register_example ---

# --- begin:lookup_const ---
function lookup_kernel!(out, table, indices)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    # Wrap the read-only table in CUDA.Const inside the kernel to enable
    # compiler-assisted read-only cached loads.
    ctable = CUDA.Const(table)
    if i <= length(indices)
        @inbounds out[i] = ctable[indices[i]]
    end
    return nothing
end

table = CUDA.fill(1.0f0, 256)
indices = CuArray(Int32.(rand(1:256, 10_000)))
out = similar(table, length(indices))

@cuda threads=256 blocks=cld(10_000, 256) lookup_kernel!(out, table, indices)
# --- end:lookup_const ---
