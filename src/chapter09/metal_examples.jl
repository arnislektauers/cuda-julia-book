# Metal.jl programming examples

using Metal

# --- begin:metal_arrays ---
using Metal

A = MtlArray(rand(Float32, 1024, 1024))
B = Metal.ones(Float32, 1024)

# Broadcasting via GPUArrays.jl
C = sin.(A) .+ A .^ 2

# Reductions
s = sum(A)
m = maximum(A)
# --- end:metal_arrays ---

# --- begin:metal_memset ---
function memset_kernel!(A, val)
    i = thread_position_in_grid().x
    if i <= length(A)
        @inbounds A[i] = val
    end
    return nothing
end

N = 4096
A = MtlArray{Float32}(undef, N)
@metal threads=512 groups=cld(length(A), 512) memset_kernel!(A, 42f0)
# --- end:metal_memset ---
