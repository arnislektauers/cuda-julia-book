# oneAPI.jl programming examples

using oneAPI

# --- begin:oneapi_arrays ---
using oneAPI

A = oneArray(rand(Float32, 1024, 1024))
B = oneAPI.ones(Float32, 1024)

# Broadcasting via GPUArrays.jl
C = sin.(A) .+ cos.(A)

# Linear algebra via oneMKL
D = A * A'  # Dispatches to oneMKL GEMM
# --- end:oneapi_arrays ---

# --- begin:oneapi_vadd ---
function vadd_oneapi!(c, a, b)
    i = get_global_id(1)
    if i <= length(a)
        @inbounds c[i] = a[i] + b[i]
    end
    return nothing
end

@oneapi items=length(a) vadd_oneapi!(c, a, b)
# --- end:oneapi_vadd ---
