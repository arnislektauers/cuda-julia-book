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

a = oneArray(rand(Float32, 4096))
b = oneArray(rand(Float32, 4096))
c = similar(a)

items = 256
groups = cld(length(a), items)
@oneapi items=items groups=groups vadd_oneapi!(c, a, b)
# --- end:oneapi_vadd ---

# ---------------------------------------------------------------------------
# Driver, outside the tagged region so the book is unaffected. The listings
# print nothing, so completing proves only that they did not crash.
#
# Needs an Intel GPU exposing a Level Zero device. Verified on homebox's
# integrated UHD 770 (i5-13600K, oneAPI.jl 2.8.0) after enabling integrated
# graphics in UEFI -- the iGPU is off by default when a discrete card is
# fitted. Primary display stays on the NVIDIA card; the iGPU is compute-only.
let
    A = oneArray(rand(Float32, 256, 256))
    h = Array(A)
    C = sin.(A) .+ cos.(A)
    @assert Array(C) ≈ sin.(h) .+ cos.(h) "broadcast differs from the host computation"
    D = A * A'                       # oneMKL GEMM
    @assert Array(D) ≈ h * h' "GEMM differs from the host computation"

    # @oneapi is asynchronous; c is only settled after a synchronize.
    oneAPI.synchronize()
    @assert Array(c) ≈ Array(a) .+ Array(b) "vadd kernel result is wrong"
    println("oneapi: broadcast, oneMKL GEMM and vadd_oneapi! all CORRECT")
end
