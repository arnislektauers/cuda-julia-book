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

# ---------------------------------------------------------------------------
# Driver, outside the tagged region so the book is unaffected. The listings
# print nothing, so completing proves only that they did not crash; these
# checks establish that the array operations and the kernel launch actually
# computed something.
#
# Runs on Apple Silicon only. Metal.jl needs a Metal-capable device, so this
# file cannot be exercised on the NVIDIA host the rest of the book uses --
# see runlist.txt.
let
    A = MtlArray(rand(Float32, 256, 256))
    h = Array(A)
    C = sin.(A) .+ A .^ 2
    @assert Array(C) ≈ sin.(h) .+ h .^ 2 "broadcast result differs from the host computation"
    # Float32 reductions over 65_536 elements accumulate in a different order
    # on the GPU, so compare with a tolerance rather than exactly.
    @assert isapprox(sum(A), sum(h); rtol = 1f-4) "sum: $(sum(A)) vs $(sum(h))"
    @assert maximum(A) == maximum(h) "maximum: $(maximum(A)) vs $(maximum(h))"

    N = 4096
    B = MtlArray{Float32}(undef, N)
    @metal threads=512 groups=cld(N, 512) memset_kernel!(B, 42f0)
    Metal.synchronize()
    @assert all(Array(B) .== 42f0) "memset kernel left $(count(Array(B) .!= 42f0)) cells unwritten"
    println("metal: broadcast, reductions and memset_kernel! all CORRECT")
end
