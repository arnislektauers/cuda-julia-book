# Branch divergence example

using CUDA

# --- begin:divergent_kernel ---
function divergent_kernel!(A, B)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(A)
        # Some threads take if-path, others take else-path
        if A[i] > 0.0f0          
            @inbounds B[i] = A[i] * 2.0f0
        else
            @inbounds B[i] = A[i] * -1.0f0
        end
    end
    return nothing
end
# --- end:divergent_kernel ---

# ---------------------------------------------------------------------------
# Driver, outside the tagged region so the book is unaffected. Alternating
# signs put both branches in every warp, which is the divergence the listing
# is about; the check is that both paths compute what they claim.
let N = 1024
    A = CuArray(Float32[iseven(i) ? i : -i for i in 1:N])
    B = CUDA.zeros(Float32, N)
    @cuda threads=256 blocks=cld(N, 256) divergent_kernel!(A, B)
    CUDA.synchronize()
    a, b = Array(A), Array(B)
    @assert all(b .== ifelse.(a .> 0, a .* 2.0f0, a .* -1.0f0)) "divergent branches disagree"
    println("divergent_kernel!: both branches CORRECT")
end
