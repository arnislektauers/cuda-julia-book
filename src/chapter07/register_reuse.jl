# Register-level reuse: avoiding redundant global memory reads

using CUDA

# --- begin:register_reuse ---
# Poor: repeated global memory reads
function poor_reuse!(out, A, x, y)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(out)
        @inbounds out[i] = A[i] * A[i] + A[i] * x + y   # A[i] loaded 3×
    end
    return nothing
end

# Better: register-level reuse
function good_reuse!(out, A, x, y)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(out)
        ai = @inbounds A[i]          # one load -> register
        @inbounds out[i] = ai * ai + ai * x + y   # reused from register
    end
    return nothing
end
# --- end:register_reuse ---

# ---------------------------------------------------------------------------
# Driver, outside the tagged region so the book is unaffected. The listing's
# claim is that the two kernels differ only in how many times A[i] is loaded,
# so the check is that they agree exactly -- not approximately, since both
# evaluate the same expression in the same order.
let N = 1 << 16, x = 3.0f0, y = 1.5f0
    A = CUDA.rand(Float32, N)
    o1, o2 = CUDA.zeros(Float32, N), CUDA.zeros(Float32, N)
    @cuda threads=256 blocks=cld(N, 256) poor_reuse!(o1, A, x, y)
    @cuda threads=256 blocks=cld(N, 256) good_reuse!(o2, A, x, y)
    CUDA.synchronize()
    @assert Array(o1) == Array(o2) "register reuse changed the result"
    a = Array(A)
    @assert Array(o2) ≈ a .* a .+ a .* x .+ y "result does not match the formula"
    println("poor_reuse!/good_reuse!: identical and CORRECT")
end
