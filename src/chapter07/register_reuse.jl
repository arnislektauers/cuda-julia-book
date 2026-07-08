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
