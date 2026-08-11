# Branch divergence: branchless patterns for GPU kernels

using CUDA

# --- begin:relu_branchless ---
# Conditional: threads in the same warp may take different paths
function relu_divergent!(out, x)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(x)
        @inbounds out[i] = x[i] > 0 ? x[i] : 0.0f0   # potential divergence
    end
    return nothing
end

# Branchless: selection made explicit with ifelse
function relu_branchless!(out, x)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(x)
        xi = @inbounds x[i]
        @inbounds out[i] = ifelse(xi > 0, xi, 0.0f0)   # branchless
    end
    return nothing
end

# --- end:relu_branchless ---

# --- begin:activation_branchless ---
# Version 1: Divergent (if-else)
function activation_divergent!(out, x)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    while i <= length(x)
        xi = @inbounds x[i]
        if xi > 0
            @inbounds out[i] = xi                   # identity
        elseif xi > -1.0f0
            @inbounds out[i] = xi + 0.5f0 * xi * xi  # quadratic
        else
            @inbounds out[i] = 0.0f0                  # zero
        end
        i += stride
    end
    return nothing
end

# Version 2: Branchless
function activation_branchless!(out, x)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    while i <= length(x)
        xi = @inbounds x[i]
        # Compute all three branches
        v_pos = xi
        v_mid = xi + 0.5f0 * xi * xi
        v_neg = 0.0f0
        # Select result without branching
        result = ifelse(xi > 0, v_pos, ifelse(xi > -1.0f0, v_mid, v_neg))
        @inbounds out[i] = result
        i += stride
    end
    return nothing
end
# --- end:activation_branchless ---

# ---------------------------------------------------------------------------
# Driver, outside the tagged region so the book is unaffected. Both pairs claim
# the branchless form is equivalent to the branching one, so each pair is
# checked against its partner and against the formula.
#
# The inputs straddle every boundary the three-way activation tests: above 0,
# between -1 and 0, and below -1. A range that missed one would leave a branch
# unexercised while still passing.
let N = 1 << 16
    x = CuArray(Float32.(range(-2.0f0, 2.0f0; length = N)))
    a = Array(x)
    o1, o2 = CUDA.zeros(Float32, N), CUDA.zeros(Float32, N)

    @cuda threads=256 blocks=cld(N, 256) relu_divergent!(o1, x)
    @cuda threads=256 blocks=cld(N, 256) relu_branchless!(o2, x)
    CUDA.synchronize()
    @assert Array(o1) == Array(o2) "relu: branchless disagrees with branching"
    @assert Array(o2) == max.(a, 0.0f0) "relu: does not match max(x, 0)"

    # Grid-stride loops, so the launch need not cover the input exactly.
    @cuda threads=256 blocks=64 activation_divergent!(o1, x)
    @cuda threads=256 blocks=64 activation_branchless!(o2, x)
    CUDA.synchronize()
    want = [xi > 0 ? xi : (xi > -1.0f0 ? xi + 0.5f0 * xi * xi : 0.0f0) for xi in a]
    @assert Array(o1) == Array(o2) "activation: branchless disagrees with branching"
    @assert Array(o2) == want "activation: does not match the three-way formula"
    println("relu_*/activation_*: branchless matches branching, both CORRECT")
end
