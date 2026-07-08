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
