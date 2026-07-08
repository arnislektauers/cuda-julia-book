# Branch divergence: data-dependent branching vs branchless alternative
# Demonstrates the use of ifelse() to avoid warp divergence

using CUDA

# BAD: divergence likely within warps
function divergent_kernel!(out, x)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(x)
        if x[i] > 0           # data-dependent branch -> divergence
            out[i] = sqrt(x[i])
        else
            out[i] = -x[i]
        end
    end
    return nothing
end

# BETTER: branchless alternative
function branchless_kernel!(out, x)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(x)
        xi = @inbounds x[i]
        @inbounds out[i] = ifelse(xi > 0, sqrt(xi), -xi)
    end
    return nothing
end
