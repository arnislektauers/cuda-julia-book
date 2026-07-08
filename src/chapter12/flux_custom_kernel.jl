# Custom CUDA kernel integration with Flux.jl automatic differentiation

# --- begin:flux_custom ---
using Flux, CUDA, ChainRulesCore

# Step 1: Forward kernel
# smooth_clamp(x) = x / (1 + |x|) is the standard softsign function
function smooth_clamp_kernel!(y, x)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(x)
        @inbounds y[i] = x[i] / (1.0f0 + abs(x[i]))
    end
    return nothing
end

# Step 2: Wrapper function
function smooth_clamp(x::CuArray)
    y = similar(x)
    threads = 256
    blocks = cld(length(x), threads)
    @cuda threads=threads blocks=blocks smooth_clamp_kernel!(y, x)
    return y
end

# Step 3: AD rule via ChainRulesCore
function ChainRulesCore.rrule(::typeof(smooth_clamp), x::CuArray)
    y = smooth_clamp(x)
    function smooth_clamp_pullback(dy)
        # d/dx [x / (1 + |x|)] = 1 / (1 + |x|)^2
        dx = dy ./ (1 .+ abs.(x)).^2
        return NoTangent(), dx
    end
    return y, smooth_clamp_pullback
end

# Step 4: Use in Flux model
model = Chain(
    Dense(784 => 256),
    smooth_clamp,
    Dense(256 => 10)
) |> gpu
# --- end:flux_custom ---
