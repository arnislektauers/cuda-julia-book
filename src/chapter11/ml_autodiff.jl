# Machine learning -- custom automatic differentiation rules on the GPU
#
# Neither ChainRulesCore nor Enzyme collides with Flux or Lux exports, but
# these regions are grouped here so the framework files stay single-purpose.

# --- begin:chainrules_gpu ---
using ChainRulesCore, CUDA

# Custom GPU operation: soft thresholding
function soft_threshold(x::CuArray, λ::Real)
    return sign.(x) .* max.(abs.(x) .- λ, 0)
end

# Define the backward rule for AD
function ChainRulesCore.rrule(::typeof(soft_threshold),
                               x::CuArray, λ::Real)
    y = soft_threshold(x, λ)
    function soft_threshold_pullback(ȳ)
        active = abs.(x) .> λ
        # Gradient for x: pass ȳ through active entries, 0 elsewhere
        dx = ȳ .* active
        # Gradient for λ: each active output shrinks by sign(x)
        dλ = -sum(ȳ .* sign.(x) .* active)
        return NoTangent(), dx, dλ
    end
    return y, soft_threshold_pullback
end
# --- end:chainrules_gpu ---

# --- begin:enzyme_gpu ---
using Enzyme, CUDA

function my_kernel!(y, x, α)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(x)
        @inbounds y[i] = α * x[i]^2 + sin(x[i])
    end
    return nothing
end

# Host wrapper that launches the kernel; Enzyme differentiates
# through the @cuda launch via CUDA.jl's Enzyme extension
function apply!(y, x, α)
    @cuda threads=256 blocks=cld(length(y), 256) my_kernel!(y, x, α)
    return nothing
end

# Enzyme generates the gradient kernel automatically
x = CUDA.rand(Float32, 1024)
y = CUDA.zeros(Float32, 1024)
dx = CUDA.zeros(Float32, 1024)
dy = CUDA.ones(Float32, 1024)

# Reverse-mode differentiation through the GPU kernel launch
Enzyme.autodiff(Reverse, apply!, Const, Duplicated(y, dy),
                Duplicated(x, dx), Const(2.0f0))
# --- end:enzyme_gpu ---
