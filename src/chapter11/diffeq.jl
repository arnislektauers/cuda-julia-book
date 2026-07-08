# Differential equations -- GPU integration with DifferentialEquations.jl

using OrdinaryDiffEq, CUDA

# --- begin:lorenz96_gpu ---
using OrdinaryDiffEq, CUDA

# Define ODE: du/dt = f(u, p, t) where u is a large vector.
# The RHS uses vectorized circshift/broadcast so it runs on CuArrays
# without scalar indexing: with periodic wrap-around, u[i+1] is
# circshift(u, -1), u[i-1] is circshift(u, 1), u[i-2] is circshift(u, 2)
function lorenz96!(du, u, p, t)
    F = p[1]
    du .= (circshift(u, -1) .- circshift(u, 2)) .* circshift(u, 1) .-
          u .+ F
    return nothing
end

# GPU-resident initial condition (N = 100,000 variables)
N = 100_000
u0 = CuArray(randn(Float32, N))
p = (8.0f0,)    # Forcing parameter
tspan = (0.0f0, 10.0f0)

prob = ODEProblem(lorenz96!, u0, tspan, p)
sol = solve(prob, Tsit5(); saveat=0.1f0)    # Entire solve runs on GPU
# --- end:lorenz96_gpu ---

# --- begin:ensemble_gpu ---
using OrdinaryDiffEq, DiffEqGPU, CUDA

# Simple Lotka-Volterra ODE
function lotka_volterra!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[1] * u[2]
    du[2] = δ * u[1] * u[2] - γ * u[2]
    return nothing
end

# Base problem
u0 = Float32[1.0, 1.0]
tspan = (0.0f0, 10.0f0)
p = Float32[1.5, 1.0, 3.0, 1.0]
prob = ODEProblem(lotka_volterra!, u0, tspan, p)

# Generate 10,000 parameter variations
function prob_func(prob, ctx)
    remake(prob; p = prob.p .* (1.0f0 .+ 0.1f0 .* randn(Float32, 4)))
end

ensemble_prob = EnsembleProblem(prob; prob_func=prob_func)
sol = solve(ensemble_prob, Tsit5(), EnsembleGPUArray(CUDA.CUDABackend());
            trajectories=10_000, saveat=0.1f0)
# --- end:ensemble_gpu ---

# --- begin:ensemble_gpu_kernel ---
using OrdinaryDiffEq, DiffEqGPU, StaticArrays, CUDA

# Out-of-place RHS with SVector state and parameters
function lotka_volterra(u, p, t)
    α, β, γ, δ = p
    return SVector{2}(α * u[1] - β * u[1] * u[2],
                      δ * u[1] * u[2] - γ * u[2])
end

prob = ODEProblem{false}(lotka_volterra,
                         SVector{2, Float32}(1.0f0, 1.0f0),
                         (0.0f0, 10.0f0),
                         SVector{4, Float32}(1.5f0, 1.0f0, 3.0f0, 1.0f0))

# Vary parameters across the ensemble via SVector
function prob_func(prob, ctx)
    remake(prob; p = prob.p .* (1.0f0 .+ 0.1f0 .* @SVector(randn(Float32, 4))))
end

ensemble_prob = EnsembleProblem(prob; prob_func=prob_func)
sol = solve(ensemble_prob, GPUTsit5(), EnsembleGPUKernel(CUDA.CUDABackend());
            trajectories=10_000, saveat=0.1f0)
# --- end:ensemble_gpu_kernel ---

# --- begin:gray_scott_gpu ---
using OrdinaryDiffEq, CUDA

function gray_scott_gpu!(du_flat, u_flat, p, t)
    Du, Dv, F, k, N = p
    half = N * N

    u = reshape(@view(u_flat[1:half]), N, N)
    v = reshape(@view(u_flat[half+1:end]), N, N)
    du = reshape(@view(du_flat[1:half]), N, N)
    dv = reshape(@view(du_flat[half+1:end]), N, N)

    # Periodic Laplacian via circshift, as a 5-point stencil in grid units
    # (the standard Gray-Scott scaling; the diffusion coefficients absorb the
    # grid spacing). Each shifted copy supplies one stencil neighbor, so no
    # scalar indexing occurs on the GPU.
    lap_u = circshift(u, (1, 0)) .+ circshift(u, (-1, 0)) .+
            circshift(u, (0, 1)) .+ circshift(u, (0, -1)) .- 4.0f0 .* u
    lap_v = circshift(v, (1, 0)) .+ circshift(v, (-1, 0)) .+
            circshift(v, (0, 1)) .+ circshift(v, (0, -1)) .- 4.0f0 .* v

    uv2 = u .* v .^ 2
    du .= Du .* lap_u .- uv2 .+ F .* (1.0f0 .- u)
    dv .= Dv .* lap_v .+ uv2 .- (F .+ k) .* v
    return nothing
end

# Initial condition built on the CPU: u background 1, v background 0,
# with a small central square perturbation that seeds pattern formation
N = 256
u_init = ones(Float32, N, N)
v_init = zeros(Float32, N, N)
c = N ÷ 2
u_init[c-10:c+10, c-10:c+10] .= 0.5f0
v_init[c-10:c+10, c-10:c+10] .= 0.25f0

# Concatenate the two fields and move the state to the GPU
u0 = CuArray(vcat(vec(u_init), vec(v_init)))

p = (0.16f0, 0.08f0, 0.06f0, 0.062f0, N)  # Du, Dv, F, k, N (Int)
tspan = (0.0f0, 1000.0f0)

prob = ODEProblem(gray_scott_gpu!, u0, tspan, p)
sol = solve(prob, Tsit5(); saveat=100.0f0)
# --- end:gray_scott_gpu ---
