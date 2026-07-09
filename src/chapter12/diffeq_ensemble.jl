# GPU-accelerated ensemble ODE solving with DiffEqGPU.jl
# SIR epidemic model parameter sweep

# --- begin:sir_ensemble ---
using DiffEqGPU, OrdinaryDiffEq, CUDA, StaticArrays

# SIR model (out-of-place for GPU compatibility)
function sir(u, p, t)
    S, I, R = u
    beta, gamma = p
    dS = -beta * S * I
    dI = beta * S * I - gamma * I
    dR = gamma * I
    return SVector{3, Float32}(dS, dI, dR)
end

u0 = @SVector Float32[0.99, 0.01, 0.0]
tspan = (0.0f0, 200.0f0)
p = @SVector Float32[0.3, 0.1]
prob = ODEProblem{false}(sir, u0, tspan, p)

# Vary parameters across ensemble
function prob_func(prob, i, repeat)
    remake(prob, p = @SVector Float32[
        0.1f0 + 0.4f0 * rand(Float32),
        0.05f0 + 0.15f0 * rand(Float32)
    ])
end

ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)

# GPU ensemble solve
sol = solve(ensemble_prob, GPUTsit5(),
            EnsembleGPUKernel(CUDA.CUDABackend()),
            trajectories = 100_000,
            saveat = 1.0f0)
# --- end:sir_ensemble ---
