# Digital twin of a heated 1D rod: simulate the state, observe it through
# sensors, calibrate the diffusivity, and time the full loop on the GPU.

# --- begin:twin_state_update ---
using CUDA
using Statistics

const N         = 4096       # grid points along the rod
const K         = 8          # number of sensors
const T_HOT     = 100.0f0    # fixed temperature at the heated end
const T_COLD    = 20.0f0     # ambient temperature (cold end, initial state)
const DX        = 1.0f0 / (N - 1)
const ALPHA_MAX = 2.0f-4     # largest diffusivity the twin must handle
const DT        = 0.4f0 * DX^2 / ALPHA_MAX  # r = alpha*DT/DX^2 <= 0.4 < 1/2

function step_kernel!(T_new, T_old, r, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if 2 <= i <= n - 1
        @inbounds T_new[i] = T_old[i] +
            r * (T_old[i + 1] - 2.0f0 * T_old[i] + T_old[i - 1])
    elseif i == 1 || i == n
        @inbounds T_new[i] = T_old[i]   # fixed boundary temperatures
    end
    return nothing
end

function simulate!(T_a, T_b, r, nsteps)
    n = length(T_a)
    threads = 256
    blocks = cld(n, threads)
    for _ in 1:nsteps
        @cuda threads=threads blocks=blocks step_kernel!(T_b, T_a, r, n)
        T_a, T_b = T_b, T_a             # ping-pong swap; no in-place hazard
    end
    return T_a                          # buffer holding the latest state
end

function initial_state(n)
    T = fill(T_COLD, n)
    T[1] = T_HOT
    return CuArray(T)
end
# --- end:twin_state_update ---

# --- begin:twin_sensor_model ---
function sample_kernel!(readings, T, idx)
    k = threadIdx().x
    if k <= length(readings)
        @inbounds readings[k] = T[idx[k]]
    end
    return nothing
end

function read_sensors!(readings, T, idx, sigma)
    @cuda threads=length(readings) sample_kernel!(readings, T, idx)
    if sigma > 0.0f0
        readings .+= sigma .* CUDA.randn(Float32, length(readings))
    end
    return readings
end

# Sensors sit near the heated end, where the thermal transient lives
sensor_idx = CuArray(Int32.(8:8:8 * K))
# --- end:twin_sensor_model ---

# --- begin:twin_calibration ---
const ALPHA_TRUE = 1.0f-4    # diffusivity of the physical rod (unknown to twin)
const NSTEPS     = 2000      # simulated steps per twin cycle
const SIGMA      = 0.2f0     # sensor noise standard deviation, degrees

# "Measured" readings from the physical asset, generated once
T_init = initial_state(N)
T_a, T_b = initial_state(N), initial_state(N)
measured = CUDA.zeros(Float32, K)
T_end = simulate!(T_a, T_b, ALPHA_TRUE * DT / DX^2, NSTEPS)
read_sensors!(measured, T_end, sensor_idx, SIGMA)

# Sweep M candidate diffusivities, reusing the preallocated state buffers
M = 32
alphas = collect(Float32, range(0.5f-4, ALPHA_MAX, length = M))
sim = CUDA.zeros(Float32, K)
errors = zeros(Float32, M)
for (m, alpha) in enumerate(alphas)
    copyto!(T_a, T_init)
    copyto!(T_b, T_init)
    T_cand = simulate!(T_a, T_b, alpha * DT / DX^2, NSTEPS)
    read_sensors!(sim, T_cand, sensor_idx, 0.0f0)   # noiseless prediction
    errors[m] = sum(abs2, sim .- measured)          # fused mismatch reduction
end
alpha_hat = alphas[argmin(errors)]
@assert isapprox(alpha_hat, ALPHA_TRUE; rtol = 0.1) "calibration failed"
# --- end:twin_calibration ---

# --- begin:twin_loop_timing ---
r_hat = alpha_hat * DT / DX^2
copyto!(T_a, T_init)
copyto!(T_b, T_init)
simulate!(T_a, T_b, r_hat, 10)          # warm-up (compilation, first launch)
readings = CUDA.zeros(Float32, K)

t = CUDA.@elapsed CUDA.@sync begin
    T_now = simulate!(T_a, T_b, r_hat, NSTEPS)
    read_sensors!(readings, T_now, sensor_idx, SIGMA)
end

steps_per_sec = NSTEPS / t
bytes = 2 * N * sizeof(Float32) * NSTEPS   # one read + one write per cell
bandwidth_gbs = bytes / t / 1e9
# --- end:twin_loop_timing ---

println("Digital twin summary")
println("  grid points: $N, sensors: $K, steps per cycle: $NSTEPS")
println("  true alpha:      $ALPHA_TRUE")
println("  recovered alpha: $alpha_hat")
println("  mean sensor reading: $(round(mean(Array(readings)), digits = 2)) deg")
println("  cycle time: $(round(t * 1e3, digits = 2)) ms, " *
        "$(round(steps_per_sec / 1e3, digits = 1)) thousand steps/s")
println("  effective bandwidth: $(round(bandwidth_gbs, digits = 1)) GB/s")
