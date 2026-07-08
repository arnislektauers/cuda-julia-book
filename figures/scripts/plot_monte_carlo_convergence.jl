# Plot: Monte Carlo convergence for Chapter 12
#
# Measures real GPU/CPU timings for a Monte Carlo π estimator, writes the
# figure (PDF/SVG) and the underlying data (.dat) next to this script, and
# displays the plot:
#   * Inline in JupyterLab (auto-detected via IJulia)
#   * In a native interactive window when run from the command line
#
# Requires: CUDA.jl, CairoMakie, BenchmarkTools.
# Optional: GLMakie (only loaded outside Jupyter, for the native window).
#
# Run with a CUDA-capable GPU. Memory requirement at the largest N (10^9):
# 2 × Float32 arrays × 4 GB = 8 GB device memory; lower MAX_K if needed.

using CUDA
using CairoMakie
using BenchmarkTools
using Random
using Printf

# Chunk size for the GPU estimator. Each chunk allocates 2 × Float32 arrays
# of this length plus broadcast temporaries. 10^8 elements ≈ 400 MB per array,
# so peak working set is ~1.5 GB — fits comfortably on an 8 GB GPU.
const GPU_CHUNK = 10^8

# --- Backend selection ----------------------------------------------------------
# Detect IJulia (JupyterLab / classic Jupyter): we want CairoMakie active so the
# returned `fig` renders inline in the cell. Outside Jupyter, prefer GLMakie for
# an interactive window.
const IN_JUPYTER = isdefined(Main, :IJulia)

if IN_JUPYTER
    CairoMakie.activate!()
else
    try
        @eval using GLMakie
        Base.invokelatest(GLMakie.activate!)
    catch err
        @warn "GLMakie not available; falling back to CairoMakie (no interactive window)" exception=err
        CairoMakie.activate!()
    end
end

# --- Sample sizes (10^MIN_K ... 10^MAX_K) ---
const MIN_K = 2
const MAX_K = 9
N = [10^k for k in MIN_K:MAX_K]

# --- Estimators -----------------------------------------------------------------

"""
GPU array-based Monte Carlo π estimator.

Processes the samples in chunks of `GPU_CHUNK` so the peak working-set fits
on a typical 8 GB GPU. The intermediate arrays from each chunk are released
explicitly via `CUDA.unsafe_free!` before the next iteration to avoid
relying on the GC to keep memory pressure low.

Returns a `Float64` estimate — the final ratio uses Float64 so it isn't
capped by Float32's ~7-digit precision (which would otherwise show up at
the largest N).
"""
function monte_carlo_pi_gpu(n::Int)
    inside::Int64 = 0
    remaining = n
    while remaining > 0
        m = min(remaining, GPU_CHUNK)
        x = CUDA.rand(Float32, m)
        y = CUDA.rand(Float32, m)
        hits = x .^ 2 .+ y .^ 2 .<= 1.0f0
        inside += Int64(sum(hits))
        CUDA.unsafe_free!(x)
        CUDA.unsafe_free!(y)
        CUDA.unsafe_free!(hits)
        remaining -= m
    end
    return 4.0 * Float64(inside) / Float64(n)
end

"""Single-threaded CPU Monte Carlo π estimator."""
function monte_carlo_pi_cpu(n::Int)
    rng = Random.default_rng()
    inside = 0
    @inbounds for _ in 1:n
        x = rand(rng, Float32)
        y = rand(rng, Float32)
        inside += (x * x + y * y) <= 1.0f0
    end
    return 4.0f0 * Float32(inside) / Float32(n)
end

# --- Benchmark helpers ----------------------------------------------------------

"""Pick `samples`/`seconds` budgets that scale down for large N."""
function _budget(n::Int)
    if n <= 10^4
        return (samples = 50, seconds = 1.0)
    elseif n <= 10^6
        return (samples = 20, seconds = 2.0)
    elseif n <= 10^8
        return (samples = 5,  seconds = 5.0)
    else
        return (samples = 3,  seconds = 30.0)
    end
end

function bench_gpu(n::Int)
    # Warmup (compile + initial CURAND state)
    CUDA.@sync monte_carlo_pi_gpu(n)
    b = _budget(n)
    # @benchmarkable + run() lets us pass runtime budgets; @benchmark would
    # need literal kwargs at macro-expansion time.
    bm = @benchmarkable CUDA.@sync monte_carlo_pi_gpu($n)
    bench = run(bm; samples = b.samples, seconds = b.seconds, evals = 1)
    return minimum(bench).time / 1e9   # ns → s
end

function bench_cpu(n::Int)
    monte_carlo_pi_cpu(min(n, 10_000))  # warmup
    b = _budget(n)
    bm = @benchmarkable monte_carlo_pi_cpu($n)
    bench = run(bm; samples = b.samples, seconds = b.seconds, evals = 1)
    return minimum(bench).time / 1e9
end

# --- Error estimation -----------------------------------------------------------
# A single Monte Carlo trial is noisy: its absolute error fluctuates by a
# factor of two or more even when E[|π̂ − π|] ∝ 1/√N. To recover the
# theoretical convergence trend on the plot, we average |π̂ − π| over
# several independent trials per N.

"""How many independent trials to average per N (more for cheap N)."""
function _trial_count(n::Int)
    if n <= 10^4
        return 50
    elseif n <= 10^6
        return 20
    elseif n <= 10^8
        return 8
    else
        return 4
    end
end

"""
Mean absolute error |π̂ − π| over `T` independent GPU trials, plus the
standard error of that mean.

Returns `(mean, sem)` where:
  * `mean` = (1/T) Σ |π̂ᵢ − π|
  * `sem`  = std({|π̂ᵢ − π|}) / √T  — uncertainty of the estimated mean
"""
function mean_abs_error(n::Int, T::Int = _trial_count(n))
    samples = Vector{Float64}(undef, T)
    for i in 1:T
        π̂ = monte_carlo_pi_gpu(n)
        samples[i] = abs(π̂ - π)
    end
    μ = sum(samples) / T
    sem = if T > 1
        # unbiased sample variance → standard error of the mean
        v = sum((s - μ)^2 for s in samples) / (T - 1)
        sqrt(v / T)
    else
        0.0
    end
    return (mean = μ, sem = sem)
end

# --- Measurement loop -----------------------------------------------------------

err     = Float64[]
err_sem = Float64[]   # standard error of the mean for each N
gpu_t   = Float64[]
cpu_t   = Float64[]

println("Measuring Monte Carlo convergence and timings…")
println("CUDA device: ", CUDA.name(CUDA.device()))
@printf "%10s  %5s  %10s  %10s  %12s  %12s\n" "N" "T" "mean|π̂-π|" "SEM" "GPU (s)" "CPU (s)"
@printf "%s\n" repeat("-", 72)

for n in N
    T = _trial_count(n)
    e = mean_abs_error(n, T)
    push!(err,     e.mean)
    push!(err_sem, e.sem)

    push!(gpu_t, bench_gpu(n))
    push!(cpu_t, bench_cpu(n))

    @printf "%10d  %5d  %10.3e  %10.3e  %12.3e  %12.3e\n" n T err[end] err_sem[end] gpu_t[end] cpu_t[end]

    # Free large GPU arrays before next iteration
    GC.gc(); CUDA.reclaim()
end

# O(1/√N) reference, anchored at the first error point.
ref = [err[1] * sqrt(N[1] / n) for n in N]

# --- Colors (book palette) ------------------------------------------------------
C_ERR  = "#2563EB"  # Blue
C_GPU  = "#DC2626"  # Red
C_CPU  = "#6B7280"  # Gray
C_REF  = "#9CA3AF"  # Lighter gray
C_NOTE = "#16A34A"  # Green (crossover highlight)

# --- Plot -----------------------------------------------------------------------
fig = Figure(size = (720, 440), fontsize = 11)

ax_err = Axis(fig[1, 1],
    xscale = log10,
    yscale = log10,
    xlabel = "Sample count N",
    ylabel = "Absolute error |π̂ − π|",
    xticks = (10.0 .^ (MIN_K:MAX_K),
              ["10$(s)" for s in ["²","³","⁴","⁵","⁶","⁷","⁸","⁹"][1:(MAX_K - MIN_K + 1)]]),
    yticks = (10.0 .^ (-5:1:2), ["10⁻⁵","10⁻⁴","10⁻³","10⁻²","10⁻¹","10⁰","10¹","10²"]),
    ylabelcolor = C_ERR,
    yticklabelcolor = C_ERR,
    limits = ((10.0^MIN_K, 10.0^MAX_K), (1e-5, 1e2)),
)

ax_t = Axis(fig[1, 1],
    xscale = log10,
    yscale = log10,
    ylabel = "Execution time (s)",
    yaxisposition = :right,
    yticks = (10.0 .^ (-5:1:2), ["10⁻⁵","10⁻⁴","10⁻³","10⁻²","10⁻¹","10⁰","10¹","10²"]),
    ylabelcolor = C_GPU,
    yticklabelcolor = C_GPU,
    limits = ((10.0^MIN_K, 10.0^MAX_K), (1e-5, 1e2)),
)
hidespines!(ax_t)
hidexdecorations!(ax_t)
linkxaxes!(ax_err, ax_t)

# Error bars (standard error of the mean) on the absolute-error points.
# On a log-y axis the lower whisker can't reach 0; clamp it to a small
# fraction of the mean so the bar stays visible without underflowing.
let
    # Asymmetric offsets: errorbars!(x, y, low, high) draws from y-low to y+high.
    low  = [min(err_sem[i], 0.95 * err[i]) for i in eachindex(err)]
    high = err_sem
    errorbars!(ax_err, Float64.(N), err, low, high;
        color = C_ERR, whiskerwidth = 8, linewidth = 1.0)
end

l_err = scatterlines!(ax_err, N, err;
    color = C_ERR, linewidth = 2.0, markersize = 7, marker = :circle)

l_ref = lines!(ax_err, N, ref;
    color = C_REF, linestyle = :dot, linewidth = 1.4)

l_gpu = scatterlines!(ax_t, N, gpu_t;
    color = C_GPU, linestyle = :dash, linewidth = 1.8,
    markersize = 7, marker = :utriangle)

l_cpu = scatterlines!(ax_t, N, cpu_t;
    color = C_CPU, linestyle = :dash, linewidth = 1.8,
    markersize = 7, marker = :rect)

# Crossover: smallest N where GPU is at least as fast as CPU
let cross_idx = findfirst(i -> gpu_t[i] <= cpu_t[i], eachindex(N))
    if cross_idx !== nothing
        N_cross = N[cross_idx]
        vlines!(ax_err, [Float64(N_cross)];
                color = C_NOTE, linestyle = :dash, linewidth = 1.0)
        text!(ax_err, 1.1 * N_cross, 40; text = "GPU advantage",
              color = C_NOTE, fontsize = 10, font = :bold,
              align = (:left, :center))
    end
end

axislegend(ax_err,
    [l_err, l_ref, l_gpu, l_cpu],
    ["|π̂ − π|", "O(1/√N) reference", "GPU time (s)", "CPU time (s)"];
    position = :lb, labelsize = 9, framevisible = true, padding = (6, 6, 4, 4))

# --- Save (vector output via CairoMakie) ----------------------------------------
# Always use CairoMakie for the file output. If GLMakie is the active backend
# (non-Jupyter run), switch around the saves and switch back.
let prev_backend = Makie.current_backend()
    CairoMakie.activate!()
    save("monte-carlo-convergence_en.svg", fig)
    save("monte-carlo-convergence_en.pdf", fig)
    prev_backend === nothing || prev_backend.activate!()
end

# Space-separated columns; matches the format used under figures/data/.
# Suitable for pgfplot table: imports.
open("monte-carlo-convergence.dat", "w") do io
    println(io, "# N  abs_error  abs_error_sem  gpu_time_s  cpu_time_s  ref_inv_sqrt_n")
    for i in eachindex(N)
        @printf(io, "%.6e  %.6e  %.6e  %.6e  %.6e  %.6e\n",
                N[i], err[i], err_sem[i], gpu_t[i], cpu_t[i], ref[i])
    end
end

println("\nFigures saved: monte-carlo-convergence_en.{svg,pdf}")
println("Data saved:    monte-carlo-convergence.dat")

# --- Display --------------------------------------------------------------------
# In Jupyter we just return `fig` so IJulia auto-renders it inline as the
# cell's last expression. Outside Jupyter we open a native interactive window
# (GLMakie) and block on it when run as a non-interactive script.
if IN_JUPYTER
    fig
else
    screen = display(fig)
    if !isinteractive()
        println("Close the plot window to exit…")
        wait(screen)
    end
    fig
end
