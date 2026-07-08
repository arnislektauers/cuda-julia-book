# Plot: Neural-network training-time scaling vs batch size for Chapter 12
#
# Shows the GPU's batch-size sensitivity for the quadrant-classification MLP
# (2 → 30 → 1, ReLU + sigmoid). Two panels:
#
#   (a) Wall-clock time per epoch on CPU vs GPU, log-log, with the crossover
#       batch size annotated.
#   (b) Throughput (training samples / second) on CPU vs GPU, log-log.
#
# The figure renders correctly without CUDA: by default the script uses a
# baked-in dataset of representative measurements so the build never depends
# on a GPU. Set the env var BENCHMARK=true to re-measure on the local
# hardware (CUDA + Flux required); the script will then write its measured
# numbers into `nn-batch-scaling.dat` and use them.
#
# Run from the figures/ directory or wherever — outputs land alongside the
# script.
#
# Requires: CairoMakie. Optional: CUDA, Flux, BenchmarkTools (only when
# BENCHMARK=true).

using CairoMakie
using Printf

# --- Configuration --------------------------------------------------------------
const TOTAL_SAMPLES = 100_000          # training-set size held constant across runs
const BATCH_SIZES   = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
const EPOCHS_BENCH  = 20               # epochs per measurement when benchmarking
const DATA_PATH     = joinpath(@__DIR__, "nn-batch-scaling.dat")

const BENCHMARK = lowercase(get(ENV, "BENCHMARK", "false")) in ("1", "true", "yes")

# --- Baked-in representative measurements --------------------------------------
# Numbers approximate a single-thread Julia + OpenBLAS CPU vs an RTX-class GPU
# running the chapter's 2 → 30 → 1 MLP for one full pass over 100k samples.
# Small batches: GPU is dominated by per-launch overhead (~10 launches/batch
# × ~50 μs each), CPU is fastest. Large batches: GPU saturates and pulls ahead.
#
# Layout: (batch, cpu_s_per_epoch, gpu_s_per_epoch)
const BAKED_DATA = [
    (   32, 0.180, 0.950),
    (   64, 0.110, 0.550),
    (  128, 0.075, 0.320),
    (  256, 0.060, 0.190),
    (  512, 0.055, 0.115),
    ( 1024, 0.062, 0.075),
    ( 2048, 0.078, 0.057),
    ( 4096, 0.110, 0.050),
    ( 8192, 0.180, 0.054),
    (16384, 0.330, 0.072),
]

# --- Live benchmarking (optional) ----------------------------------------------

"""
Build a fresh small MLP matching the chapter's 2 → 30 → 1 architecture.

`device` is `cpu` or `gpu` (the Flux helpers, not strings). Returns
`(model, opt_state)` ready for `Flux.train!`.
"""
function build_model(device)
    @eval using Flux
    m = Chain(Dense(2, 30, relu), Dense(30, 1, sigmoid))
    m = device(m)
    opt_state = Flux.setup(Flux.Adam(0.01f0), m)
    return m, opt_state
end

"""
Time a single epoch (one full pass over `total_samples` in chunks of
`batch_size`). Returns seconds.

`device` selects CPU vs GPU; the synthetic data is generated on the chosen
device so we don't double-count host→device transfers per step.
"""
function time_one_epoch(batch_size::Int, total_samples::Int, device)
    @eval using Flux
    @eval using BenchmarkTools

    num_batches = cld(total_samples, batch_size)
    # Pre-build all batches on the target device so the timing isolates the
    # forward/backward/update cost.
    Xs = [Base.invokelatest(device, randn(Float32, 2, batch_size)) for _ in 1:num_batches]
    Ys = [Base.invokelatest(device, Float32.(rand(Bool, 1, batch_size))) for _ in 1:num_batches]

    model, opt_state = build_model(device)
    loss(m, x, y) = Flux.binarycrossentropy(m(x), y)

    # Warmup: one full pass to compile + amortise CUDA initialisation
    for i in 1:num_batches
        gs = Flux.gradient(m -> loss(m, Xs[i], Ys[i]), model)
        Flux.update!(opt_state, model, gs[1])
    end

    # Time: best of 3 epochs
    times = Float64[]
    for _ in 1:3
        t0 = time_ns()
        for i in 1:num_batches
            gs = Flux.gradient(m -> loss(m, Xs[i], Ys[i]), model)
            Flux.update!(opt_state, model, gs[1])
        end
        # Sync if on GPU
        if device !== Flux.cpu
            @eval CUDA.synchronize()
        end
        push!(times, (time_ns() - t0) / 1e9)
    end
    return minimum(times)
end

function run_benchmark()
    @eval using CUDA
    @eval using Flux

    println("Measuring NN batch-size scaling…")
    println("CUDA device: ", CUDA.name(CUDA.device()))
    @printf "%8s  %12s  %12s\n" "batch" "CPU s/epoch" "GPU s/epoch"
    @printf "%s\n" repeat("-", 40)

    rows = NTuple{3, Float64}[]
    for B in BATCH_SIZES
        cpu_t = time_one_epoch(B, TOTAL_SAMPLES, Flux.cpu)
        gpu_t = time_one_epoch(B, TOTAL_SAMPLES, Flux.gpu)
        push!(rows, (Float64(B), cpu_t, gpu_t))
        @printf "%8d  %12.4f  %12.4f\n" B cpu_t gpu_t
        GC.gc(); CUDA.reclaim()
    end

    open(DATA_PATH, "w") do io
        println(io, "# batch  cpu_s_per_epoch  gpu_s_per_epoch")
        for (B, c, g) in rows
            @printf(io, "%6d  %14.6e  %14.6e\n", Int(B), c, g)
        end
    end
    println("Wrote measured data to $(DATA_PATH)")
    return rows
end

function load_data()
    if BENCHMARK
        return run_benchmark()
    end
    # Prefer a previously measured .dat if present, otherwise fall back to
    # baked-in numbers so the build is reproducible without a GPU.
    if isfile(DATA_PATH)
        rows = NTuple{3, Float64}[]
        for line in eachline(DATA_PATH)
            startswith(strip(line), "#") && continue
            isempty(strip(line)) && continue
            parts = split(strip(line))
            push!(rows, (parse(Float64, parts[1]),
                         parse(Float64, parts[2]),
                         parse(Float64, parts[3])))
        end
        if !isempty(rows)
            println("Loaded measured data from $(DATA_PATH)")
            return rows
        end
    end
    println("Using baked-in representative data (set BENCHMARK=true to re-measure on this hardware)")
    return [(Float64(b), c, g) for (b, c, g) in BAKED_DATA]
end

# --- Plot -----------------------------------------------------------------------

CairoMakie.activate!()

const C_CPU = "#6B7280"  # Gray
const C_GPU = "#DC2626"  # Red
const C_NOTE = "#16A34A" # Green
const C_AXIS = "#1F2937"

function build_figure(rows)
    batches = [r[1] for r in rows]
    cpu_t   = [r[2] for r in rows]
    gpu_t   = [r[3] for r in rows]
    cpu_thr = TOTAL_SAMPLES ./ cpu_t
    gpu_thr = TOTAL_SAMPLES ./ gpu_t

    # Crossover: smallest batch size where GPU time <= CPU time
    cross_idx = findfirst(i -> gpu_t[i] <= cpu_t[i], eachindex(batches))
    B_cross  = cross_idx === nothing ? nothing : batches[cross_idx]

    fig = Figure(size = (760, 380), fontsize = 11)

    # Common log-axis tick labels using superscripts (clean book typography)
    sup = Dict(0=>"⁰", 1=>"¹", 2=>"²", 3=>"³", 4=>"⁴", 5=>"⁵", 6=>"⁶", 7=>"⁷", 8=>"⁸", 9=>"⁹",
               -1=>"⁻¹", -2=>"⁻²", -3=>"⁻³", -4=>"⁻⁴", -5=>"⁻⁵")
    pow10 = (k) -> "10$(sup[k])"

    # ---- Panel (a): Wall-clock time per epoch ----
    ax1 = Axis(fig[1, 1],
        title = "(a) Time per epoch",
        xscale = log10, yscale = log10,
        xlabel = "Batch size", ylabel = "Wall-clock time per epoch (s)",
        xticks = (batches, string.(Int.(batches))),
        yticks = (10.0 .^ (-2:1:1), [pow10(k) for k in -2:1:1]),
        xticklabelrotation = π/4,
        limits = ((batches[1] / 1.4, batches[end] * 1.4), (1e-2, 2e0)),
    )
    l_cpu = scatterlines!(ax1, batches, cpu_t;
        color = C_CPU, linewidth = 2.0, marker = :rect,    markersize = 8)
    l_gpu = scatterlines!(ax1, batches, gpu_t;
        color = C_GPU, linewidth = 2.0, marker = :utriangle, markersize = 9)

    if B_cross !== nothing
        vlines!(ax1, [B_cross];
                color = C_NOTE, linestyle = :dash, linewidth = 1.0)
        # Annotate near the bottom of the panel so it never collides with the
        # legend in the top-right.
        text!(ax1, B_cross * 1.1, 1.4e-2;
              text = "GPU faster →",
              color = C_NOTE, fontsize = 10, font = :bold,
              align = (:left, :center))
    end

    axislegend(ax1, [l_cpu, l_gpu], ["CPU", "GPU"];
        position = :rt, labelsize = 10,
        framevisible = true, padding = (6, 6, 4, 4))

    # ---- Panel (b): Throughput ----
    ax2 = Axis(fig[1, 2],
        title = "(b) Training throughput",
        xscale = log10, yscale = log10,
        xlabel = "Batch size", ylabel = "Samples per second",
        xticks = (batches, string.(Int.(batches))),
        yticks = (10.0 .^ (5:1:7), [pow10(k) for k in 5:1:7]),
        xticklabelrotation = π/4,
        limits = ((batches[1] / 1.4, batches[end] * 1.4), (8e4, 5e6)),
    )
    l_cpu2 = scatterlines!(ax2, batches, cpu_thr;
        color = C_CPU, linewidth = 2.0, marker = :rect,    markersize = 8)
    l_gpu2 = scatterlines!(ax2, batches, gpu_thr;
        color = C_GPU, linewidth = 2.0, marker = :utriangle, markersize = 9)

    if B_cross !== nothing
        vlines!(ax2, [B_cross];
                color = C_NOTE, linestyle = :dash, linewidth = 1.0)
        text!(ax2, B_cross * 1.1, 1.0e5;
              text = "GPU faster →",
              color = C_NOTE, fontsize = 10, font = :bold,
              align = (:left, :center))
    end

    axislegend(ax2, [l_cpu2, l_gpu2], ["CPU", "GPU"];
        position = :lt, labelsize = 10,
        framevisible = true, padding = (6, 6, 4, 4))

    return fig
end

rows = load_data()
fig = build_figure(rows)

let outdir = @__DIR__
    save(joinpath(outdir, "nn-batch-scaling_en.svg"), fig)
    save(joinpath(outdir, "nn-batch-scaling_en.pdf"), fig)
    println("Figures saved: nn-batch-scaling_en.{svg,pdf} in $(outdir)")
end

fig
