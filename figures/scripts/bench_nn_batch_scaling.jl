# Measure the CPU-vs-GPU batch-size scaling behind the Chapter 12 figure
# `fig-nn-batch-scaling` (per-epoch time and training throughput).
#
# Emits one `.dat` file consumed by two visutwin specs:
#
#   nn-batch-scaling.dat  -> figures/spec/nn-batch-scaling-time.yaml       (panel a)
#                         -> figures/spec/nn-batch-scaling-throughput.yaml (panel b)
#
# Both panels come from the same measurement: throughput is TOTAL_SAMPLES
# divided by the per-epoch time, so the file carries all four columns and
# neither spec has to do arithmetic.
#
# What is timed
# -------------
# One epoch: a full pass over a fixed 100 000-sample training set in batches of
# the swept size. The network is the chapter's quadrant-classification MLP
# (2 -> 30 -> 1, ReLU + sigmoid), and the GPU side is the book's own code --
# `Dense`, `forward!`, `backward!`, `bce_loss`, `bce_grad` are `include`d from
# the generated listings rather than re-implemented here, so the figure measures
# exactly what the chapter prints. The CPU baseline is the transcription below:
# the same equations with `Array` in place of `CuArray`, because the book's
# methods are typed on `CuMatrix` and cannot run on the host.
#
# Three decisions that determine what the numbers mean:
#
#   * The per-batch loss stays in the timed loop. `bce_loss` returns a host
#     scalar, so the printed training loop synchronizes with the device once
#     per batch. Dropping it would flatter the GPU at small batch sizes -- and
#     that synchronization is precisely what the chapter text blames for the
#     small-batch behavior, so it belongs inside the clock. The CPU baseline
#     computes the same loss, so the comparison stays like-for-like.
#
#   * Batches are staged on their device before the clock starts, host-side for
#     the CPU and device-side for the GPU. The chapter's loop does the same: it
#     builds its batches once and then reuses them across all epochs, so no
#     host-device copy happens per step. (Chapter 11's table went the other way
#     and included the transfers, because there the input starts on the host
#     every time. Whether transfers are in the measurement decides the answer;
#     it is stated here so the number can be read for what it is.)
#
#   * The CPU baseline is single-threaded -- BLAS is pinned to one thread below
#     and the thread count is recorded -- matching the caption's claim. At a
#     30-wide hidden layer the GEMMs are far too small for multithreaded BLAS to
#     help anyway, but the pin makes that a decision rather than an accident.
#
# The reported time is the minimum over at least MIN_TRIALS epochs, extended
# until MIN_SECONDS of wall clock have been spent; the minimum is the right
# statistic here for the same reason it is in bench_chapter_figures.jl, because
# contention only ever adds time.
#
# Run on the reference machine (Appendix A: RTX 4070 SUPER, Julia 1.12.6,
# CUDA.jl 6.2.1), clearing LD_LIBRARY_PATH first:
#
#   LD_LIBRARY_PATH= julia --project=<env-with-CUDA> bench_nn_batch_scaling.jl
#
# OUT_DIR selects where the `.dat` and the environment stamp are written
# (default figures/data/); LISTINGS_DIR points at book/listings/ when the script
# runs outside a full checkout.

using CUDA
using LinearAlgebra
using Printf
using Random

const OUT_DIR      = get(ENV, "OUT_DIR", normpath(joinpath(@__DIR__, "..", "data")))
const LISTINGS_DIR = get(ENV, "LISTINGS_DIR",
                         normpath(joinpath(@__DIR__, "..", "..", "book", "listings")))

const TOTAL_SAMPLES = 100_000               # training-set size, held constant
const BATCH_SIZES   = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
const LR            = 0.5f0                 # learning rate, as in the chapter
const HIDDEN        = 30                    # hidden width, as in the chapter
const SEED          = 20260509

const MIN_TRIALS  = 3                       # timed epochs per configuration
const MAX_TRIALS  = 20
const MIN_SECONDS = 1.0                     # keep timing until this much elapsed

# The book's network, verbatim: these are the generated listings for
# lst-nn-layers, lst-nn-forward and lst-nn-backward. The fourth region of
# src/chapter12/neural_network.jl (nn_training) is deliberately not included --
# it ends by calling its own 1000-epoch driver.
include(joinpath(LISTINGS_DIR, "ch12_nn_layers.jl"))
include(joinpath(LISTINGS_DIR, "ch12_nn_forward.jl"))
include(joinpath(LISTINGS_DIR, "ch12_nn_backward.jl"))

# =================================================================================
# CPU baseline: the same network on the host
# =================================================================================
# A transcription of the three listings above with `Array` for `CuArray`. The
# arithmetic is identical operation for operation -- a GEMM plus a bias
# broadcast forward, a GEMM pair plus two broadcasts backward -- so the two
# curves differ in hardware and not in formulation.

mutable struct DenseCPU
    W::Matrix{Float32}
    b::Vector{Float32}
    σ::Function
    dσ::Function
    input::Matrix{Float32}
    z::Matrix{Float32}
end

function DenseCPU(nin::Int, nout::Int, σ, dσ)
    W = randn(Float32, nout, nin) .* sqrt(2.0f0 / nin)
    b = zeros(Float32, nout)
    DenseCPU(W, b, σ, dσ, zeros(Float32, nin, 0), zeros(Float32, nout, 0))
end

function forward_cpu!(layer::DenseCPU, A::Matrix{Float32})
    layer.input = A
    layer.z = layer.W * A .+ layer.b
    return layer.σ.(layer.z)
end

function forward_cpu!(layers::Vector{DenseCPU}, X::Matrix{Float32})
    A = X
    for layer in layers
        A = forward_cpu!(layer, A)
    end
    return A
end

function bce_loss_cpu(pred::Matrix{Float32}, target::Matrix{Float32})
    m = size(pred, 2)
    p = clamp.(pred, 1.0f-7, 1.0f0 - 1.0f-7)
    return -sum(target .* log.(p) .+ (1.0f0 .- target) .* log.(1.0f0 .- p)) / m
end

bce_grad_cpu(pred::Matrix{Float32}, target::Matrix{Float32}) =
    (p = clamp.(pred, 1.0f-7, 1.0f0 - 1.0f-7);
     @. -(target / p - (1.0f0 - target) / (1.0f0 - p)))

function backward_cpu!(layer::DenseCPU, dA::Matrix{Float32}, lr::Float32)
    m = size(dA, 2)
    dZ = dA .* layer.dσ.(layer.z)
    dA_prev = layer.W' * dZ
    layer.W .-= lr .* (dZ * layer.input') ./ m
    layer.b .-= lr .* vec(sum(dZ; dims = 2)) ./ m
    return dA_prev
end

function backward_cpu!(layers::Vector{DenseCPU}, dY::Matrix{Float32}, lr::Float32)
    dA = dY
    for layer in Iterators.reverse(layers)
        dA = backward_cpu!(layer, dA, lr)
    end
    return nothing
end

# =================================================================================
# Data and epochs
# =================================================================================

"""
The chapter's synthetic quadrant-classification set: label 1 when the two
coordinates share a sign, 0 otherwise. Returns host matrices in the
(features x samples) column-major layout the network expects.
"""
function make_dataset(total::Int)
    X = rand(Float32, 2, total) .- 0.5f0
    y = Float32.((X[1, :] .> 0) .== (X[2, :] .> 0))
    return X, reshape(y, 1, total)
end

"""
Split `(X, y)` into consecutive mini-batches of `batch` columns. The final
batch is short when `batch` does not divide the sample count; an epoch is a
full pass either way.
"""
function split_batches(X, y, batch::Int)
    total = size(X, 2)
    return [(X[:, i:min(i + batch - 1, total)], y[:, i:min(i + batch - 1, total)])
            for i in 1:batch:total]
end

to_device(batches) = [(CuMatrix(X), CuMatrix(y)) for (X, y) in batches]

new_gpu_layers() = [Dense(2, HIDDEN, relu, drelu), Dense(HIDDEN, 1, sigmoid, dsigmoid)]
new_cpu_layers() = [DenseCPU(2, HIDDEN, relu, drelu), DenseCPU(HIDDEN, 1, sigmoid, dsigmoid)]

"""
One epoch of the chapter's training loop on the device: forward, loss,
backward, weight update, for every batch. The accumulated cost is returned
rather than discarded so the loss cannot be optimized away.
"""
function gpu_epoch!(layers, batches)
    cost = 0.0f0
    for (X, y) in batches
        Ŷ = forward!(layers, X)
        backward!(layers, bce_grad(Ŷ, y), LR)
        cost += bce_loss(Ŷ, y)
    end
    return cost
end

"""
The same epoch with the per-batch loss removed. Nothing in the loop reads a
device scalar, so the whole epoch queues asynchronously and synchronizes once
at the end. Not plotted: it exists to separate the two costs the chapter names
at small batch sizes -- kernel launches, which this still pays, and the
per-step synchronization, which it does not.
"""
function gpu_epoch_nosync!(layers, batches)
    for (X, y) in batches
        Ŷ = forward!(layers, X)
        backward!(layers, bce_grad(Ŷ, y), LR)
    end
    return nothing
end

function cpu_epoch!(layers, batches)
    cost = 0.0f0
    for (X, y) in batches
        Ŷ = forward_cpu!(layers, X)
        backward_cpu!(layers, bce_grad_cpu(Ŷ, y), LR)
        cost += bce_loss_cpu(Ŷ, y)
    end
    return cost
end

"""
Minimum wall-clock seconds for one call of `f`, with `sync` draining any
asynchronous work before the clock stops. Runs at least `MIN_TRIALS` times and
keeps going until `MIN_SECONDS` have been spent, up to `MAX_TRIALS`.
"""
function best_epoch(f, sync)
    f(); sync()                                  # warm up: compile, allocate, cache
    best, total, trials = Inf, 0.0, 0
    while trials < MAX_TRIALS && (trials < MIN_TRIALS || total < MIN_SECONDS)
        t = @elapsed begin
            f()
            sync()
        end
        best = min(best, t)
        total += t
        trials += 1
    end
    return best, trials
end

# =================================================================================
# Sanity check: the timed loop is a working trainer
# =================================================================================
# A benchmark of a network that does not learn would still produce a smooth
# curve, so check the arithmetic before trusting the timings. Both
# implementations train briefly on the same data and must classify a held-out
# split well above chance.

function check_learns()
    X, y = make_dataset(20_000)
    Xt, yt = X[:, 16_001:end], y[:, 16_001:end]
    train = split_batches(X[:, 1:16_000], y[:, 1:16_000], 256)

    cpu_layers = new_cpu_layers()
    for _ in 1:40
        cpu_epoch!(cpu_layers, train)
    end
    acc_cpu = sum((forward_cpu!(cpu_layers, Xt) .> 0.5f0) .== (yt .> 0.5f0)) / size(yt, 2)

    gpu_layers = new_gpu_layers()
    dtrain = to_device(train)
    for _ in 1:40
        gpu_epoch!(gpu_layers, dtrain)
    end
    pred = Array(forward!(gpu_layers, CuMatrix(Xt)))
    acc_gpu = sum((pred .> 0.5f0) .== (yt .> 0.5f0)) / size(yt, 2)

    @printf("  held-out accuracy after 40 epochs: CPU %.3f, GPU %.3f\n", acc_cpu, acc_gpu)
    (acc_cpu > 0.85 && acc_gpu > 0.85) ||
        error("training does not converge (CPU $acc_cpu, GPU $acc_gpu); timings would be meaningless")
    return nothing
end

# =================================================================================
# The sweep
# =================================================================================

"""
Batch size at which the GPU curve crosses the CPU curve, interpolated in
log-log space between the bracketing measurements. Returns `nothing` when the
sweep does not contain a crossing.
"""
function crossover(batches, cpu_t, gpu_t)
    for i in 2:length(batches)
        if gpu_t[i-1] > cpu_t[i-1] && gpu_t[i] <= cpu_t[i]
            d1 = log(gpu_t[i-1]) - log(cpu_t[i-1])
            d2 = log(gpu_t[i]) - log(cpu_t[i])
            f = d1 / (d1 - d2)                   # fraction of the log-x interval
            return exp(log(batches[i-1]) + f * (log(batches[i]) - log(batches[i-1])))
        end
    end
    return nothing
end

"""
Print, for a few batch sizes, how much of the GPU epoch survives when the
per-batch loss (and with it the per-step synchronization) is dropped. Reported
to stdout only -- the figure keeps the loss, because the chapter's loop does.
"""
function report_sync_share(X, y)
    for bs in (32, 256, 2048, 16384)
        dev = to_device(split_batches(X, y, bs))
        t_full, _ = best_epoch(() -> gpu_epoch!(new_gpu_layers(), dev), CUDA.synchronize)
        t_bare, _ = best_epoch(() -> gpu_epoch_nosync!(new_gpu_layers(), dev), CUDA.synchronize)
        @printf("  batch %6d: with loss %8.4f s, without %8.4f s -> per-step sync is %4.1f%% of the epoch\n",
                bs, t_full, t_bare, 100 * (t_full - t_bare) / t_full)
        foreach(b -> (CUDA.unsafe_free!(b[1]); CUDA.unsafe_free!(b[2])), dev)
        GC.gc()
    end
    return nothing
end

"""
Count the device-side operations one training step issues, at both ends and the
middle of the sweep. The count barely moves with the batch size, which is what
makes this a *fixed* per-step cost and what the chapter quotes; reported to
stdout only. Needs CUPTI, so it degrades to a note if profiling is unavailable.
"""
function report_step_operations(X, y)
    for bs in (32, 1024, 16384)
        layers = new_gpu_layers()
        Xb, yb = CuMatrix(X[:, 1:bs]), CuMatrix(y[:, 1:bs])
        step = () -> (Ŷ = forward!(layers, Xb);
                      backward!(layers, bce_grad(Ŷ, yb), LR);
                      bce_loss(Ŷ, yb))
        step(); CUDA.synchronize()                    # warm up: compile
        try
            res = CUDA.@profile trace=true step()
            names = res.device.name
            copies = count(n -> occursin("copy", n), names)
            @printf("  batch %6d: %3d device operations per step = %3d kernel launches + %3d transfers\n",
                    bs, length(names), length(names) - copies, copies)
        catch err
            println("  profiling unavailable ($(sprint(showerror, err)))")
            break
        end
    end
    return nothing
end

function bench_batch_scaling(X, y)
    cpu_t, gpu_t = Float64[], Float64[]

    for bs in BATCH_SIZES
        host = split_batches(X, y, bs)
        dev  = to_device(host)
        steps = length(host)

        tc, nc = best_epoch(() -> cpu_epoch!(new_cpu_layers(), host), () -> nothing)
        tg, ng = best_epoch(() -> gpu_epoch!(new_gpu_layers(), dev), CUDA.synchronize)
        push!(cpu_t, tc); push!(gpu_t, tg)

        @printf("  batch %6d (%5d steps/epoch): CPU %8.4f s (%2d) %9.0f samples/s  |  GPU %8.4f s (%2d) %9.0f samples/s\n",
                bs, steps, tc, nc, TOTAL_SAMPLES / tc, tg, ng, TOTAL_SAMPLES / tg)

        foreach(b -> (CUDA.unsafe_free!(b[1]); CUDA.unsafe_free!(b[2])), dev)
        GC.gc()
    end

    return cpu_t, gpu_t
end

# =================================================================================

function environment_lines()
    dev = CUDA.device()
    cpu = Sys.cpu_info()
    return ["device=$(CUDA.name(dev))",
            "cpu=$(strip(cpu[1].model)) ($(length(cpu)) logical cores)",
            "blas=$(BLAS.get_config().loaded_libs[1].libname |> basename)",
            "blas_threads=$(BLAS.get_num_threads())",
            "julia_threads=$(Threads.nthreads())",
            "julia=$(VERSION)",
            "cuda_jl=$(pkgversion(CUDA))",
            "cuda_runtime=$(CUDA.runtime_version())",
            "cuda_driver=$(CUDA.driver_version())"]
end

function main()
    CUDA.functional() || error("no functional CUDA device")
    BLAS.set_num_threads(1)                      # single-thread CPU baseline
    Random.seed!(SEED)
    CUDA.seed!(SEED)

    env = environment_lines()
    foreach(println, env)
    println()

    X, y = make_dataset(TOTAL_SAMPLES)

    println("[1/4] convergence check"); check_learns(); println()
    println("[2/4] batch-size sweep ($(TOTAL_SAMPLES) samples per epoch)")
    cpu_t, gpu_t = bench_batch_scaling(X, y)
    println()
    println("[3/4] cost of the per-step synchronization (diagnostic, not plotted)")
    report_sync_share(X, y)
    println()
    println("[4/4] device operations per training step (diagnostic, not plotted)")
    report_step_operations(X, y)
    println()

    xover = crossover(BATCH_SIZES, cpu_t, gpu_t)
    if xover === nothing
        println("no CPU/GPU crossover inside the swept range")
    else
        @printf("crossover at batch %.0f (update the dashed line in nn-batch-scaling-time.yaml)\n",
                xover)
    end

    # A companion-repo clone has no figures/data/ until something writes one.
    mkpath(OUT_DIR)

    path = joinpath(OUT_DIR, "nn-batch-scaling.dat")
    open(path, "w") do io
        println(io, "# CPU vs GPU training scaling for the Chapter 12 quadrant MLP (2 -> $HIDDEN -> 1,")
        println(io, "# ReLU + sigmoid), one epoch = one full pass over $TOTAL_SAMPLES samples.")
        println(io, "# Generated by figures/scripts/bench_nn_batch_scaling.jl; see its header for")
        println(io, "# what is inside the clock. Minimum over >= $MIN_TRIALS timed epochs.")
        for line in env
            println(io, "# ", line)
        end
        xover === nothing || @printf(io, "# crossover_batch=%.0f\n", xover)
        println(io, "# batch  cpu_s_per_epoch  gpu_s_per_epoch  cpu_samples_per_s  gpu_samples_per_s")
        for (i, bs) in enumerate(BATCH_SIZES)
            @printf(io, "%6d  %15.6e  %15.6e  %17.4e  %17.4e\n",
                    bs, cpu_t[i], gpu_t[i], TOTAL_SAMPLES / cpu_t[i], TOTAL_SAMPLES / gpu_t[i])
        end
    end
    println("wrote $path")

    stamp = joinpath(OUT_DIR, "nn-batch-environment.txt")
    open(stamp, "w") do io
        foreach(l -> println(io, l), env)
        println(io, "total_samples=$TOTAL_SAMPLES")
        xover === nothing || @printf(io, "crossover_batch=%.0f\n", xover)
    end
    println("wrote $stamp")
    println("done")
end

main()
