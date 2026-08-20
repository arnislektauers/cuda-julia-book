# Measure the CPU-vs-GPU timings printed in Chapter 11's data-science table
# (tbl-gpu-vs-cpu-analytics).
#
# Two workloads at four dataset sizes:
#
#   1. Z-score normalization, the single reduction-plus-broadcast of
#      sec-dataframes-gpu.
#   2. The four-feature engineering pipeline of lst-gpu-feature-pipeline,
#      reproduced verbatim from src/chapter11/data_science.jl for the GPU and
#      written as the equivalent host code for the CPU baseline.
#
# GPU times include the host-device transfers, because that is what the table
# claims: the input starts in host memory (a DataFrame column) and the result
# is read back. Timing only the kernels would flatter the GPU and move the
# crossover to the left.
#
# The CPU baseline is single-threaded, matching the broadcast code the chapter
# prints; the thread count is reported so the number can be read for what it is.
#
# Run on the reference machine (Appendix A: RTX 4070 SUPER, Julia 1.12.6,
# CUDA.jl 6.2.1), clearing LD_LIBRARY_PATH first:
#
#   LD_LIBRARY_PATH= julia --project=<env-with-CUDA> bench_ch11_analytics.jl
#
# Prints a markdown table ready to paste into the chapter, plus the environment
# line that belongs in the caption.

using CUDA
using DataFrames
using Statistics
using Printf

const SIZES = [100_000, 1_000_000, 10_000_000, 100_000_000]

# Trials per measurement. The minimum is the right statistic here for the same
# reason it is in bench_chapter_figures.jl: contention only ever adds time.
trials_for(n) = n >= 100_000_000 ? 3 : (n >= 10_000_000 ? 5 : 10)

"""
Minimum wall-clock seconds over `trials` runs of `f`, with the GPU drained
before the clock stops so asynchronous work is not billed to the next trial.
"""
function best_wall(f; trials)
    f(); CUDA.synchronize()                     # warm up: compile, cache, pool
    best = Inf
    for _ in 1:trials
        t = @elapsed begin
            f()
            CUDA.synchronize()
        end
        best = min(best, t)
    end
    return best
end

# --- workload 1: z-score normalization -------------------------------------------

zscore_cpu(v) = (v .- mean(v)) ./ std(v)

function zscore_gpu(v)
    d = CuArray(v)                              # host -> device
    μ = mean(d); σ = std(d)
    return Array((d .- μ) ./ σ)                 # device -> host
end

# The same arithmetic on data that is already on the device and stays there.
function zscore_gpu_resident(d)
    μ = mean(d); σ = std(d)
    return (d .- μ) ./ σ
end

# --- workload 2: the chapter's feature pipeline -----------------------------------

# Verbatim from src/chapter11/data_science.jl (lst-gpu-feature-pipeline).
function gpu_feature_pipeline(df::DataFrame)
    d_values = CuArray(Float32.(df.reading))
    d_time   = CuArray(Float32.(df.timestamp))

    μ = mean(d_values)
    σ = std(d_values)
    d_zscore = (d_values .- μ) ./ σ

    d_sq_dev = (d_values .- μ) .^ 2

    d_time_norm = d_time ./ maximum(d_time)
    d_weighted  = d_values .* d_time_norm

    d_log_values = log.(max.(d_values, 1.0f-7))

    df.zscore     = Array(d_zscore)
    df.sq_dev     = Array(d_sq_dev)
    df.weighted   = Array(d_weighted)
    df.log_values = Array(d_log_values)

    return df
end

# The pipeline with both ends already on the device: no transfer, no writeback.
function gpu_feature_pipeline_resident(d_values, d_time)
    μ = mean(d_values)
    σ = std(d_values)
    d_zscore     = (d_values .- μ) ./ σ
    d_sq_dev     = (d_values .- μ) .^ 2
    d_weighted   = d_values .* (d_time ./ maximum(d_time))
    d_log_values = log.(max.(d_values, 1.0f-7))
    return d_zscore, d_sq_dev, d_weighted, d_log_values
end

# The same four features on the host, expression for expression.
function cpu_feature_pipeline(df::DataFrame)
    values = Float32.(df.reading)
    time   = Float32.(df.timestamp)

    μ = mean(values)
    σ = std(values)

    df.zscore     = (values .- μ) ./ σ
    df.sq_dev     = (values .- μ) .^ 2
    df.weighted   = values .* (time ./ maximum(time))
    df.log_values = log.(max.(values, 1.0f-7))

    return df
end

# --- driver -----------------------------------------------------------------------

fmt(t) = t >= 1 ? @sprintf("%.2f s", t) : @sprintf("%.2f ms", 1000t)

function main()
    println("# Julia $(VERSION), $(Threads.nthreads()) thread(s), CUDA.jl $(pkgversion(CUDA))")
    println("# GPU: $(CUDA.name(CUDA.device()))   CPU: $(Sys.CPU_NAME) x$(Sys.CPU_THREADS)")
    println()

    rows = Dict{String,Vector{Float64}}(
        "zscore_cpu"   => Float64[], "zscore_gpu"   => Float64[],
        "zscore_res"   => Float64[], "pipeline_res" => Float64[],
        "pipeline_cpu" => Float64[], "pipeline_gpu" => Float64[],
    )

    for n in SIZES
        tr = trials_for(n)
        v  = rand(Float32, n)

        push!(rows["zscore_cpu"], best_wall(() -> zscore_cpu(v);  trials=tr))
        push!(rows["zscore_gpu"], best_wall(() -> zscore_gpu(v);  trials=tr))

        # A fresh frame per size; the pipeline writes four columns back into it.
        df = DataFrame(reading=v, timestamp=collect(Float32, 1:n))
        push!(rows["pipeline_cpu"], best_wall(() -> cpu_feature_pipeline(df); trials=tr))
        push!(rows["pipeline_gpu"], best_wall(() -> gpu_feature_pipeline(df); trials=tr))

        d_values = CuArray(v)
        d_time   = CuArray(collect(Float32, 1:n))
        push!(rows["zscore_res"],   best_wall(() -> zscore_gpu_resident(d_values); trials=tr))
        push!(rows["pipeline_res"], best_wall(() -> gpu_feature_pipeline_resident(d_values, d_time); trials=tr))

        df = nothing; v = nothing; d_values = nothing; d_time = nothing
        GC.gc(); CUDA.reclaim()
        println("# done n = $n (trials = $tr)")
    end

    println()
    sizelabel(n) = n >= 1_000_000 ? "$(n ÷ 1_000_000)M rows" : "$(n ÷ 1000)K rows"
    println("| Operation | " * join(sizelabel.(SIZES), " | ") * " |")
    println("|:----------|" * repeat(":---------|", length(SIZES)))
    for (label, key) in (("Z-score normalization (CPU)", "zscore_cpu"),
                         ("Z-score normalization (GPU, with transfer)", "zscore_gpu"),
                         ("Z-score normalization (GPU, data resident)", "zscore_res"),
                         ("Feature pipeline (CPU)",      "pipeline_cpu"),
                         ("Feature pipeline (GPU, with transfer)", "pipeline_gpu"),
                         ("Feature pipeline (GPU, data resident)",  "pipeline_res"))
        println("| $label | " * join(fmt.(rows[key]), " | ") * " |")
    end

    println()
    for (name, ck, gk) in (("z-score  (with transfer)", "zscore_cpu", "zscore_gpu"),
                           ("z-score  (resident)     ", "zscore_cpu", "zscore_res"),
                           ("pipeline (with transfer)", "pipeline_cpu", "pipeline_gpu"),
                           ("pipeline (resident)     ", "pipeline_cpu", "pipeline_res"))
        ratios = rows[ck] ./ rows[gk]
        println("# $name CPU/GPU ratio by size: ",
                join((@sprintf("%.2f", r) for r in ratios), "  "))
    end
end

main()
