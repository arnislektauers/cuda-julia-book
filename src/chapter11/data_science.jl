# Data science and analytics -- GPU-accelerated DataFrame operations

using CUDA, DataFrames, Statistics

# --- begin:gpu_column_ops ---
using CUDA, DataFrames, Statistics

# Load data (CPU)
df = DataFrame(
    sensor_id = repeat(1:1000, inner=10_000),
    timestamp = rand(Float64, 10_000_000),
    reading   = rand(Float32, 10_000_000),
    quality   = rand(Float32, 10_000_000)
)

# Transfer columns to GPU for computation
d_readings = CuArray(df.reading)
d_quality  = CuArray(df.quality)

# GPU-accelerated transformations
d_normalized = (d_readings .- mean(d_readings)) ./ std(d_readings)
d_filtered   = d_normalized .* (d_quality .> 0.5f0)

# Collect results back to CPU and add to DataFrame
df.normalized = Array(d_normalized)
df.filtered   = Array(d_filtered)
# --- end:gpu_column_ops ---

# --- begin:gpu_group_stats ---
using CUDA, DataFrames

function gpu_group_statistics(df::DataFrame, group_col::Symbol,
                              value_col::Symbol)
    # Remap group values to contiguous indices 1..K so the kernel's
    # accumulators align with the `groups` vector in the result
    groups = sort(unique(df[!, group_col]))
    n_groups = length(groups)
    group_ids = Int32.(indexin(df[!, group_col], groups))

    # Transfer value column and group indices to GPU
    d_values = CuArray(Float32.(df[!, value_col]))
    d_groups = CuArray(group_ids)

    # Compute per-group statistics on GPU using custom kernel
    d_sums   = CUDA.zeros(Float32, n_groups)
    d_counts = CUDA.zeros(Int32, n_groups)

    function group_reduce_kernel!(sums, counts, values, groups)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        if i <= length(values)
            g = groups[i]
            CUDA.@atomic sums[g] += values[i]
            CUDA.@atomic counts[g] += Int32(1)
        end
        return nothing
    end

    threads = 256
    blocks = cld(length(d_values), threads)
    @cuda threads=threads blocks=blocks group_reduce_kernel!(
        d_sums, d_counts, d_values, d_groups
    )

    # Collect and build result DataFrame
    sums   = Array(d_sums)
    counts = Array(d_counts)
    return DataFrame(
        group = groups,
        mean  = sums ./ counts,
        count = counts
    )
end
# --- end:gpu_group_stats ---

# --- begin:gpu_feature_pipeline ---
using CUDA, DataFrames, Statistics

function gpu_feature_pipeline(df::DataFrame)
    n = nrow(df)

    # Transfer numerical columns to GPU
    d_values = CuArray(Float32.(df.reading))
    d_time   = CuArray(Float32.(df.timestamp))

    # Feature 1: Z-score normalization
    μ = mean(d_values)
    σ = std(d_values)
    d_zscore = (d_values .- μ) ./ σ

    # Feature 2: Squared deviation (useful for anomaly detection)
    d_sq_dev = (d_values .- μ) .^ 2

    # Feature 3: Time-weighted values
    d_time_norm = d_time ./ maximum(d_time)
    d_weighted  = d_values .* d_time_norm

    # Feature 4: Log-transform (with safety clamp)
    d_log_values = log.(max.(d_values, 1.0f-7))

    # Collect all features
    df.zscore     = Array(d_zscore)
    df.sq_dev     = Array(d_sq_dev)
    df.weighted   = Array(d_weighted)
    df.log_values = Array(d_log_values)

    return df
end
# --- end:gpu_feature_pipeline ---
