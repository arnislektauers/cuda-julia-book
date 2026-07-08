# Single-node multi-GPU programming patterns

using CUDA

# --- begin:distribute_vector ---
using CUDA

function distribute_vector(host_data::Vector{T}, ngpus::Int) where T
    N = length(host_data)
    chunk_size = cld(N, ngpus)
    gpu_chunks = Vector{CuVector{T}}(undef, ngpus)

    for i in 1:ngpus
        CUDA.device!(i - 1)
        first = (i - 1) * chunk_size + 1
        last  = min(i * chunk_size, N)
        gpu_chunks[i] = CuArray(host_data[first:last])
    end

    return gpu_chunks
end
# --- end:distribute_vector ---

# --- begin:multi_gpu_map ---
using CUDA

function multi_gpu_map!(f, results, inputs)
    ngpus = length(inputs)

    @sync for i in 1:ngpus
        @async begin
            CUDA.device!(i - 1)            # Set device for this task
            results[i] = f(inputs[i])      # Execute on assigned GPU
            CUDA.synchronize()             # Wait for GPU work to finish
        end
    end

    return results
end
# --- end:multi_gpu_map ---

# --- begin:vector_add_multi_gpu ---
using CUDA

function vadd_kernel!(c, a, b)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(a)
        @inbounds c[i] = a[i] + b[i]
    end
    return nothing
end

function multi_gpu_vadd(a_host::Vector{Float32}, b_host::Vector{Float32})
    ngpus = length(CUDA.devices())
    N = length(a_host)
    chunk_size = cld(N, ngpus)

    # Distribute data
    a_chunks = Vector{CuVector{Float32}}(undef, ngpus)
    b_chunks = Vector{CuVector{Float32}}(undef, ngpus)
    c_chunks = Vector{CuVector{Float32}}(undef, ngpus)

    for i in 1:ngpus
        CUDA.device!(i - 1)
        first = (i - 1) * chunk_size + 1
        last  = min(i * chunk_size, N)
        a_chunks[i] = CuArray(a_host[first:last])
        b_chunks[i] = CuArray(b_host[first:last])
        c_chunks[i] = CUDA.zeros(Float32, last - first + 1)
    end

    # Launch kernels concurrently on all GPUs
    @sync for i in 1:ngpus
        @async begin
            CUDA.device!(i - 1)
            n = length(a_chunks[i])
            threads = 256
            blocks = cld(n, threads)
            @cuda threads=threads blocks=blocks vadd_kernel!(
                c_chunks[i], a_chunks[i], b_chunks[i]
            )
            CUDA.synchronize()
        end
    end

    # Gather results
    c_host = Vector{Float32}(undef, N)
    for i in 1:ngpus
        CUDA.device!(i - 1)
        first = (i - 1) * chunk_size + 1
        last  = min(i * chunk_size, N)
        copyto!(c_host, first, c_chunks[i], 1, last - first + 1)
    end

    return c_host
end

# Usage
N = 10_000_000
a = rand(Float32, N)
b = rand(Float32, N)
c = multi_gpu_vadd(a, b)
@assert c ≈ a .+ b
# --- end:vector_add_multi_gpu ---

CUDA.device!(0)
compute_stream = CuStream()
transfer_stream = CuStream()

# Launch computation on one stream
CUDA.stream!(compute_stream) do
    @cuda threads=256 blocks=1024 compute_kernel!(d_data)
end

# Simultaneously begin a transfer on another stream
CUDA.stream!(transfer_stream) do
    copyto!(d_remote_buf, d_prev_result)
end

# Wait for both
CUDA.synchronize(compute_stream)
CUDA.synchronize(transfer_stream)

# --- begin:overlap_computation ---
using CUDA

# Streams and both buffer sets are created once, outside the loop
compute_stream = CuStream()
comm_stream = CuStream()

N = 1_000_000
inputs = (CUDA.rand(Float32, N), CUDA.rand(Float32, N))
results = (CUDA.zeros(Float32, N), CUDA.zeros(Float32, N))
remote_buf = CUDA.zeros(Float32, N)    # Destination on the peer device

nsteps = 100
for k in 1:nsteps
    cur = 1 + k % 2          # Buffer half computed in this iteration
    prev = 1 + (k + 1) % 2   # Half holding the previous result

    # Compute step k on the compute stream ...
    CUDA.stream!(compute_stream) do
        @cuda threads=256 blocks=cld(N, 256) compute_kernel!(
            results[cur], inputs[cur]
        )
    end

    # ... while the result of step k - 1 travels on the comm stream
    if k > 1
        CUDA.stream!(comm_stream) do
            copyto!(remote_buf, results[prev])
        end
    end

    # Both streams must drain before the halves swap roles
    CUDA.synchronize(compute_stream)
    CUDA.synchronize(comm_stream)
end

# Transfer the final result and wait for all outstanding work
copyto!(remote_buf, results[1 + nsteps % 2])
CUDA.synchronize()
# --- end:overlap_computation ---

# --- begin:measure_throughputs ---
function measure_throughputs()
    throughputs = map(CUDA.devices()) do dev
        CUDA.device!(dev)
        a = CUDA.rand(Float32, 100_000)
        a .= a .* 2.0f0 .+ 1.0f0    # Untimed warmup: exclude kernel compilation
        elapsed = CUDA.@elapsed for _ in 1:1000
            a .= a .* 2.0f0 .+ 1.0f0
        end
        return 1.0 / elapsed
    end
    total = sum(throughputs)
    return throughputs ./ total    # Normalized fractions
end
# --- end:measure_throughputs ---
