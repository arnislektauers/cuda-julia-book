# Distributed.jl with GPU arrays

using Distributed, CUDA

# --- begin:distributed_gpu_setup ---
using Distributed, CUDA

# Spawn one worker per GPU
addprocs(length(CUDA.devices()))
@everywhere using CUDA

# Assign each worker a unique GPU
asyncmap(zip(workers(), CUDA.devices())) do (p, dev)
    remotecall_wait(p) do
        CUDA.device!(dev)
        println("Worker $p -> $(CUDA.name(CUDA.device()))")
    end
end
# --- end:distributed_gpu_setup ---

# Not part of either listing: each tagged region above and below is a complete,
# standalone example that spawns one worker per GPU. Run end to end as one file,
# they would spawn two workers per GPU -- and because the cluster's GPUs are in
# Exclusive_Process mode, the second set cannot retain a context on a device the
# first set already holds, and dies with
#   CUDA error: CUDA-capable device(s) is/are busy or unavailable (code 46)
# Releasing the first set here lets the file run as the sum of its listings.
rmprocs(workers())

# --- begin:monte_carlo_distributed ---
using Distributed, CUDA

addprocs(length(CUDA.devices()))
@everywhere using CUDA, Random

# Assign GPUs to workers
for (p, dev) in zip(workers(), CUDA.devices())
    remotecall_wait(p) do
        CUDA.device!(dev)
    end
end

function estimate_pi(N_total::Int)
    ngpus = nworkers()
    N_per_gpu = cld(N_total, ngpus)

    # Each worker estimates π on its GPU
    futures = map(workers()) do p
        remotecall(p) do
            n = N_per_gpu
            x = CUDA.rand(Float32, n)
            y = CUDA.rand(Float32, n)
            inside = sum(x .^ 2 .+ y .^ 2 .<= 1.0f0)
            return (inside, n)
        end
    end

    # Gather and combine results
    total_inside = 0
    total_n = 0
    for f in futures
        inside, n = fetch(f)
        total_inside += inside
        total_n += n
    end

    return 4.0 * total_inside / total_n
end

π_est = estimate_pi(100_000_000)
println("π ≈ $π_est")
# --- end:monte_carlo_distributed ---
