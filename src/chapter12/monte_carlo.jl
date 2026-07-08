# Monte Carlo integration examples for estimating π

# --- begin:mc_array ---
using CUDA

function monte_carlo_pi_array(N::Int)
    x = CUDA.rand(Float32, N)
    y = CUDA.rand(Float32, N)
    inside = sum(x.^2 .+ y.^2 .<= 1.0f0)
    return 4.0f0 * Float32(inside) / Float32(N)
end

pi_estimate = monte_carlo_pi_array(100_000_000)
println("π ≈ $pi_estimate")
# --- end:mc_array ---

# --- begin:mc_kernel ---
using CUDA

function mc_pi_kernel!(count, N)
    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    local_count = Int32(0)

    i = tid
    while i <= N
        x = rand(Float32)
        y = rand(Float32)
        if x * x + y * y <= 1.0f0
            local_count += Int32(1)
        end
        i += stride
    end

    CUDA.@atomic count[1] += local_count
    return nothing
end

function monte_carlo_pi_kernel(N::Int)
    d_count = CUDA.zeros(Int32, 1)
    threads = 256
    # Cap the grid at a small multiple of the SM count so each thread
    # accumulates many samples before its single atomic update
    sms = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    blocks = min(cld(N, threads), 32 * sms)
    @cuda threads=threads blocks=blocks mc_pi_kernel!(d_count, N)
    return 4.0f0 * Float32(Array(d_count)[1]) / Float32(N)
end
# --- end:mc_kernel ---
