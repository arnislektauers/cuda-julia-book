# Sine integral approximation using the trapezoidal rule on the GPU

# --- begin:trapezoidal ---
using CUDA

function trapezoidal_kernel!(results, a, h, n)
    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    local_sum = 0.0f0
    i = tid
    while i <= n - 1
        x = a + i * h
        local_sum += sin(x)  # Integrand: f(x) = sin(x)
        i += stride
    end

    CUDA.@atomic results[1] += local_sum
    return nothing
end

function trapezoidal_gpu(a::Float32, b::Float32, n::Int)
    h = (b - a) / n
    d_result = CUDA.zeros(Float32, 1)

    threads = 256
    # Cap the grid at a small multiple of the SM count so each thread
    # accumulates many elements before its single atomic update
    sms = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    blocks = min(cld(n, threads), 32 * sms)
    @cuda threads=threads blocks=blocks trapezoidal_kernel!(
        d_result, a, h, n)

    interior_sum = Array(d_result)[1]

    # Add endpoint contributions on host
    return h * (sin(a) / 2 + interior_sum + sin(b) / 2)
end

# Approximate integral of sin(x) on [0, pi] = 2.0
result = trapezoidal_gpu(0.0f0, Float32(π), 10_000_000)
println("Integral = $result (exact: 2.0)")
# --- end:trapezoidal ---
