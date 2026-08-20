# Launch configuration tuning: block size sweep and 2D auto-tuning

using CUDA

# --- begin:benchmark_block_sizes ---
function benchmark_block_sizes(kernel!, c, a, b, N)
    for threads in [32, 64, 128, 256, 512, 1024]
        blocks = cld(N, threads)
        # Warm up
        @cuda threads=threads blocks=blocks kernel!(c, a, b)
        CUDA.synchronize()
        # Measure
        t = CUDA.@elapsed begin
            for _ in 1:100
                @cuda threads=threads blocks=blocks kernel!(c, a, b)
            end
        end
        t_per = t / 100
        bw = 3 * N * sizeof(Float32) / t_per / 1e9
        println("threads=$threads: $(round(t_per * 1e6, digits=1)) μs, " *
                "$(round(bw, digits=1)) GB/s")
    end
end
# --- end:benchmark_block_sizes ---

# --- begin:autotune_2d ---
function autotune_2d(kernel!, C, A, B, M, N)
    best_time = Inf
    best_config = (16, 16)

    for tx in [8, 16, 32], ty in [4, 8, 16, 32]
        tx * ty > 1024 && continue
        blocks = (cld(M, tx), cld(N, ty))

        # Warm up
        @cuda threads=(tx, ty) blocks=blocks kernel!(C, A, B)
        CUDA.synchronize()

        t = CUDA.@elapsed begin
            for _ in 1:10
                @cuda threads=(tx, ty) blocks=blocks kernel!(C, A, B)
            end
        end

        if t < best_time
            best_time = t
            best_config = (tx, ty)
        end
    end

    println("Best config: $(best_config) -> " *
            "$(round(best_time * 1e6 / 10, digits=1)) μs/launch")
    return best_config
end
# --- end:autotune_2d ---

# ---------------------------------------------------------------------------
# Driver, outside the tagged region so the book is unaffected. Both listings
# take the kernel as an argument and neither ships one, so the driver supplies
# a 1D and a 2D kernel of the shape each expects.
#
# These are timing sweeps, so there is no reference answer: what can be
# checked is that every configuration in the sweep actually launches (a block
# size the kernel cannot support would throw), that the tuner returns a
# configuration from its own search space, and that the kernel still computes
# the right thing at the chosen size.
function add_kernel!(c, a, b)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(c)
        @inbounds c[i] = a[i] + b[i]
    end
    return nothing
end

function add2d_kernel!(C, A, B)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= size(C, 1) && j <= size(C, 2)
        @inbounds C[i, j] = A[i, j] + B[i, j]
    end
    return nothing
end

let N = 1 << 20, M = 1024, K = 1024
    a, b = CUDA.rand(Float32, N), CUDA.rand(Float32, N)
    c = CUDA.zeros(Float32, N)
    benchmark_block_sizes(add_kernel!, c, a, b, N)
    @assert Array(c) == Array(a) .+ Array(b) "1D kernel wrong after the sweep"

    A, B = CUDA.rand(Float32, M, K), CUDA.rand(Float32, M, K)
    C = CUDA.zeros(Float32, M, K)
    best = autotune_2d(add2d_kernel!, C, A, B, M, K)
    @assert best[1] in (8, 16, 32) && best[2] in (4, 8, 16, 32) "config outside search space: $best"
    @assert best[1] * best[2] <= 1024 "config exceeds the block-size limit: $best"
    @assert Array(C) == Array(A) .+ Array(B) "2D kernel wrong after the sweep"
    println("launch_config: sweep completed, best 2D config $best, results CORRECT")
end
