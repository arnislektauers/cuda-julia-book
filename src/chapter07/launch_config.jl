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
        println("threads=$threads: $(round(t_per * 1e6, digits=1)) μs, $(round(bw, digits=1)) GB/s")
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

    println("Best config: $(best_config) -> $(round(best_time * 1e6 / 10, digits=1)) μs/launch")
    return best_config
end
# --- end:autotune_2d ---
