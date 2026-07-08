# Pipeline parallelism: overlapping transfers with computation

using CUDA

# --- begin:pipeline_computation ---
function pipelined_computation!(d_result, h_data, h_result, chunk_size)
    N = length(h_data)
    nchunks = cld(N, chunk_size)
    streams = [CuStream() for _ in 1:3]

    # Pin host buffers so copies are truly asynchronous and can overlap
    CUDA.pin(h_data)
    CUDA.pin(h_result)

    d_chunk = [CuArray{Float32}(undef, chunk_size) for _ in 1:3]

    for c in 1:nchunks
        s = mod1(c, 3)
        offset = (c - 1) * chunk_size + 1
        n = min(chunk_size, N - offset + 1)
        rng = offset:offset+n-1

        # All operations inside run on this chunk's stream
        CUDA.stream!(streams[s]) do
            # Async copy H->D
            copyto!(view(d_chunk[s], 1:n), view(h_data, rng))

            # Kernel on same stream (waits for copy to complete)
            @cuda threads=256 blocks=cld(n, 256) process_kernel!(
                view(d_result, rng), view(d_chunk[s], 1:n))

            # Async copy D->H
            copyto!(view(h_result, rng), view(d_result, rng))
        end
    end

    # Wait for all streams
    for s in streams
        CUDA.synchronize(s)
    end
end
# --- end:pipeline_computation ---
