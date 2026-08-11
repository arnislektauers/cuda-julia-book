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
            # Async copy H->D. The offset form, not copyto! of two views:
            # host views have no fast path and fall back to scalar iteration,
            # which CUDA.jl refuses.
            copyto!(d_chunk[s], 1, h_data, offset, n)

            # Kernel on same stream (waits for copy to complete)
            @cuda threads=256 blocks=cld(n, 256) process_kernel!(
                view(d_result, rng), view(d_chunk[s], 1:n))

            # Async copy D->H
            copyto!(h_result, offset, d_result, offset, n)
        end
    end

    # Wait for all streams
    for s in streams
        CUDA.synchronize(s)
    end
end
# --- end:pipeline_computation ---

# ---------------------------------------------------------------------------
# Driver, outside the tagged region so the book is unaffected.
#
# process_kernel! is the placeholder the listing calls and never defines;
# supplying it here is what makes the file runnable at all.
function process_kernel!(out, in)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(in)
        @inbounds out[i] = in[i] * 2.0f0 + 1.0f0
    end
    return nothing
end

# A chunk count that is not a multiple of the stream count, and a final chunk
# shorter than chunk_size: both are where a pipeline drops or duplicates data,
# and both are invisible if the sizes divide evenly.
let N = 100_000, chunk_size = 8192
    h_data = rand(Float32, N)
    h_result = zeros(Float32, N)
    d_result = CUDA.zeros(Float32, N)
    pipelined_computation!(d_result, h_data, h_result, chunk_size)
    @assert h_result ≈ h_data .* 2.0f0 .+ 1.0f0 "pipeline result differs at $(findfirst(!isapprox.(h_result, h_data .* 2 .+ 1)))"
    println("pipelined_computation!: $(cld(N, chunk_size)) chunks over 3 streams CORRECT")
end
