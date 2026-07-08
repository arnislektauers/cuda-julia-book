# Host-device transfer patterns: pinned memory and async streams

using CUDA

# --- begin:pinned_memory ---
using CUDA

# Allocate a regular host array and pin (page-lock) it
N = 1_000_000
h_pinned = Vector{Float32}(undef, N)
CUDA.pin(h_pinned)

# Fill with data
h_pinned .= rand(Float32, N)

# Transfer to device (uses DMA directly, no staging copy)
d = CuArray{Float32}(undef, N)
copyto!(d, h_pinned)
# --- end:pinned_memory ---

# --- begin:async_streams ---
using CUDA

N = 10_000_000
chunk = cld(N, 3)
streams = [CuStream() for _ in 1:3]

# Host-side pinned buffers (required for async transfers)
h_a = CUDA.pin(rand(Float32, N))
h_b = CUDA.pin(rand(Float32, N))
h_c = CUDA.pin(zeros(Float32, N))

# Device-side buffers
d_a = CuArray{Float32}(undef, N)
d_b = CuArray{Float32}(undef, N)
d_c = CuArray{Float32}(undef, N)

function add_kernel!(c, a, b, offset, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + offset - 1
    if i <= offset + n - 1
        @inbounds c[i] = a[i] + b[i]
    end
    return nothing
end

for (s, stream) in enumerate(streams)
    offset = (s - 1) * chunk + 1
    n = min(chunk, N - offset + 1)

    # All operations inside the block are enqueued on this stream
    CUDA.stream!(stream) do
        # Async copy H->D. The 5-argument copyto!(dest, doffs, src, soffs, n)
        # transfers a contiguous chunk as a single DMA; a view-to-view
        # copyto! has no bulk path here and falls back to slow elementwise
        # (scalar) indexing.
        copyto!(d_a, offset, h_a, offset, n)
        copyto!(d_b, offset, h_b, offset, n)

        # Kernel on the same stream waits for the copies above
        @cuda threads=256 blocks=cld(n, 256) add_kernel!(d_c, d_a, d_b, offset, n)

        # Async copy D->H
        copyto!(h_c, offset, d_c, offset, n)
    end
end

# Wait for all streams to finish
for stream in streams
    CUDA.synchronize(stream)
end
# --- end:async_streams ---
