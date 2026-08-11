# Synchronization example: staged kernel using sync_threads()

using CUDA

# --- begin:staged_kernel ---
function staged_kernel!(out, A)
    tid = threadIdx().x
    i = (blockIdx().x - 1) * blockDim().x + tid
    shmem = CuStaticSharedArray(Float32, 256)

    # Phase 1: each thread writes its element to shared memory
    if i <= length(A)
        @inbounds shmem[tid] = A[i]
    end

    sync_threads()  # barrier: all Phase 1 writes are now visible

    # Phase 2: read a neighbor's slot (safe only because of the barrier)
    if i <= length(A)
        @inbounds begin
            # first thread of the block uses its own value
            left = tid > 1 ? shmem[tid - 1] : shmem[tid]
            out[i] = shmem[tid] + left
        end
    end
    return nothing
end
# --- end:staged_kernel ---

# ---------------------------------------------------------------------------
# Driver, outside the tagged region so the book is unaffected. The kernel's
# shared array is statically sized at 256, so the block size must match it --
# launching with more threads would write past the end.
#
# out[i] = A[i] + A[i-1], except at each block's first thread, which doubles
# its own value. Checking that boundary matters: it is the case the barrier
# and the `tid > 1` guard exist for.
let N = 1024, threads = 256
    A = CuArray(Float32.(1:N))
    out = CUDA.zeros(Float32, N)
    @cuda threads=threads blocks=cld(N, threads) staged_kernel!(out, A)
    CUDA.synchronize()
    a, got = Array(A), Array(out)
    want = [(i - 1) % threads == 0 ? 2a[i] : a[i] + a[i-1] for i in 1:N]
    @assert got == want "staged kernel: first mismatch at $(findfirst(got .!= want))"
    println("staged_kernel!: neighbour sums and block boundaries CORRECT")
end
