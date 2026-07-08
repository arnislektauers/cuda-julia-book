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
