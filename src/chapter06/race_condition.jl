# Race condition example: missing sync_threads() between write and read

using CUDA

# --- begin:racy_kernel ---
# BUG: missing sync_threads() between write and read
function racy_kernel!(out, A)
    sdata = CuDynamicSharedArray(Float32, blockDim().x)
    tid = threadIdx().x
    sdata[tid] = A[tid]
    # sync_threads()  <- MISSING! Other threads may not have written yet
    if tid > 1
        @inbounds out[tid] = sdata[tid] + sdata[tid - 1]
    end
    return nothing
end
# --- end:racy_kernel ---
