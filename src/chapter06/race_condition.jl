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

# ---------------------------------------------------------------------------
# Driver, outside the tagged region so the book is unaffected.
#
# The listing is deliberately wrong, so "does it produce the right answer" is
# the wrong question: asserting that it MISBEHAVES would be flaky, because
# threads within a warp execute the write in lockstep and the racy read often
# agrees by luck. What is asserted instead is that the corrected kernel is
# right; the racy one is run and its agreement reported, not required.
#
# The shared array is sized blockDim().x at runtime, so the launch must pass
# shmem for that many Float32s.
function fixed_kernel!(out, A)
    sdata = CuDynamicSharedArray(Float32, blockDim().x)
    tid = threadIdx().x
    sdata[tid] = A[tid]
    sync_threads()                       # the line the racy version omits
    if tid > 1
        @inbounds out[tid] = sdata[tid] + sdata[tid - 1]
    end
    return nothing
end

let threads = 256
    A = CuArray(Float32.(1:threads))
    a = Array(A)
    want = [i > 1 ? a[i] + a[i-1] : 0.0f0 for i in 1:threads]

    fixed = CUDA.zeros(Float32, threads)
    @cuda threads=threads blocks=1 shmem=threads*sizeof(Float32) fixed_kernel!(fixed, A)
    CUDA.synchronize()
    @assert Array(fixed) == want "the synchronized kernel is wrong -- that is a real bug"

    racy = CUDA.zeros(Float32, threads)
    @cuda threads=threads blocks=1 shmem=threads*sizeof(Float32) racy_kernel!(racy, A)
    CUDA.synchronize()
    agreed = Array(racy) == want
    println("fixed_kernel!: CORRECT; racy_kernel! ", agreed ?
            "happened to agree this run (the race is benign here, not absent)" :
            "disagreed, as the missing barrier allows")
end
