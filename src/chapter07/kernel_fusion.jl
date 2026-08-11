# Kernel fusion and dynamic work distribution

using CUDA

# --- begin:kernel_fusion ---
# Split: two kernels, intermediate allocation
function step1!(B, A)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(A)
        @inbounds B[i] = sqrt(A[i])
    end
    return nothing
end

function step2!(C, B)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(B)
        @inbounds C[i] = B[i] * 2.0f0 + 1.0f0
    end
    return nothing
end

# Fused: single kernel, no intermediate allocation
function fused_step!(C, A)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(A)
        @inbounds C[i] = sqrt(A[i]) * 2.0f0 + 1.0f0
    end
    return nothing
end
# --- end:kernel_fusion ---

# --- begin:dynamic_work ---
function dynamic_work!(output, input, counter)
    chunk_shared = CuStaticSharedArray(Int32, 1)
    while true
        # Claim next work item atomically
        if threadIdx().x == 1
            chunk = CUDA.atomic_add!(pointer(counter, 1), Int32(1))
            chunk_shared[1] = chunk
        end
        sync_threads()

        chunk_id = chunk_shared[1]
        chunk_id >= cld(length(input), blockDim().x) && return nothing

        # Process the claimed chunk
        i = chunk_id * blockDim().x + threadIdx().x
        if i <= length(input)
            @inbounds output[i] = expensive_computation(input[i])
        end
        sync_threads()
    end
    return nothing
end
# --- end:dynamic_work ---

# ---------------------------------------------------------------------------
# Driver, outside the tagged region so the book is unaffected.
#
# expensive_computation is the placeholder the dynamic_work listing calls and
# never defines; supplying it here is what makes the file runnable at all.
@inline expensive_computation(x) = sqrt(x) * 2.0f0 + 1.0f0

let N = 1 << 16
    A = CUDA.rand(Float32, N)
    a = Array(A)
    want = sqrt.(a) .* 2.0f0 .+ 1.0f0

    # Fusion's claim is identical output from one kernel instead of two.
    B = CUDA.zeros(Float32, N)
    C1 = CUDA.zeros(Float32, N)
    C2 = CUDA.zeros(Float32, N)
    @cuda threads=256 blocks=cld(N, 256) step1!(B, A)
    @cuda threads=256 blocks=cld(N, 256) step2!(C1, B)
    @cuda threads=256 blocks=cld(N, 256) fused_step!(C2, A)
    CUDA.synchronize()
    @assert Array(C1) == Array(C2) "fused kernel differs from the split pair"
    @assert Array(C2) == want "neither matches sqrt(x)*2+1"

    # Dynamic work distribution: fewer blocks than chunks, so blocks must come
    # back for more work. Every element still has to be written exactly once --
    # a claim race would leave gaps or double-process, and only checking the
    # whole output catches that.
    out = CUDA.fill(-1.0f0, N)
    counter = CUDA.zeros(Int32, 1)
    @cuda threads=256 blocks=8 dynamic_work!(out, A, counter)
    CUDA.synchronize()
    @assert Array(out) == want "dynamic work queue left gaps or wrong values"
    println("kernel fusion and dynamic_work!: CORRECT")
end
