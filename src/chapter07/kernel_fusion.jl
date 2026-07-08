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
