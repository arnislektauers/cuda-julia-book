# Shared memory programming: static/dynamic allocation and block reduction

using CUDA

# --- begin:static_shmem ---
function static_shmem_kernel!(out, A)
    sdata = CuStaticSharedArray(Float32, 256)  # fixed size

    tid = threadIdx().x
    i = (blockIdx().x - 1) * blockDim().x + tid
    if i <= length(A)
        @inbounds sdata[tid] = A[i]
    end
    sync_threads()

    # Use sdata for computation...
    if i <= length(A)
        @inbounds out[i] = sdata[tid] * 2.0f0
    end
    return nothing
end
# --- end:static_shmem ---

# --- begin:dynamic_shmem ---
function dynamic_shmem_kernel!(out, A)
    sdata = CuDynamicSharedArray(Float32, blockDim().x)

    tid = threadIdx().x
    i = (blockIdx().x - 1) * blockDim().x + tid
    if i <= length(A)
        @inbounds sdata[tid] = A[i]
    end
    sync_threads()

    if i <= length(A)
        @inbounds out[i] = sdata[tid] * 2.0f0
    end
    return nothing
end

N = 1_000_000
A = CUDA.rand(Float32, N)
out = similar(A)

shmem_size = 256 * sizeof(Float32)
@cuda(threads=256, blocks=cld(N, 256), shmem=shmem_size,
      dynamic_shmem_kernel!(out, A))
# --- end:dynamic_shmem ---

# --- begin:block_reduce_sum ---
function block_reduce_sum!(output, input)
    sdata = CuDynamicSharedArray(Float32, blockDim().x)
    tid = threadIdx().x
    i = (blockIdx().x - 1) * blockDim().x + tid

    # Load from global to shared memory
    sdata[tid] = (i <= length(input)) ? @inbounds(input[i]) : 0.0f0
    sync_threads()

    # Tree reduction in shared memory
    # Note: the halving stride requires blockDim().x to be a power of two (256 below)
    stride = blockDim().x ÷ 2
    while stride > 0
        if tid <= stride
            @inbounds sdata[tid] += sdata[tid + stride]
        end
        sync_threads()
        stride ÷= 2
    end

    # Thread 1 writes the block's sum to global memory
    if tid == 1
        @inbounds output[blockIdx().x] = sdata[1]
    end
    return nothing
end

N = 1_000_000
input = CUDA.rand(Float32, N)
threads = 256
blocks = cld(N, threads)
partial_sums = CuArray{Float32}(undef, blocks)

@cuda(threads=threads, blocks=blocks,
      shmem=threads*sizeof(Float32),
      block_reduce_sum!(partial_sums, input))
result = sum(partial_sums)  # final reduction on the partial sums
# --- end:block_reduce_sum ---
