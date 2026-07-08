# AMDGPU.jl programming examples

using AMDGPU

# --- begin:amdgpu_arrays ---
using AMDGPU

# Array creation
A = AMDGPU.ones(Float32, 1024, 1024)
B = AMDGPU.rand(Float32, 1024, 1024)

# Transfer from host
C_gpu = ROCArray(rand(Float32, 1024))
C_cpu = Array(C_gpu)

# Broadcasting (dispatches to GPU kernels via GPUArrays.jl)
D = sin.(A) .+ 2f0 .* B
# --- end:amdgpu_arrays ---

# --- begin:amdgpu_vadd ---
function vadd_kernel!(c, a, b)
    i = workitemIdx().x + (workgroupIdx().x - 1) * workgroupDim().x
    if i <= length(a)
        @inbounds c[i] = a[i] + b[i]
    end
    return nothing
end

N = 1024
a = AMDGPU.rand(Float32, N)
b = AMDGPU.rand(Float32, N)
c = ROCArray{Float32}(undef, N)

groupsize = 256
gridsize = cld(N, groupsize)
@roc groupsize=groupsize gridsize=gridsize vadd_kernel!(c, a, b)
AMDGPU.synchronize()
# --- end:amdgpu_vadd ---

# --- begin:amdgpu_reduce ---
function reduce_kernel!(output, input)
    shared = @ROCStaticLocalArray(Float32, 256)

    tid = workitemIdx().x
    i = workitemIdx().x + (workgroupIdx().x - 1) * workgroupDim().x

    @inbounds shared[tid] = i <= length(input) ? input[i] : 0f0
    AMDGPU.sync_workgroup()

    s = workgroupDim().x >> 1
    while s > 0
        if tid <= s
            @inbounds shared[tid] += shared[tid + s]
        end
        AMDGPU.sync_workgroup()
        s >>= 1
    end

    if tid == 1
        @inbounds output[workgroupIdx().x] = shared[1]
    end
    return nothing
end
# --- end:amdgpu_reduce ---
