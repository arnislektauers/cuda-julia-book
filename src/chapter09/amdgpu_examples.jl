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

# ---------------------------------------------------------------------------
# Driver, outside the tagged region so the book is unaffected. The listings
# print nothing, so completing proves only that they did not crash.
#
# Needs an AMD GPU with a working ROCm stack. RTU has none -- its `epyc` nodes
# are AMD CPUs with NVIDIA GPUs -- so this was verified on a rented MI300X
# (gfx942, ROCm 6.4.4, AMDGPU.jl on Julia 1.12.6). See runlist.txt for the two
# environment traps that cost an earlier attempt.
let
    # The arrays region: A is ones, B is random, so the broadcast has a
    # closed-form host equivalent.
    a_h, b_h = Array(A), Array(B)
    @assert Array(D) ≈ sin.(a_h) .+ 2f0 .* b_h "broadcast differs from the host computation"
    @assert length(C_cpu) == 1024 "host round-trip lost elements"

    # The vadd region already launched and synchronized; check what it wrote.
    @assert Array(c) ≈ Array(a) .+ Array(b) "vadd kernel result is wrong"

    # reduce_kernel! is defined by the third region and never launched there.
    # Its shared array is fixed at 256 Float32s, so groupsize must be 256.
    # gridsize counts workgroups, matching the vadd launch above and
    # AMDGPU.jl's own documented example.
    N = 1 << 16
    groupsize = 256
    ngroups = cld(N, groupsize)
    input = AMDGPU.ones(Float32, N)
    output = ROCArray{Float32}(undef, ngroups)
    @roc groupsize=groupsize gridsize=ngroups reduce_kernel!(output, input)
    AMDGPU.synchronize()
    partials = Array(output)
    @assert all(partials .== Float32(groupsize)) "per-group sums: $(unique(partials))"
    @assert sum(partials) == Float32(N) "total $(sum(partials)), expected $N"
    println("amdgpu: broadcast, vadd_kernel! and reduce_kernel! all CORRECT")
end
