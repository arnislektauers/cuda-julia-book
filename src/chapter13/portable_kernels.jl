# Backend portability with KernelAbstractions.jl

using KernelAbstractions

# --- begin:ka_reduce_kernel ---
using KernelAbstractions

@kernel function ka_reduce_kernel!(output, @Const(input), op, identity, N)
    lid = @index(Local)
    gid = @index(Global)

    # Shared memory via KernelAbstractions; the workgroup size is fixed
    # at 256 because @localmem requires a compile-time constant
    shared = @localmem eltype(input) (256,)

    # Grid-stride accumulation: covers inputs larger than the ndrange
    @uniform gridstride = prod(@ndrange())
    acc = identity
    i = gid
    while i <= N
        @inbounds acc = op(acc, input[i])
        i += gridstride
    end
    @inbounds shared[lid] = acc

    @synchronize

    s = 256 >> 1  # same fixed workgroup size as the @localmem allocation
    while s > 0
        if lid <= s
            @inbounds shared[lid] = op(shared[lid], shared[lid + s])
        end
        @synchronize
        s >>= 1
    end

    if lid == 1
        @inbounds output[@index(Group)]  = shared[1]
    end
end
# --- end:ka_reduce_kernel ---

# --- begin:portable_reduce ---
using GPUArraysCore: AbstractGPUVector

function portable_reduce(op, input::AbstractGPUVector{T}) where T
    backend = KernelAbstractions.get_backend(input)
    N = length(input)
    threads = 256
    blocks = min(cld(N, threads), 1024)  # grid-stride loop in the kernel covers larger N
    output = similar(input, blocks)

    kernel! = ka_reduce_kernel!(backend, threads)
    kernel!(output, input, op, _identity(op, T), N; ndrange=blocks*threads)
    KernelAbstractions.synchronize(backend)

    return blocks == 1 ? Array(output)[1] : portable_reduce(op, output)
end
# --- end:portable_reduce ---
