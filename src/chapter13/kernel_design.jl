# Modular GPU kernel design -- interfaces, dispatch, extensibility

using CUDA

# --- begin:reduce_module ---
module CUDAReductions

using CUDA

export parallel_reduce

# Identity elements for the supported operators. This is the extension
# point: a new operator or element type only needs one more `_identity`
# method (see the Vec3 example), and the kernel below works unchanged.
_identity(::typeof(+), ::Type{T}) where {T} = zero(T)
_identity(::typeof(*), ::Type{T}) where {T} = one(T)
_identity(::typeof(max), ::Type{T}) where {T} = typemin(T)
_identity(::typeof(min), ::Type{T}) where {T} = typemax(T)

# Low-level tree reduction: grid-stride accumulation into a shared-memory
# scratchpad, then a binary-tree reduction to one value per block.
function _reduce_kernel!(op, output, input, N)
    shared = CuDynamicSharedArray(eltype(input), blockDim().x)

    tid = threadIdx().x
    gid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    gridstride = blockDim().x * gridDim().x

    acc = _identity(op, eltype(input))
    i = gid
    while i <= N
        @inbounds acc = op(acc, input[i])
        i += gridstride
    end
    @inbounds shared[tid] = acc

    sync_threads()

    s = blockDim().x >> 1
    while s > 0
        if tid <= s
            @inbounds shared[tid] = op(shared[tid], shared[tid + s])
        end
        sync_threads()
        s >>= 1
    end

    if tid == 1
        @inbounds output[blockIdx().x] = shared[1]
    end

    return nothing
end

"""
    parallel_reduce(op, input::CuVector{T}) -> T

Reduce `input` with the associative binary operator `op`. Any `op` for
which `_identity(op, T)` is defined is supported (`+`, `*`, `max`, and
`min` out of the box). The high-level interface validates the input,
configures the launch, and recursively reduces the per-block partials.

# Examples
```julia
x = CUDA.rand(Float32, 1_000_000)
s = parallel_reduce(+, x)       # Sum
m = parallel_reduce(max, x)     # Maximum
```
"""
function parallel_reduce(op, input::CuVector{T}) where {T}
    N = length(input)
    N == 0 && throw(ArgumentError("input must be non-empty"))

    threads = 256
    blocks = min(cld(N, threads), 1024)  # the grid-stride loop covers larger N

    d_partials = CuVector{T}(undef, blocks)

    @cuda(threads=threads, blocks=blocks, shmem=threads * sizeof(T),
          _reduce_kernel!(op, d_partials, input, N))

    return blocks == 1 ? Array(d_partials)[1] : parallel_reduce(op, d_partials)
end

# CPU fallback: plain vectors use Base's sequential reduce
parallel_reduce(op, input::AbstractVector) = reduce(op, input)

end # module
# --- end:reduce_module ---

# `import` (not `using`) brings `parallel_reduce` into scope *for method
# extension*: the specialized methods below add to the module's function
# rather than shadowing it with a new one in Main.
import .CUDAReductions: parallel_reduce

# --- begin:dispatch_specialized ---
# Specialized paths: delegate Float32/Float64 summation to CUDA.jl's
# optimized reduction instead of the generic kernel. They add methods to
# CUDAReductions.parallel_reduce (imported for extension, not just use).
function parallel_reduce(::typeof(+), input::CuVector{Float32})
    isempty(input) && throw(ArgumentError("input must be non-empty"))
    return CUDA.sum(input)
end

function parallel_reduce(::typeof(+), input::CuVector{Float64})
    isempty(input) && throw(ArgumentError("input must be non-empty"))
    return CUDA.sum(input)
end
# --- end:dispatch_specialized ---

# --- begin:vec3_extension ---
# User-defined reduction over a custom GPU-compatible struct
struct Vec3{T}
    x::T
    y::T
    z::T
end

Base.:+(a::Vec3{T}, b::Vec3{T}) where {T} = Vec3(a.x + b.x, a.y + b.y, a.z + b.z)

# Extend the library's identity element -- the extension point in action.
# `parallel_reduce(+, cu_vec3_array)` now works with no change to the kernel.
CUDAReductions._identity(::typeof(+), ::Type{Vec3{T}}) where {T} =
    Vec3(zero(T), zero(T), zero(T))
# --- end:vec3_extension ---

# Driver (not shown in the book): exercise the module on the GPU
x = CUDA.rand(Float32, 1_000_000)
println("sum  = ", parallel_reduce(+, x))
println("max  = ", parallel_reduce(max, x))

vs = CuArray([Vec3(1.0f0, 2.0f0, 3.0f0) for _ in 1:100_000])
r = parallel_reduce(+, vs)
println("vec3 sum.x = ", r.x)
