# Documentation with Documenter.jl

# Outside the tagged regions: the listing's signature says CuVector, so the
# file needs CUDA loaded to be read at all. The chapter has established this
# long before, which is why the listing itself does not repeat it.
using CUDA

# --- begin:docstring_example ---
"""
    parallel_reduce(op, input::CuVector{T}) -> T

Perform a parallel tree reduction of `input` using binary operator `op`.

# Arguments
- `op`: An associative binary function (e.g., `+`, `max`, `min`, `*`).
  Must be safe for GPU execution (no heap allocations or I/O).
- `input::CuVector{T}`: Input vector on the GPU. Element type `T` must
  satisfy `isbitstype(T) == true`.

# Returns
- A scalar of type `T` representing the reduced result.

# GPU Behavior
- Allocates temporary GPU memory for partial block results (freed after
  the call returns).
- Launches cld(N, 256) thread blocks of 256 threads each (capped at 1024 blocks).
- Uses `sizeof(T) * 256` bytes of shared memory per block.
- Synchronizes before returning (result is available immediately).

# Examples
```jldoctest
julia> using CUDA, CUDAReductions

julia> x = CuArray(Float32[1, 2, 3, 4, 5]);

julia> parallel_reduce(+, x)
15.0f0

julia> parallel_reduce(max, x)
5.0f0
```

See also: [`parallel_scan`](@ref)
"""
function parallel_reduce(op, input::CuVector{T}) where T
    # ... implementation
end
# --- end:docstring_example ---

# --- begin:documenter_make ---
using Documenter
using CUDAReductions

makedocs(;
    modules  = [CUDAReductions],
    sitename = "CUDAReductions.jl",
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "API Reference" => "api.md",
        "GPU Considerations" => "gpu.md",
    ],
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
    ),
)

deploydocs(;
    repo = "github.com/username/CUDAReductions.jl",
)
# --- end:documenter_make ---
