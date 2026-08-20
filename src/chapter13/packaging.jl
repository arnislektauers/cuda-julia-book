# Package creation and distribution

using PkgTemplates

# --- begin:pkgtemplates ---
using PkgTemplates

template = Template(;
    user = "username",
    authors = ["Your Name"],
    plugins = [
        License(; name = "MIT"),
        Git(; manifest=false),
        GitHubActions(),
        Documenter{GitHubActions}(),
        Codecov(),
    ],
)

template("CUDAReductions")
# --- end:pkgtemplates ---

# --- begin:module_file ---
module CUDAReductions

using CUDA

# Internal implementation files
include("utils.jl")
include("launch_config.jl")
include("reduce.jl")
include("scan.jl")

# Public API
export parallel_reduce, parallel_scan

end # module
# --- end:module_file ---

# --- begin:package_extension ---
# ext/CUDAReductionsKernelAbstractionsExt.jl
module CUDAReductionsKernelAbstractionsExt

using CUDAReductions
using KernelAbstractions
using GPUArraysCore: AbstractGPUArray

# Portable kernel implementation
@kernel function _ka_reduce_kernel!(output, @Const(input), op, identity, N)
    # ... kernel code from §13.2.4
end

# Method that dispatches for any one-dimensional AbstractGPUArray
function CUDAReductions.parallel_reduce(op, input::AbstractGPUArray{T,1}) where T
    # ... portable implementation
end

end # module
# --- end:package_extension ---
