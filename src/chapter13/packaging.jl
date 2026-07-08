# Package creation and distribution

using PkgTemplates

# --- begin:pkgtemplates ---
using PkgTemplates

template = Template(;
    user = "username",
    authors = ["Author Name"],
    plugins = [
        License(; name = "MIT"),
        Git(; manifest=false),
        GitHubActions(; extra_versions=["1.12", "nightly"]),
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
using GPUArraysCore: AbstractGPUVector

# Portable kernel implementation
@kernel function _ka_reduce_kernel!(output, @Const(input), op, identity, N)
    # ... kernel code from §13.2.4
end

# Method that dispatches for any AbstractGPUVector
function CUDAReductions.parallel_reduce(op, input::AbstractGPUVector{T}) where T
    # ... portable implementation
end

end # module
# --- end:package_extension ---
