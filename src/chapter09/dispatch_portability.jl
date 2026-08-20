# Multiple dispatch patterns for performance portability

using CUDA
using AMDGPU

# --- begin:dispatch_specialized ---
# Generic implementation: works on any GPU (or CPU)
function my_transform!(output::AbstractArray, input::AbstractArray)
    output .= sin.(input) .+ input .^ 2
    return output
end

# CUDA-optimized path (e.g., using a fused custom kernel)
function fused_kernel!(output, input)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(input)
        @inbounds output[i] = sin(input[i]) + input[i]^2
    end
    return nothing
end

function my_transform!(output::CuArray, input::CuArray)
    blocks = cld(length(input), 256)
    @cuda threads=256 blocks=blocks fused_kernel!(output, input)
    return output
end

# AMD-optimized path using the corresponding ROCm kernel launch API.
function amd_fused_kernel!(output, input)
    i = workitemIdx().x + (workgroupIdx().x - 1) * workgroupDim().x
    if i <= length(input)
        @inbounds output[i] = sin(input[i]) + input[i]^2
    end
    return nothing
end

function my_transform!(output::ROCArray, input::ROCArray)
    groupsize = 256
    gridsize = cld(length(input), groupsize)
    @roc groupsize=groupsize gridsize=gridsize amd_fused_kernel!(output, input)
    return output
end
# --- end:dispatch_specialized ---

# --- begin:architecture_abstraction ---
abstract type AbstractArchitecture end
struct CPUArch <: AbstractArchitecture end
struct GPUArch <: AbstractArchitecture end

# Array allocation dispatches on architecture
device_array(::CPUArch, T, dims...) = Array{T}(undef, dims...)
device_array(::GPUArch, T, dims...) = CuArray{T}(undef, dims...)

# Grid construction uses architecture throughout
function create_grid(arch::AbstractArchitecture; size)
    x = device_array(arch, Float64, size...)
    y = device_array(arch, Float64, size...)
    return (x=x, y=y)
end
# --- end:architecture_abstraction ---
