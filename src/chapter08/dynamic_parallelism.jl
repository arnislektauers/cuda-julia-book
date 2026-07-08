# Dynamic parallelism: launching kernels from kernels

using CUDA

# --- begin:dynamic_parallelism ---
function parent_kernel!(output, input, threshold)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(input) && @inbounds(input[i]) > threshold
        # Launch a child kernel from the device
        @cuda dynamic=true threads=32 child_kernel!(output, input, i)
    end
    return nothing
end

function child_kernel!(output, input, parent_idx)
    tid = threadIdx().x
    # Perform additional computation for the selected element
    if tid == 1
        @inbounds output[parent_idx] = expensive_refinement(input[parent_idx])
    end
    return nothing
end
# --- end:dynamic_parallelism ---
