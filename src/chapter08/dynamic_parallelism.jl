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

# ---------------------------------------------------------------------------
# Driver, outside the tagged region so the book is unaffected.
#
# expensive_refinement is the placeholder the listing calls and never defines;
# supplying it here is what makes the file runnable at all.
@inline expensive_refinement(x) = x * 10.0f0

# Device-side launch needs the parent kernel compiled with dynamic parallelism
# support, which every architecture this book targets has (sm_35 and up).
let N = 1024, threshold = 0.5f0
    input = CuArray(Float32.(range(0.0f0, 1.0f0; length = N)))
    output = CUDA.zeros(Float32, N)
    @cuda threads=256 blocks=cld(N, 256) parent_kernel!(output, input, threshold)
    CUDA.synchronize()
    a, got = Array(input), Array(output)
    want = [x > threshold ? x * 10.0f0 : 0.0f0 for x in a]
    @assert got == want "child launches wrote $(count(got .!= want)) wrong elements"
    @assert count(a .> threshold) > 0 "threshold selected nothing -- the test proves nothing"
    println("dynamic_parallelism: $(count(a .> threshold)) child launches CORRECT")
end
