# Branch divergence example

using CUDA

# --- begin:divergent_kernel ---
function divergent_kernel!(A, B)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(A)
        # Some threads take if-path, others take else-path
        if A[i] > 0.0f0          
            @inbounds B[i] = A[i] * 2.0f0
        else
            @inbounds B[i] = A[i] * -1.0f0
        end
    end
    return nothing
end
# --- end:divergent_kernel ---
