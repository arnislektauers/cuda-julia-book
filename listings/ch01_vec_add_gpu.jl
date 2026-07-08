using CUDA

function vec_add_kernel!(c, a, b)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(a)
        @inbounds c[i] = a[i] + b[i]
    end
    return nothing
end

N = 1_000_000
a = CUDA.rand(Float32, N)
b = CUDA.rand(Float32, N)
c = similar(a)

threads = 256
blocks = cld(N, threads)
@cuda threads=threads blocks=blocks vec_add_kernel!(c, a, b)

c_host = Array(c)
