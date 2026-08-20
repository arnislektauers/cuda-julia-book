# Interoperability with C++/CUDA libraries

using CUDA

# --- begin:ccall_binding ---
const libmykernels = "path/to/libmykernels.so"

function launch_saxpy!(n::Int, α::Float32,
                       x::CuVector{Float32}, y::CuVector{Float32})
    ccall((:launch_saxpy, libmykernels), Cvoid,
          (Cint, Cfloat, CuPtr{Cfloat}, CuPtr{Cfloat}),
          n, α, pointer(x), pointer(y))
end
# --- end:ccall_binding ---

# ---------------------------------------------------------------------------
# Driver, outside the tagged region so the book is unaffected.
#
# The listing ccalls into "path/to/libmykernels.so", a library the reader is
# expected to have built. That placeholder is why this file had never run: the
# ccall signature, and in particular CuPtr{Cfloat} for a device pointer,
# which is the part readers get wrong, was never exercised.
#
# So build the library. The .cu below is the C++ side the chapter describes,
# compiled to exactly the path the listing names, relative to the working
# directory. Requires nvcc; the file is marked SKIP without it.
let
    lib = joinpath("path", "to", "libmykernels.so")
    if !isfile(lib)
        nvcc = Sys.which("nvcc")
        nvcc === nothing && error("interop needs nvcc to build $lib")
        mkpath(dirname(lib))
        src = tempname() * ".cu"
        write(src, """
        extern "C" __global__ void saxpy(int n, float a, const float *x, float *y) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) y[i] = a * x[i] + y[i];
        }
        extern "C" void launch_saxpy(int n, float a, const float *x, float *y) {
            int threads = 256, blocks = (n + threads - 1) / threads;
            saxpy<<<blocks, threads>>>(n, a, x, y);
            cudaDeviceSynchronize();
        }
        """)
        run(`$nvcc -shared -Xcompiler -fPIC -o $lib $src`)
    end

    n = 1 << 16
    α = 2.5f0
    x = CUDA.rand(Float32, n)
    y = CUDA.rand(Float32, n)
    xh, yh = Array(x), Array(y)
    launch_saxpy!(n, α, x, y)
    CUDA.synchronize()
    @assert Array(y) ≈ α .* xh .+ yh "saxpy through ccall gave the wrong result"
    println("launch_saxpy!: ccall into $lib CORRECT")
end
