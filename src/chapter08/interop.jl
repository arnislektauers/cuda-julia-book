# Interoperability with C++/CUDA libraries

using CUDA

# --- begin:ccall_binding ---
const libmykernels = "path/to/libmykernels.so"

function launch_saxpy!(n::Int, α::Float32, x::CuVector{Float32}, y::CuVector{Float32})
    ccall((:launch_saxpy, libmykernels), Cvoid,
          (Cint, Cfloat, CuPtr{Cfloat}, CuPtr{Cfloat}),
          n, α, pointer(x), pointer(y))
end
# --- end:ccall_binding ---
