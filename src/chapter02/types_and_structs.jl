# Type system, structs, and GPU compatibility

using CUDA

# --- begin:step_kernel ---
function step_kernel!(particles::CuDeviceArray{Particle, 1}, 
    dt::Float32)
    
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(particles)
        p = particles[i]
        new_pos = (p.pos[1] + p.vel[1] * dt,
                   p.pos[2] + p.vel[2] * dt,
                   p.pos[3] + p.vel[3] * dt)
        @inbounds particles[i] = Particle(new_pos, p.vel, p.mass)
    end
    return nothing
end
# --- end:step_kernel ---

# --- begin:type_instability ---
# Pattern 1: Dynamic dispatch on an Any-typed value
struct Box
    value::Any           # Every use of this field is inferred as Any
end

# Bad: box.value * 2.0f0 cannot be resolved statically; the compiler
# emits a runtime call (jl_apply_generic), which GPUCompiler rejects
kernel_bad!(out, box::Box) = out[threadIdx().x] = box.value * 2.0f0

# Fix: parameterize so the field type is concrete
struct TypedBox{T<:Real}
    value::T
end

kernel_good!(out, box::TypedBox) = out[threadIdx().x] = box.value * 2.0f0

# Pattern 2: Abstract field types
struct BadConfig
    value::Real          # Abstract field — makes entire struct GPU-incompatible
end

struct GoodConfig{T<:Real}
    value::T             # Concrete when T is bound
end

# Pattern 3: Passing a Dict as a kernel argument
# @cuda fails before the body is ever inferred: kernel argument
# conversion rejects Dict because it is not an isbits type
function kernel_bad2!(out, params)
    out[threadIdx().x] = params[:threshold]
    return nothing
end
# --- end:type_instability ---
