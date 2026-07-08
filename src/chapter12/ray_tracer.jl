# GPU ray tracer with custom structs
# Demonstrates GPU-compatible structs, StaticArrays on GPU, and 2D thread indexing

using CUDA, StaticArrays, LinearAlgebra

# --- begin:ray_types ---
using StaticArrays
using LinearAlgebra

const Vec3{T<:AbstractFloat} = SVector{3, T}
# --- end:ray_types ---

# --- begin:ray_structs ---
struct Ray{T<:AbstractFloat}
    origin::Vec3{T}
    direction::Vec3{T}
end

struct Sphere{T<:AbstractFloat}
    center::Vec3{T}
    radius::T
end
# --- end:ray_structs ---

# --- begin:ray_intersect ---
function ray_sphere_intersect(origin::Vec3{Float32}, dir::Vec3{Float32},
                               center::Vec3{Float32}, radius::Float32)
    oc = origin - center
    a = dot(dir, dir)
    b = 2.0f0 * dot(oc, dir)
    c = dot(oc, oc) - radius^2
    discriminant = b^2 - 4.0f0 * a * c
    if discriminant < 0.0f0
        return -1.0f0
    end
    return (-b - sqrt(discriminant)) / (2.0f0 * a)
end
# --- end:ray_intersect ---

# --- begin:render_kernel ---
function render_kernel!(framebuffer, width, height, spheres, n_spheres,
                        cam_origin, lower_left, horizontal, vertical)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if ix <= width && iy <= height
        u = Float32(ix - 1) / Float32(width - 1)
        v = Float32(iy - 1) / Float32(height - 1)

        ray_dir = lower_left + u * horizontal + v * vertical - cam_origin
        ray_dir = ray_dir / norm(ray_dir)

        # Find the closest hit, then shade it once after the loop
        closest_t = typemax(Float32)
        hit_idx = 0

        for s in 1:n_spheres
            t = ray_sphere_intersect(cam_origin, ray_dir,
                                     spheres[s].center, spheres[s].radius)
            if t > 0.001f0 && t < closest_t
                closest_t = t
                hit_idx = s
            end
        end

        hit_color = Vec3{Float32}(0.5f0, 0.7f0, 1.0f0)  # Sky background
        if hit_idx > 0
            hit_point = cam_origin + closest_t * ray_dir
            normal = (hit_point - spheres[hit_idx].center) / spheres[hit_idx].radius
            light_dir = Vec3{Float32}(1.0f0, 1.0f0, 0.5f0)
            light_dir = light_dir / norm(light_dir)
            diffuse = max(0.0f0, dot(normal, light_dir))
            hit_color = diffuse * Vec3{Float32}(0.8f0, 0.3f0, 0.3f0)
        end

        pixel_idx = (iy - 1) * width + ix
        @inbounds framebuffer[pixel_idx] = hit_color
    end
    return nothing
end
# --- end:render_kernel ---

# --- begin:render_scene ---
using CUDA, StaticArrays

function render_scene()
    width, height = 800, 400

    # Define scene (array of Sphere structs)
    spheres_host = [
        Sphere(Vec3{Float32}(0.0f0, 0.0f0, -1.0f0), 0.5f0),
        Sphere(Vec3{Float32}(0.0f0, -100.5f0, -1.0f0), 100.0f0),
        Sphere(Vec3{Float32}(1.0f0, 0.0f0, -1.0f0), 0.5f0),
    ]
    d_spheres = CuArray(spheres_host)

    # Camera setup
    cam_origin = Vec3{Float32}(0.0f0, 0.0f0, 0.0f0)
    lower_left = Vec3{Float32}(-2.0f0, -1.0f0, -1.0f0)
    horizontal = Vec3{Float32}(4.0f0, 0.0f0, 0.0f0)
    vertical   = Vec3{Float32}(0.0f0, 2.0f0, 0.0f0)

    # Framebuffer
    d_fb = CuArray{Vec3{Float32}}(undef, width * height)

    threads = (16, 16)
    blocks = (cld(width, 16), cld(height, 16))

    @cuda threads=threads blocks=blocks render_kernel!(
        d_fb, width, height, d_spheres, Int32(length(spheres_host)),
        cam_origin, lower_left, horizontal, vertical
    )

    return Array(d_fb), width, height
end
# --- end:render_scene ---

# Run
framebuffer, w, h = render_scene()
println("Rendered $(w)x$(h) image")
