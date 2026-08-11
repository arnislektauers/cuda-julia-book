# Memory coalescing patterns: 1D stride, 2D column-major, SoA vs AoS

using CUDA

# --- begin:coalesced_vs_strided ---
# Coalesced access: adjacent threads read adjacent elements
function coalesced_kernel!(B, A)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(A)
        @inbounds B[i] = A[i] * 2.0f0    # OK coalesced
    end
    return nothing
end

# Non-coalesced: stride-2 access
function strided_kernel!(B, A)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if 2i - 1 <= length(A)
        @inbounds B[i] = A[2i - 1]        # BAD stride-2 -> 2× transactions
    end
    return nothing
end
# --- end:coalesced_vs_strided ---

# --- begin:coalesced_2d ---
# OK Coalesced: threadIdx().x -> row index (contiguous in memory)
function coalesced_2d_kernel!(C, A, B)
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    col = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if row <= size(A, 1) && col <= size(A, 2)
        @inbounds C[row, col] = A[row, col] + B[row, col]
    end
    return nothing
end

# BAD Non-coalesced: threadIdx().x -> column index (strided in memory)
function noncoalesced_2d_kernel!(C, A, B)
    col = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    row = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if row <= size(A, 1) && col <= size(A, 2)
        @inbounds C[row, col] = A[row, col] + B[row, col]
    end
    return nothing
end
# --- end:coalesced_2d ---

# --- begin:soa_vs_aos ---
N = 10_000                   # number of particles

# Array of Structs (AoS): poor coalescing
struct Particle
    x::Float32
    y::Float32
    z::Float32
    mass::Float32
end
particles = CuArray([Particle(rand(Float32, 4)...) for _ in 1:N])
# Reading all x values: stride = sizeof(Particle) = 16 bytes -> non-coalesced

# Struct of Arrays (SoA): good coalescing
struct ParticlesSoA
    x::CuVector{Float32}
    y::CuVector{Float32}
    z::CuVector{Float32}
    mass::CuVector{Float32}
end
particles_soa = ParticlesSoA(CUDA.rand(N), CUDA.rand(N),
                              CUDA.rand(N), CUDA.rand(N))
# Reading all x values: contiguous -> coalesced
# --- end:soa_vs_aos ---
