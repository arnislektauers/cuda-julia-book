# 2D Heat equation with shared memory tiling

# --- begin:heat2d_kernel ---
using CUDA

const BX = 16
const BY = 16

function heat2d_kernel!(u_new, u, alpha_dt_dx2, Nx, Ny)
    # Shared memory tile with halo
    tile = CuStaticSharedArray(Float32, (BX + 2, BY + 2))

    # Local thread indices (1-based, shifted for halo)
    tx = threadIdx().x
    ty = threadIdx().y

    # Global indices
    gx = (blockIdx().x - 1) * BX + tx
    gy = (blockIdx().y - 1) * BY + ty

    # Load center region into shared memory
    if gx <= Nx && gy <= Ny
        @inbounds tile[tx + 1, ty + 1] = u[gx, gy]
    end

    # Load halo cells (each condition also checks the orthogonal bound)
    if tx == 1 && gx > 1 && gy <= Ny
        @inbounds tile[1, ty + 1] = u[gx - 1, gy]
    end
    if tx == BX && gx < Nx && gy <= Ny
        @inbounds tile[BX + 2, ty + 1] = u[gx + 1, gy]
    end
    if ty == 1 && gy > 1 && gx <= Nx
        @inbounds tile[tx + 1, 1] = u[gx, gy - 1]
    end
    if ty == BY && gy < Ny && gx <= Nx
        @inbounds tile[tx + 1, BY + 2] = u[gx, gy + 1]
    end

    sync_threads()

    # Compute stencil from shared memory
    if gx >= 2 && gx <= Nx - 1 && gy >= 2 && gy <= Ny - 1
        @inbounds u_new[gx, gy] = tile[tx + 1, ty + 1] + alpha_dt_dx2 * (
            tile[tx + 2, ty + 1] + tile[tx, ty + 1] +
            tile[tx + 1, ty + 2] + tile[tx + 1, ty] -
            4.0f0 * tile[tx + 1, ty + 1]
        )
    end

    return nothing
end
# --- end:heat2d_kernel ---

# ---------------------------------------------------------------------------
# Driver. Outside the tagged region, so it never reaches the book: the listing
# shows the kernel, and this exercises it. Without a driver the file only
# defines a method, and Julia compiles no function body until it is called --
# an out-of-bounds index or a bad launch config would pass unnoticed.
#
# Checked rather than merely run: a single interior hot cell must diffuse to
# its four neighbours, and with alpha*dt/dx^2 = 0.2 the centre drops to
# 1 - 4*0.2 = 0.2 while each neighbour rises to 0.2.
let Nx = 64, Ny = 64, r = 0.2f0
    u = CUDA.zeros(Float32, Nx, Ny)
    CUDA.@allowscalar u[32, 32] = 1.0f0
    u_new = copy(u)
    threads = (BX, BY)
    blocks = (cld(Nx, BX), cld(Ny, BY))
    @cuda threads=threads blocks=blocks heat2d_kernel!(u_new, u, r, Nx, Ny)
    CUDA.synchronize()
    h = Array(u_new)
    @assert isapprox(h[32, 32], 0.2f0; atol = 1f-5) "centre: $(h[32,32])"
    for (i, j) in ((31, 32), (33, 32), (32, 31), (32, 33))
        @assert isapprox(h[i, j], 0.2f0; atol = 1f-5) "neighbour ($i,$j): $(h[i,j])"
    end
    println("heat2d_kernel!: diffusion step CORRECT")
end
