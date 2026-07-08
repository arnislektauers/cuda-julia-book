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
