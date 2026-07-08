# Shared-memory Sobel filter: each block stages a (BX+2) x (BY+2) tile
# with a one-pixel halo, then evaluates both convolutions on-chip.
# Follows the conventions of sobel_filter.jl (Float32 image, clamped borders).

using CUDA

# --- begin:sobel_shmem ---
function sobel_shmem_kernel!(out, img, H, W,
                             ::Val{BX}, ::Val{BY}) where {BX, BY}
    tx = threadIdx().x                    # row within the block
    ty = threadIdx().y                    # column within the block
    x = (blockIdx().x - 1) * BX + tx      # global row
    y = (blockIdx().y - 1) * BY + ty      # global column

    tile = CuStaticSharedArray(Float32, (BX + 2, BY + 2))

    # Clamped source indices replicate the border pixels
    xc = min(max(x, 1), H)
    yc = min(max(y, 1), W)

    # Every thread loads its own pixel into the tile interior
    @inbounds tile[tx + 1, ty + 1] = img[xc, yc]

    # Threads on the block edges also load the one-pixel halo
    @inbounds begin
        if tx == 1
            tile[1, ty + 1] = img[max(x - 1, 1), yc]
        end
        if tx == BX
            tile[BX + 2, ty + 1] = img[min(x + 1, H), yc]
        end
        if ty == 1
            tile[tx + 1, 1] = img[xc, max(y - 1, 1)]
        end
        if ty == BY
            tile[tx + 1, BY + 2] = img[xc, min(y + 1, W)]
        end
        if tx == 1 && ty == 1
            tile[1, 1] = img[max(x - 1, 1), max(y - 1, 1)]
        end
        if tx == BX && ty == 1
            tile[BX + 2, 1] = img[min(x + 1, H), max(y - 1, 1)]
        end
        if tx == 1 && ty == BY
            tile[1, BY + 2] = img[max(x - 1, 1), min(y + 1, W)]
        end
        if tx == BX && ty == BY
            tile[BX + 2, BY + 2] = img[min(x + 1, H), min(y + 1, W)]
        end
    end

    sync_threads()

    if x <= H && y <= W
        @inbounds begin
            a11 = tile[tx, ty]
            a12 = tile[tx, ty + 1]
            a13 = tile[tx, ty + 2]
            a21 = tile[tx + 1, ty]
            a23 = tile[tx + 1, ty + 2]
            a31 = tile[tx + 2, ty]
            a32 = tile[tx + 2, ty + 1]
            a33 = tile[tx + 2, ty + 2]

            gx = (a13 + 2.0f0 * a23 + a33) - (a11 + 2.0f0 * a21 + a31)
            gy = (a31 + 2.0f0 * a32 + a33) - (a11 + 2.0f0 * a12 + a13)

            out[x, y] = sqrt(gx * gx + gy * gy)
        end
    end
    return nothing
end

H, W = 512, 512
img = CUDA.rand(Float32, H, W)   # synthetic grayscale test image
out = CUDA.zeros(Float32, H, W)

BX, BY = 16, 16
blocks = (cld(H, BX), cld(W, BY))

t = CUDA.@elapsed begin
    @cuda threads=(BX, BY) blocks=blocks sobel_shmem_kernel!(
        out, img, H, W, Val(BX), Val(BY))
end
println("Shared-memory Sobel kernel time: $t seconds")
# --- end:sobel_shmem ---
