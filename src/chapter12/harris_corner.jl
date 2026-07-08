# Harris corner detection: five-stage GPU pipeline.
# K1 gradients and K3 Gaussian smoothing use shared-memory tiles with halos;
# K2 (tensor products) and K4 (corner response) are fused broadcasts;
# K5 (non-maximum suppression) appends corners to a compact list via atomics.

using CUDA

# --- begin:harris_kernels ---
# Stage a (BX+2) x (BY+2) tile of A in shared memory, with a one-pixel
# halo and clamped (border-replicating) indices. Shared by K1 and K3.
@inline function load_tile!(tile, A, x, y, tx, ty, H, W,
                            ::Val{BX}, ::Val{BY}) where {BX, BY}
    xc = min(max(x, 1), H)
    yc = min(max(y, 1), W)
    @inbounds begin
        tile[tx + 1, ty + 1] = A[xc, yc]
        if tx == 1
            tile[1, ty + 1] = A[max(x - 1, 1), yc]
        end
        if tx == BX
            tile[BX + 2, ty + 1] = A[min(x + 1, H), yc]
        end
        if ty == 1
            tile[tx + 1, 1] = A[xc, max(y - 1, 1)]
        end
        if ty == BY
            tile[tx + 1, BY + 2] = A[xc, min(y + 1, W)]
        end
        if tx == 1 && ty == 1
            tile[1, 1] = A[max(x - 1, 1), max(y - 1, 1)]
        end
        if tx == BX && ty == 1
            tile[BX + 2, 1] = A[min(x + 1, H), max(y - 1, 1)]
        end
        if tx == 1 && ty == BY
            tile[1, BY + 2] = A[max(x - 1, 1), min(y + 1, W)]
        end
        if tx == BX && ty == BY
            tile[BX + 2, BY + 2] = A[min(x + 1, H), min(y + 1, W)]
        end
    end
    return nothing
end

# K1: Sobel gradients Ix and Iy from a shared-memory tile
function gradient_kernel!(Ix, Iy, img, H, W,
                          ::Val{BX}, ::Val{BY}) where {BX, BY}
    tx = threadIdx().x
    ty = threadIdx().y
    x = (blockIdx().x - 1) * BX + tx
    y = (blockIdx().y - 1) * BY + ty

    tile = CuStaticSharedArray(Float32, (BX + 2, BY + 2))
    load_tile!(tile, img, x, y, tx, ty, H, W, Val(BX), Val(BY))
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

            Ix[x, y] = (a13 + 2.0f0 * a23 + a33) - (a11 + 2.0f0 * a21 + a31)
            Iy[x, y] = (a31 + 2.0f0 * a32 + a33) - (a11 + 2.0f0 * a12 + a13)
        end
    end
    return nothing
end

# K3: 3x3 binomial window (separable [1 2 1]/4 in each direction),
# the discrete approximation of the Gaussian weighting w(x, y)
function gauss3_kernel!(out, A, H, W,
                        ::Val{BX}, ::Val{BY}) where {BX, BY}
    tx = threadIdx().x
    ty = threadIdx().y
    x = (blockIdx().x - 1) * BX + tx
    y = (blockIdx().y - 1) * BY + ty

    tile = CuStaticSharedArray(Float32, (BX + 2, BY + 2))
    load_tile!(tile, A, x, y, tx, ty, H, W, Val(BX), Val(BY))
    sync_threads()

    if x <= H && y <= W
        @inbounds out[x, y] = (
            tile[tx, ty] + 2.0f0 * tile[tx, ty + 1] + tile[tx, ty + 2] +
            2.0f0 * tile[tx + 1, ty] + 4.0f0 * tile[tx + 1, ty + 1] +
            2.0f0 * tile[tx + 1, ty + 2] +
            tile[tx + 2, ty] + 2.0f0 * tile[tx + 2, ty + 1] + tile[tx + 2, ty + 2]
        ) * (1.0f0 / 16.0f0)
    end
    return nothing
end

# K5: non-maximum suppression; surviving corners are appended to a
# compact coordinate list through an atomic counter
function nms_kernel!(xs, ys, count, R, H, W, threshold, maxcorners)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if 2 <= x <= H - 1 && 2 <= y <= W - 1
        @inbounds begin
            r = R[x, y]
            if r > threshold &&
               r >= R[x - 1, y - 1] && r >= R[x - 1, y] && r >= R[x - 1, y + 1] &&
               r >= R[x,     y - 1] &&                     r >= R[x,     y + 1] &&
               r >= R[x + 1, y - 1] && r >= R[x + 1, y] && r >= R[x + 1, y + 1]
                # atomic_add! returns the pre-increment value
                slot = CUDA.atomic_add!(pointer(count, 1), Int32(1))
                if slot < maxcorners
                    xs[slot + 1] = Int32(x)
                    ys[slot + 1] = Int32(y)
                end
            end
        end
    end
    return nothing
end
# --- end:harris_kernels ---

# --- begin:harris_pipeline ---
function harris_corners(img; alpha = 0.05f0, rel_threshold = 0.01f0,
                        maxcorners = 4096)
    H, W = size(img)
    BX, BY = 16, 16
    threads = (BX, BY)
    blocks = (cld(H, BX), cld(W, BY))

    # K1: image gradients
    Ix = similar(img)
    Iy = similar(img)
    @cuda threads=threads blocks=blocks gradient_kernel!(
        Ix, Iy, img, H, W, Val(BX), Val(BY))

    # K2: structure tensor components (fused broadcasts)
    Ixx = Ix .* Ix
    Iyy = Iy .* Iy
    Ixy = Ix .* Iy

    # K3: Gaussian weighting of each component
    Sxx = similar(img); Syy = similar(img); Sxy = similar(img)
    @cuda threads=threads blocks=blocks gauss3_kernel!(
        Sxx, Ixx, H, W, Val(BX), Val(BY))
    @cuda threads=threads blocks=blocks gauss3_kernel!(
        Syy, Iyy, H, W, Val(BX), Val(BY))
    @cuda threads=threads blocks=blocks gauss3_kernel!(
        Sxy, Ixy, H, W, Val(BX), Val(BY))

    # K4: corner response R = det(A) - alpha * trace(A)^2 (fused broadcast)
    R = (Sxx .* Syy .- Sxy .^ 2) .- alpha .* (Sxx .+ Syy) .^ 2

    # K5: non-maximum suppression with atomic corner-list append
    threshold = rel_threshold * maximum(R)
    xs = CUDA.zeros(Int32, maxcorners)
    ys = CUDA.zeros(Int32, maxcorners)
    count = CUDA.zeros(Int32, 1)
    @cuda threads=threads blocks=blocks nms_kernel!(
        xs, ys, count, R, H, W, threshold, Int32(maxcorners))

    n = min(Int(Array(count)[1]), maxcorners)
    return Array(xs)[1:n], Array(ys)[1:n]
end

img = CUDA.rand(Float32, 512, 512)   # synthetic test image
t = CUDA.@elapsed xs, ys = harris_corners(img)
println("Harris pipeline time: $t seconds, corners found: $(length(xs))")
# --- end:harris_pipeline ---
