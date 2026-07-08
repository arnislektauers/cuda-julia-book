@inline function sobel_op(
    a00::UInt8, a01::UInt8, a02::UInt8,
    a10::UInt8,            a12::UInt8,
    a20::UInt8, a21::UInt8, a22::UInt8
)::Float32
    dx = Float32(
        Int32(a00) - Int32(a02) +
        2 * (Int32(a10) - Int32(a12)) +
        Int32(a20) - Int32(a22)
    )
    dy = Float32(
        Int32(a00) - Int32(a20) +
        2 * (Int32(a01) - Int32(a21)) +
        Int32(a02) - Int32(a22)
    )
    return sqrt(dx * dx + dy * dy) * (1f0 / 255f0)
end

using CUDA

@inline function load_u8_or_zero(img::CuDeviceMatrix{UInt8}, y::Int32, x::Int32, h::Int32, w::Int32)::UInt8
    return (1 <= x <= w && 1 <= y <= h) ? @inbounds(img[y, x]) : UInt8(0)
end

function sobel_shmem_kernel!(
    image::CuDeviceMatrix{UInt8},
    result::CuDeviceMatrix{Float32},
    width::Int32,
    height::Int32,
    ::Val{BX},
    ::Val{BY},
) where {BX, BY}

    tx = Int32(threadIdx().x)
    ty = Int32(threadIdx().y)

    x = Int32((blockIdx().x - 1) * BX) + tx
    y = Int32((blockIdx().y - 1) * BY) + ty

    tile = CuStaticSharedArray(UInt8, (BY + 2, BX + 2))

    # Default zero (avoids conditionals later)
    @inbounds tile[ty + 1, tx + 1] = UInt8(0)

    # Load center pixel
    if x >= 1 && x <= width && y >= 1 && y <= height
        @inbounds tile[ty + 1, tx + 1] = image[y, x]
    end

    # Halo loads
    if tx == 1 && x > 1 && y >= 1 && y <= height
        @inbounds tile[ty + 1, 1] = image[y, x - 1]
    end
    if tx == BX && x < width && y >= 1 && y <= height
        @inbounds tile[ty + 1, BX + 2] = image[y, x + 1]
    end
    if ty == 1 && y > 1 && x >= 1 && x <= width
        @inbounds tile[1, tx + 1] = image[y - 1, x]
    end
    if ty == BY && y < height && x >= 1 && x <= width
        @inbounds tile[BY + 2, tx + 1] = image[y + 1, x]
    end

    # Halo corners
    if tx == 1 && ty == 1 && x > 1 && y > 1
        @inbounds tile[1, 1] = image[y - 1, x - 1]
    end
    if tx == BX && ty == 1 && x < width && y > 1
        @inbounds tile[1, BX + 2] = image[y - 1, x + 1]
    end
    if tx == 1 && ty == BY && x > 1 && y < height
        @inbounds tile[BY + 2, 1] = image[y + 1, x - 1]
    end
    if tx == BX && ty == BY && x < width && y < height
        @inbounds tile[BY + 2, BX + 2] = image[y + 1, x + 1]
    end

    sync_threads()

    if x > 1 && x < width && y > 1 && y < height
        @inbounds begin
            a00 = tile[ty,     tx]
            a01 = tile[ty,     tx + 1]
            a02 = tile[ty,     tx + 2]
            a10 = tile[ty + 1, tx]
            a12 = tile[ty + 1, tx + 2]
            a20 = tile[ty + 2, tx]
            a21 = tile[ty + 2, tx + 1]
            a22 = tile[ty + 2, tx + 2]

            result[y, x] = sobel_op(a00, a01, a02, a10, a12, a20, a21, a22)
        end
    elseif x >= 1 && x <= width && y >= 1 && y <= height
        result[y, x] = 0f0
    end

    return
end

using Images

function sobel_gpu_shmem(img::AbstractMatrix{Gray{T}}; BX::Int=16, BY::Int=16) where {T<:Real}
    height, width = size(img)

    # Convert to UInt8 explicitly (0..255)
    img_u8 = round.(UInt8, clamp.(Float32.(img) .* 255f0, 0f0, 255f0))

    d_image  = CuArray(img_u8)
    d_result = CuArray{Float32}(undef, height, width)

    threads = (BX, BY)
    blocks  = (cld(width, BX), cld(height, BY))

    @cuda threads=threads blocks=blocks sobel_shmem_kernel!(
        d_image, d_result, Int32(width), Int32(height), Val(BX), Val(BY)
    )

    return Gray.(Array(d_result))
end

using TestImages

img = Gray.(testimage("cameraman"))
@time out = sobel_gpu_shmem(img; BX=16, BY=16)