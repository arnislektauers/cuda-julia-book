# Sobel edge detection on a real photograph: companion-repository variant.
#
# This file is NOT the version printed in the book: it contributes no tagged
# regions. The printed listing is the dependency-free, synthetic-image variant
# in sobel_basic.jl, which the chapter text describes (clamped borders, 2D
# thread indexing) and which the shared-memory follow-up in sobel_shared.jl
# builds on.
#
# What this variant adds, and why it is worth keeping: a real 8-bit test image
# (cameraman) with the 0-255 conversion, and a `sobel_cpu` reference for
# validating the GPU result: the correctness check that Chapter 12's coding
# exercise 1 asks the reader to write. Note two deliberate differences from the
# printed kernel: borders are zeroed rather than clamped, and the thread mapping
# is a flat 1D index decomposed into column-major (x, y) rather than a 2D grid.
# It needs Images.jl and TestImages.jl, so it runs in the extended "T2" tier.
using Images
using CUDA
using TestImages

@inline function sobel_op(
    a00::UInt8, a01::UInt8, a02::UInt8,
    a10::UInt8,             a12::UInt8,
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

function sobel_cpu(img::AbstractMatrix{Gray{T}}) where {T<:Real}
    height, width = size(img)
    out = zeros(Float32, height, width)

    img_u8 = round.(UInt8, clamp.(Float32.(img) .* 255f0, 0f0, 255f0))

    @inbounds for x in 2:width-1, y in 2:height-1
        out[y, x] = sobel_op(
            img_u8[y - 1, x - 1], img_u8[y - 1, x], img_u8[y - 1, x + 1],
            img_u8[y,     x - 1],                   img_u8[y,     x + 1],
            img_u8[y + 1, x - 1], img_u8[y + 1, x], img_u8[y + 1, x + 1]
        )
    end

    return Gray.(out)
end

function sobel_kernel!(
    image::CuDeviceMatrix{UInt8},
    result::CuDeviceMatrix{Float32},
    width::Int32,
    height::Int32
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    total = width * height

    if idx <= total
        # Column-major mapping
        y = (idx - 1) % height + 1
        x = (idx - 1) ÷ height + 1

        @inbounds if x > 1 && y > 1 && x < width && y < height
            result[y, x] = sobel_op(
                image[y - 1, x - 1], image[y - 1, x], image[y - 1, x + 1],
                image[y,     x - 1],                     image[y,     x + 1],
                image[y + 1, x - 1], image[y + 1, x], image[y + 1, x + 1]
            )
        else
            result[y, x] = 0f0
        end
    end
    return
end

function sobel_gpu(img::AbstractMatrix{Gray{T}}) where {T<:Real}
    height, width = size(img)

    img_u8 = round.(UInt8, clamp.(Float32.(img) .* 255f0, 0f0, 255f0))

    d_image  = CuArray(img_u8)
    d_result = CuArray{Float32}(undef, height, width)

    threads = 256
    blocks  = cld(width * height, threads)

    @cuda threads=threads blocks=blocks sobel_kernel!(
        d_image,
        d_result,
        Int32(width),
        Int32(height)
    )

    return Gray.(Array(d_result))
end

function main()
    img = Gray.(testimage("cameraman"))

    println("Running CPU Sobel...")
    @time cpu = sobel_cpu(img)

    println("Running GPU Sobel...")
    @time gpu = sobel_gpu(img)

    return cpu, gpu
end

cpu_result, gpu_result = main()