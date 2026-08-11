# Sobel edge detection on a synthetic grayscale image.
#
# This is the self-contained, dependency-free variant printed in the chapter:
# it generates its own test image and needs nothing beyond CUDA.jl. The
# Images.jl/TestImages.jl variant that operates on a real photograph, together
# with its CPU reference, lives in sobel_filter.jl.
# --- begin:sobel_kernel ---
using CUDA

function sobel_kernel!(out, img, H, W)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if x <= H && y <= W
        # Clamped neighbor indices replicate the border pixels
        xm = max(x - 1, 1); xp = min(x + 1, H)
        ym = max(y - 1, 1); yp = min(y + 1, W)

        @inbounds begin
            a11 = img[xm, ym]; a12 = img[xm, y]; a13 = img[xm, yp]
            a21 = img[x,  ym];                   a23 = img[x,  yp]
            a31 = img[xp, ym]; a32 = img[xp, y]; a33 = img[xp, yp]

            # Horizontal (across columns) and vertical (across rows)
            # Sobel convolutions, written out as scaled sums
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

threads = (16, 16)
blocks  = (cld(H, threads[1]), cld(W, threads[2]))

t = CUDA.@elapsed begin
    @cuda threads=threads blocks=blocks sobel_kernel!(out, img, H, W)
end
println("Sobel kernel time: $t seconds")
# --- end:sobel_kernel ---
