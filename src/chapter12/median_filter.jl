# 3x3 median filter on the GPU using texture memory
# Self-contained: generates a synthetic noisy grayscale test image

# --- begin:median9_kernel ---
using CUDA

# Compare-and-exchange: returns (min, max)
@inline exchange(a, b) = a < b ? (a, b) : (b, a)

function median9_kernel!(out, tex)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    H = size(out, 1)
    W = size(out, 2)

    if x <= H && y <= W
        # Texture fetches: clamp addressing replicates edge pixels,
        # so out-of-range coordinates need no explicit handling
        v1 = tex[x - 1, y - 1]; v2 = tex[x, y - 1]; v3 = tex[x + 1, y - 1]
        v4 = tex[x - 1, y    ]; v5 = tex[x, y    ]; v6 = tex[x + 1, y    ]
        v7 = tex[x - 1, y + 1]; v8 = tex[x, y + 1]; v9 = tex[x + 1, y + 1]

        # Median of 9 via min/max exchanges: sort each row triple,
        # then each column triple, then the anti-diagonal. The center
        # element v5 then holds the median.
        v1, v2 = exchange(v1, v2); v4, v5 = exchange(v4, v5)
        v7, v8 = exchange(v7, v8)
        v2, v3 = exchange(v2, v3); v5, v6 = exchange(v5, v6)
        v8, v9 = exchange(v8, v9)
        v1, v2 = exchange(v1, v2); v4, v5 = exchange(v4, v5)
        v7, v8 = exchange(v7, v8)

        v1, v4 = exchange(v1, v4); v4, v7 = exchange(v4, v7)
        v1, v4 = exchange(v1, v4)
        v2, v5 = exchange(v2, v5); v5, v8 = exchange(v5, v8)
        v2, v5 = exchange(v2, v5)
        v3, v6 = exchange(v3, v6); v6, v9 = exchange(v6, v9)
        v3, v6 = exchange(v3, v6)

        v3, v5 = exchange(v3, v5); v5, v7 = exchange(v5, v7)
        v3, v5 = exchange(v3, v5)

        @inbounds out[x, y] = v5
    end
    return nothing
end
# --- end:median9_kernel ---

# --- begin:median_filter_host ---
using CUDA

function median_filter_gpu(H::Int=512, W::Int=512; noise_p::Float32=0.05f0)
    # Synthetic grayscale test image with salt-and-pepper noise
    img = rand(Float32, H, W)
    r = rand(Float32, H, W)
    img[r .< noise_p / 2] .= 0.0f0                       # pepper
    img[(noise_p / 2 .<= r) .& (r .< noise_p)] .= 1.0f0  # salt

    # Upload the source image to the device
    d_src = CuArray(img)

    # Bind to texture memory:
    # - clamp addressing replicates edge pixels (nearest-pixel border)
    # - nearest-neighbor filtering, since we fetch at integer coordinates
    # - unnormalized coordinates, so the kernel can index as tex[x, y]
    tex = CuTexture(d_src;
        address_mode = CUDA.ADDRESS_MODE_CLAMP,
        interpolation = CUDA.NearestNeighbour(),
        normalized_coordinates = false
    )

    d_out = similar(d_src)

    # Launch config: one thread per output pixel (2D grid)
    threads = (16, 16)
    blocks  = (cld(H, threads[1]), cld(W, threads[2]))

    t = CUDA.@elapsed begin
        @cuda threads=threads blocks=blocks median9_kernel!(d_out, tex)
    end
    println("GPU kernel time (median 3x3, texture fetch): $t seconds")

    return Array(d_out)
end

filtered = median_filter_gpu(512, 512; noise_p=0.05f0)
# --- end:median_filter_host ---
