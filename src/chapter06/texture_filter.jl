# 2D image averaging filter using texture memory cache

using CUDA

# --- begin:tex_average ---
using CUDA

function tex_average!(output, tex, width, height)
    x = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    y = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    if x <= width && y <= height
        s = 0.0f0
        for dx in -1:1, dy in -1:1
            s += tex[x + dx, y + dy]  # texture fetch via indexing
        end
        @inbounds output[x, y] = s / 9.0f0
    end
    return nothing
end

# Host-side setup: upload the image, bind a texture object, launch
img = rand(Float32, 512, 512)
texarr = CuTextureArray(img)
tex = CuTexture(texarr; address_mode=CUDA.ADDRESS_MODE_CLAMP,
                interpolation=CUDA.NearestNeighbour())
output = CuArray{Float32}(undef, 512, 512)

@cuda threads=(16, 16) blocks=(32, 32) tex_average!(output, tex, 512, 512)
# --- end:tex_average ---
