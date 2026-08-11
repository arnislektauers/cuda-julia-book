# cuTile.jl: tile-based programming comparison

using CUDA

# --- begin:vadd_threads ---
# Thread-based (CUDA.jl)
function vadd_threads!(c, a, b)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(a)
        @inbounds c[i] = a[i] + b[i]
    end
    return nothing
end
# --- end:vadd_threads ---

# --- begin:vadd_tiles ---
# Tile-based (cuTile.jl)
import cuTile as ct

function vadd_tiles(a, b, c, tile_size::Int)
    pid = ct.bid(1)                                       # Block ID
    tile_a = ct.load(a; index=pid, shape=(tile_size,))    # Tile from global memory
    tile_b = ct.load(b; index=pid, shape=(tile_size,))
    ct.store(c; index=pid, tile=tile_a + tile_b)          # Store result
    return
end

# Launch: @cuda backend=ct blocks=grid vadd_tiles(a, b, c, ct.Constant(tile_size))
# --- end:vadd_tiles ---
