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
    pid = ct.bid(1)                          # Block ID
    tile_a = ct.load(a, pid, (tile_size,))   # Load tile from global memory
    tile_b = ct.load(b, pid, (tile_size,))
    ct.store(c, pid, tile_a + tile_b)        # Store result
    return
end

# Launch: ct.launch(vadd_tiles, (grid,), a, b, c, ct.Constant(tile_size))
# --- end:vadd_tiles ---
