# Tiled matrix multiplication with shared memory

using CUDA

# --- begin:matmul_tiled ---
const TILE = 16

function matmul_tiled!(C, A, B)
    # Shared memory tiles for A and B
    sA = CuStaticSharedArray(Float32, (TILE, TILE))
    sB = CuStaticSharedArray(Float32, (TILE, TILE))

    tx = threadIdx().x
    ty = threadIdx().y
    row = (blockIdx().x - 1) * TILE + tx
    col = (blockIdx().y - 1) * TILE + ty

    M = size(A, 1)
    K = size(A, 2)
    N = size(B, 2)

    tmp = 0.0f0
    num_tiles = cld(K, TILE)

    for t in 0:num_tiles-1
        # Cooperatively load tile of A into shared memory
        a_col = t * TILE + ty
        if row <= M && a_col <= K
            @inbounds sA[tx, ty] = A[row, a_col]
        else
            sA[tx, ty] = 0.0f0
        end

        # Cooperatively load tile of B into shared memory
        b_row = t * TILE + tx
        if b_row <= K && col <= N
            @inbounds sB[tx, ty] = B[b_row, col]
        else
            sB[tx, ty] = 0.0f0
        end

        sync_threads()

        # Compute partial dot product from tiles
        for k in 1:TILE
            @inbounds tmp += sA[tx, k] * sB[k, ty]
        end

        sync_threads()
    end

    # Store result
    if row <= M && col <= N
        @inbounds C[row, col] = tmp
    end
    return nothing
end

M, K, N = 512, 512, 512
A = CUDA.rand(Float32, M, K)
B = CUDA.rand(Float32, K, N)
C = CUDA.zeros(Float32, M, N)

blocks = (cld(M, TILE), cld(N, TILE))
@cuda threads=(TILE, TILE) blocks=blocks matmul_tiled!(C, A, B)
# --- end:matmul_tiled ---
