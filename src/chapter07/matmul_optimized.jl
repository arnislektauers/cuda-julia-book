# Optimized tiled matrix multiplication with bank-conflict avoidance

using CUDA

# --- begin:matmul_optimized ---
const TILE = 32

function matmul_optimized!(C, A, B)
    # Padded shared memory tiles to avoid bank conflicts:
    # pad the first (column-major) dimension so columns are 33 words apart
    sA = CuStaticSharedArray(Float32, (TILE + 1, TILE))
    sB = CuStaticSharedArray(Float32, (TILE + 1, TILE))

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
        # Cooperatively load tile of A
        a_col = t * TILE + ty
        if row <= M && a_col <= K
            @inbounds sA[tx, ty] = A[row, a_col]
        else
            sA[tx, ty] = 0.0f0
        end

        # Cooperatively load tile of B
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

    if row <= M && col <= N
        @inbounds C[row, col] = tmp
    end
    return nothing
end
# --- end:matmul_optimized ---

M, K, N = 1024, 1024, 1024
A = CUDA.rand(Float32, M, K)
B = CUDA.rand(Float32, K, N)
C = CUDA.zeros(Float32, M, N)

@cuda threads=(TILE, TILE) blocks=(cld(M, TILE), cld(N, TILE)) matmul_optimized!(C, A, B)
