# Optimized tiled matrix multiplication: 32x32 tiles, four outputs per thread

using CUDA

# --- begin:matmul_optimized ---
const TILE = 32              # tile edge, in elements
const TCOL = 8               # column groups per block: block is TILE x TCOL
const CPT  = TILE ÷ TCOL     # output columns each thread accumulates

function matmul_optimized!(C, A, B)
    sA = CuStaticSharedArray(Float32, (TILE, TILE))
    sB = CuStaticSharedArray(Float32, (TILE, TILE))

    tx = threadIdx().x       # 1:TILE, the row of the tile this thread owns
    ty = threadIdx().y       # 1:TCOL, its first of CPT output columns

    M = size(A, 1)
    K = size(A, 2)
    N = size(B, 2)

    row   = (blockIdx().x - 1) * TILE + tx
    cbase = (blockIdx().y - 1) * TILE

    # One accumulator per output column, held in registers, so each value
    # staged in shared memory is consumed CPT times instead of being re-read.
    acc1 = 0.0f0; acc2 = 0.0f0; acc3 = 0.0f0; acc4 = 0.0f0

    for t in 0:cld(K, TILE)-1
        # The block has TILE*TCOL threads but stages a TILE*TILE tile of each
        # operand, so every thread loads CPT elements of each.
        for r in 0:CPT-1
            j = ty + r * TCOL
            a_col = t * TILE + j
            @inbounds sA[tx, j] = (row <= M && a_col <= K) ? A[row, a_col] : 0.0f0
            b_row = t * TILE + tx
            b_col = cbase + j
            @inbounds sB[tx, j] = (b_row <= K && b_col <= N) ? B[b_row, b_col] : 0.0f0
        end
        sync_threads()

        for k in 1:TILE
            a = @inbounds sA[tx, k]          # one shared read feeds CPT products
            @inbounds acc1 += a * sB[k, ty]
            @inbounds acc2 += a * sB[k, ty + TCOL]
            @inbounds acc3 += a * sB[k, ty + 2 * TCOL]
            @inbounds acc4 += a * sB[k, ty + 3 * TCOL]
        end
        sync_threads()
    end

    if row <= M
        col = cbase + ty;            col <= N && @inbounds C[row, col] = acc1
        col = cbase + ty + TCOL;     col <= N && @inbounds C[row, col] = acc2
        col = cbase + ty + 2 * TCOL; col <= N && @inbounds C[row, col] = acc3
        col = cbase + ty + 3 * TCOL; col <= N && @inbounds C[row, col] = acc4
    end
    return nothing
end
# --- end:matmul_optimized ---

# ---------------------------------------------------------------------------
# Driver, outside the tagged region so the book is unaffected. The launch shape
# is the detail a careless caller gets wrong: the block is TILE x TCOL, not
# TILE x TILE, because each thread now produces CPT outputs rather than one.
let M = 1024, K = 1024, N = 1024
    A = CUDA.rand(Float32, M, K)
    B = CUDA.rand(Float32, K, N)
    C = CUDA.zeros(Float32, M, N)

    @cuda threads=(TILE, TCOL) blocks=(cld(M, TILE), cld(N, TILE)) matmul_optimized!(C, A, B)
    CUDA.synchronize()

    reference = A * B                                  # cuBLAS, via mul!
    err = maximum(abs.(Array(C) .- Array(reference))) / maximum(abs.(Array(reference)))
    @assert err < 1e-4 "tiled matmul disagrees with cuBLAS: relative error $err"
    println("matmul_optimized!: matches cuBLAS to $(round(err, sigdigits=2)) CORRECT")
end
