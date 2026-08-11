# Measure the performance data behind the Chapter 6 and Chapter 7 figures.
#
# Emits four `.dat` files consumed by the visutwin specs:
#
#   coalescing-bandwidth.dat    -> figures/spec/plot-coalescing-bandwidth.yaml   (ch 6)
#   matmul-optimization.dat     -> figures/spec/plot-matmul-optimization.yaml    (ch 7)
#   reduction-levels.dat        -> figures/spec/plot-reduction-levels.yaml       (ch 7)
#
# and one that no longer backs a figure:
#
#   blocksize-sensitivity.dat   -> the claim in sec-launch-config that a block-size
#                                  sweep moves a bandwidth-bound kernel by under ten
#                                  percent. The figure was cut because three flat
#                                  lines said less than the sentence does; the sweep
#                                  stays so the claim remains reproducible.
#
# Wherever the book prints a kernel, this script measures *that* kernel rather
# than a re-implementation: `strided_kernel!` generalized to arbitrary stride
# (src/chapter06/coalescing.jl), `matmul_optimized!` (src/chapter07/
# matmul_optimized.jl), `full_reduce_sum!` (src/chapter07/reductions.jl), and
# `add_kernel!` / `add2d_kernel!` (src/chapter07/launch_config.jl). The variants
# the figures compare against — naive matmul, 16x16 tiling, atomic and tree
# reductions — are defined here, because the book does not print them.
#
# Run on the reference machine (Appendix A: RTX 4070 SUPER, Julia 1.12.6,
# CUDA.jl 6.2.1). Timings are the minimum over TRIALS repetitions of a
# REPS-iteration loop; the minimum is the right statistic for kernel timing
# because contention only ever adds time.
#
#   julia --project=<env-with-CUDA> bench_chapter_figures.jl
#
# On the reference host LD_LIBRARY_PATH must be cleared first (it contains
# /usr/local/cuda/lib64, which shadows CUDA.jl's own artifacts).

using CUDA
using Printf

const OUT_DIR = get(ENV, "OUT_DIR", @__DIR__)
const REPS    = 50      # kernel launches per timed loop
const TRIALS  = 7       # timed loops; the minimum is reported

# --- timing helper ---------------------------------------------------------------

"""
Return the minimum per-launch time in seconds for `f`, a zero-argument closure
that issues exactly one kernel launch.
"""
function best_time(f)
    f(); CUDA.synchronize()                       # warm up (compile + cache)
    best = Inf
    for _ in 1:TRIALS
        t = CUDA.@elapsed begin
            for _ in 1:REPS
                f()
            end
        end
        best = min(best, t / REPS)
    end
    return best
end

write_dat(name, header, rows) = open(joinpath(OUT_DIR, name), "w") do io
    println(io, "# ", header)
    for r in rows
        println(io, r)
    end
    println("wrote $(joinpath(OUT_DIR, name))")
end

# =================================================================================
# 1. Coalescing: effective read bandwidth vs access stride  (Chapter 6)
# =================================================================================
# Generalizes src/chapter06/coalescing.jl's `strided_kernel!` from its fixed
# stride of 2 to an arbitrary stride. Two details matter for the comparison to
# mean anything across strides:
#
#   * The kernel is read-only -- values are reduced to one partial per block --
#     so no write traffic enters the measurement. An earlier version wrote one
#     output per read over a fixed-size source, which made the output shrink
#     with the stride until it fit in L2, flattering the wide strides.
#   * NREAD useful reads at every stride, and NREAD*4 B = 64 MiB exceeds this
#     device's 48 MiB L2, so the useful set is never cache-resident.
#
# The source array necessarily grows as NREAD*s: that is what striding does.

const NREAD_COALESCE = 1 << 24        # 16 Mi reads = 64 MiB useful, above L2

function strided_sum_kernel!(partial, A, s, n)
    acc = 0.0f0
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    step = blockDim().x * gridDim().x
    while i <= n
        @inbounds acc += A[1 + (i - 1) * s]
        i += step
    end
    sdata = CuDynamicSharedArray(Float32, blockDim().x)
    tid = threadIdx().x
    @inbounds sdata[tid] = acc
    sync_threads()
    k = blockDim().x ÷ 2
    while k > 0
        tid <= k && (@inbounds sdata[tid] += sdata[tid + k])
        sync_threads()
        k ÷= 2
    end
    tid == 1 && (@inbounds partial[blockIdx().x] = sdata[1])
    return nothing
end

function bench_coalescing()
    threads, blocks = 256, 4096
    partial = CUDA.zeros(Float32, blocks)
    rows = String[]
    for s in (1, 2, 4, 8, 16, 32, 64, 128)
        A = CUDA.rand(Float32, NREAD_COALESCE * s)
        t = best_time(() -> @cuda(threads = threads, blocks = blocks,
                                  shmem = threads * sizeof(Float32),
                                  strided_sum_kernel!(partial, A, s, NREAD_COALESCE)))
        gbps = NREAD_COALESCE * sizeof(Float32) / t / 1e9
        push!(rows, @sprintf("%6d  %10.2f", s, gbps))
        @printf("  stride %4d (source %5.0f MiB): %8.2f GB/s\n",
                s, NREAD_COALESCE * s * 4 / 2^20, gbps)
        CUDA.unsafe_free!(A)
        GC.gc()
    end
    write_dat("coalescing-bandwidth.dat", "stride  effective_read_gbps", rows)
end

# =================================================================================
# 2. Block-size sensitivity  (Chapter 7)
# =================================================================================
# `add_kernel!` and `add2d_kernel!` are the drivers already used by
# src/chapter07/launch_config.jl; the stencil is the third kernel the figure
# names and is defined here.

function add_kernel!(c, a, b)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(c)
        @inbounds c[i] = a[i] + b[i]
    end
    return nothing
end

function stencil3_kernel!(b, a)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if 2 <= i <= length(a) - 1
        @inbounds b[i] = (a[i-1] + a[i] + a[i+1]) / 3.0f0
    end
    return nothing
end

function add2d_kernel!(C, A, B)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= size(C, 1) && j <= size(C, 2)
        @inbounds C[i, j] = A[i, j] + B[i, j]
    end
    return nothing
end

function bench_blocksize()
    N = 1 << 24                          # 16 Mi elements
    a, b = CUDA.rand(Float32, N), CUDA.rand(Float32, N)
    c = CUDA.zeros(Float32, N)

    M = 4096                             # 4096^2 = 16 Mi elements, same footprint
    A2, B2 = CUDA.rand(Float32, M, M), CUDA.rand(Float32, M, M)
    C2 = CUDA.zeros(Float32, M, M)

    block_sizes = [32, 64, 128, 256, 512, 768, 1024]
    rows = String[]
    for bs in block_sizes
        blocks = cld(N, bs)

        t_add = best_time(() -> @cuda(threads = bs, blocks = blocks,
                                      add_kernel!(c, a, b)))
        bw_add = 3 * N * sizeof(Float32) / t_add / 1e9      # 2 reads + 1 write

        t_st = best_time(() -> @cuda(threads = bs, blocks = blocks,
                                     stencil3_kernel!(c, a)))
        bw_st = 2 * N * sizeof(Float32) / t_st / 1e9        # 1 read + 1 write (cached halo)

        # 2D kernel: keep 32 threads along the contiguous axis so the access
        # stays coalesced, and vary the second axis to reach the block size.
        ty = bs ÷ 32
        blocks2 = (cld(M, 32), cld(M, ty))
        t_mat = best_time(() -> @cuda(threads = (32, ty), blocks = blocks2,
                                      add2d_kernel!(C2, A2, B2)))
        bw_mat = 3 * M * M * sizeof(Float32) / t_mat / 1e9

        push!(rows, @sprintf("%6d  %10.2f  %10.2f  %10.2f", bs, bw_add, bw_st, bw_mat))
        @printf("  bs %5d: vecadd %7.1f  stencil %7.1f  matrix %7.1f GB/s\n",
                bs, bw_add, bw_st, bw_mat)
    end
    write_dat("blocksize-sensitivity.dat",
              "threads  vecadd_gbps  stencil_gbps  matrix_gbps", rows)
end

# =================================================================================
# 3. Matrix multiplication optimization levels  (Chapter 7)
# =================================================================================

# The baseline the textbook optimization ladder implicitly starts from: `col`
# varies with threadIdx().x, so in Julia's column-major layout the lanes of a
# warp read B[k, col] at addresses n apart and every access is uncoalesced.
# Without this rung the ladder is misleading, because the kernel below is
# already coalesced and therefore already most of the way to the tiled result.
function matmul_naive_uncoalesced!(C, A, B)
    col = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    row = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if row <= size(C, 1) && col <= size(C, 2)
        tmp = 0.0f0
        for k in 1:size(A, 2)
            @inbounds tmp += A[row, k] * B[k, col]
        end
        @inbounds C[row, col] = tmp
    end
    return nothing
end

# Coalesced thread mapping: row varies with threadIdx().x, so a warp reads 32
# consecutive rows of A at fixed k, and B[k, col] is one broadcast address.
function matmul_naive!(C, A, B)
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    col = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if row <= size(C, 1) && col <= size(C, 2)
        tmp = 0.0f0
        for k in 1:size(A, 2)
            @inbounds tmp += A[row, k] * B[k, col]
        end
        @inbounds C[row, col] = tmp
    end
    return nothing
end

# Tiled without padding, TILE = 16.
function matmul_tiled16!(C, A, B)
    T = 16
    sA = CuStaticSharedArray(Float32, (16, 16))
    sB = CuStaticSharedArray(Float32, (16, 16))
    tx, ty = threadIdx().x, threadIdx().y
    row = (blockIdx().x - 1) * T + tx
    col = (blockIdx().y - 1) * T + ty
    M, K, N = size(A, 1), size(A, 2), size(B, 2)
    tmp = 0.0f0
    for t in 0:cld(K, T)-1
        a_col = t * T + ty
        @inbounds sA[tx, ty] = (row <= M && a_col <= K) ? A[row, a_col] : 0.0f0
        b_row = t * T + tx
        @inbounds sB[tx, ty] = (b_row <= K && col <= N) ? B[b_row, col] : 0.0f0
        sync_threads()
        for k in 1:T
            @inbounds tmp += sA[tx, k] * sB[k, ty]
        end
        sync_threads()
    end
    if row <= M && col <= N
        @inbounds C[row, col] = tmp
    end
    return nothing
end

# The book's kernel, verbatim from src/chapter07/matmul_optimized.jl: 32x32
# tiles, an unpadded shared array, and a TILE x TCOL block in which each thread
# accumulates CPT output columns in registers.
const TILE = 32
const TCOL = 8
const CPT  = TILE ÷ TCOL

function matmul_optimized!(C, A, B)
    sA = CuStaticSharedArray(Float32, (TILE, TILE))
    sB = CuStaticSharedArray(Float32, (TILE, TILE))
    tx = threadIdx().x
    ty = threadIdx().y
    M = size(A, 1); K = size(A, 2); N = size(B, 2)
    row   = (blockIdx().x - 1) * TILE + tx
    cbase = (blockIdx().y - 1) * TILE
    acc1 = 0.0f0; acc2 = 0.0f0; acc3 = 0.0f0; acc4 = 0.0f0
    for t in 0:cld(K, TILE)-1
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
            a = @inbounds sA[tx, k]
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

function bench_matmul()
    M = K = N = 1024
    A = CUDA.rand(Float32, M, K)
    B = CUDA.rand(Float32, K, N)
    C = CUDA.zeros(Float32, M, N)
    flops = 2.0 * M * N * K
    ref = Array(A) * Array(B)

    function check(label)
        got = Array(C)
        err = maximum(abs.(got .- ref)) / maximum(abs.(ref))
        err < 1e-4 || error("$label: relative error $err too large")
        fill!(C, 0.0f0)
    end

    t_unc = best_time(() -> @cuda(threads = (16, 16),
                                  blocks = (cld(M, 16), cld(N, 16)),
                                  matmul_naive_uncoalesced!(C, A, B)))
    check("uncoalesced")

    t_naive = best_time(() -> @cuda(threads = (16, 16),
                                    blocks = (cld(M, 16), cld(N, 16)),
                                    matmul_naive!(C, A, B)))
    check("naive")

    t_t16 = best_time(() -> @cuda(threads = (16, 16),
                                  blocks = (cld(M, 16), cld(N, 16)),
                                  matmul_tiled16!(C, A, B)))
    check("tiled16")

    t_t32 = best_time(() -> @cuda(threads = (TILE, TCOL),
                                  blocks = (cld(M, TILE), cld(N, TILE)),
                                  matmul_optimized!(C, A, B)))
    check("tiled32 coarsened")

    t_blas = best_time(() -> CUDA.CUBLAS.gemm!('N', 'N', 1.0f0, A, B, 0.0f0, C))
    check("cuBLAS")

    # Speedups are quoted against the uncoalesced baseline, which is the rung
    # the classic optimization ladder starts from.
    times = [t_unc, t_naive, t_t16, t_t32, t_blas]
    names = ["naive uncoalesced", "naive coalesced", "tiled16", "tiled32 coarsened", "cuBLAS"]
    rows = String[]
    for (i, (nm, t)) in enumerate(zip(names, times))
        g = flops / t / 1e9
        speedup = t_unc / t
        push!(rows, @sprintf("%d  %10.1f  %8.2f  # %s", i, g, speedup, nm))
        @printf("  %-18s %9.1f GFLOP/s  (%.1fx)\n", nm, g, speedup)
    end
    write_dat("matmul-optimization.dat",
              "impl_idx  gflops  speedup_vs_uncoalesced", rows)
end

# =================================================================================
# 4. Reduction implementation levels  (Chapter 7)
# =================================================================================

function reduce_atomic!(out, input)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(input)
        @inbounds CUDA.@atomic out[1] += input[i]
    end
    return nothing
end

function reduce_tree!(out, input)
    sdata = CuDynamicSharedArray(Float32, blockDim().x)
    tid = threadIdx().x
    i = (blockIdx().x - 1) * blockDim().x + tid
    @inbounds sdata[tid] = i <= length(input) ? input[i] : 0.0f0
    sync_threads()
    s = blockDim().x ÷ 2
    while s > 0
        if tid <= s
            @inbounds sdata[tid] += sdata[tid + s]
        end
        sync_threads()
        s ÷= 2
    end
    if tid == 1
        @inbounds out[blockIdx().x] = sdata[1]
    end
    return nothing
end

# The book's kernel, verbatim from src/chapter07/reductions.jl.
@inline function warp_reduce_sum(val::Float32)
    mask = 0xffffffff
    val += CUDA.shfl_down_sync(mask, val, 16)
    val += CUDA.shfl_down_sync(mask, val, 8)
    val += CUDA.shfl_down_sync(mask, val, 4)
    val += CUDA.shfl_down_sync(mask, val, 2)
    val += CUDA.shfl_down_sync(mask, val, 1)
    return val
end

function full_reduce_sum!(output, input)
    sdata = CuDynamicSharedArray(Float32, 32)
    tid = threadIdx().x
    i = (blockIdx().x - 1) * blockDim().x + tid
    stride = blockDim().x * gridDim().x
    val = 0.0f0
    while i <= length(input)
        val += @inbounds input[i]
        i += stride
    end
    val = warp_reduce_sum(val)
    lane = (tid - 1) % 32 + 1
    warp_id = (tid - 1) ÷ 32 + 1
    if lane == 1
        @inbounds sdata[warp_id] = val
    end
    sync_threads()
    val = (tid <= blockDim().x ÷ 32) ? @inbounds(sdata[tid]) : 0.0f0
    if warp_id == 1
        val = warp_reduce_sum(val)
    end
    if tid == 1
        @inbounds output[blockIdx().x] = val
    end
    return nothing
end

function bench_reduction()
    N = 1 << 24                         # 16 Mi Float32 = 64 MiB
    input = CUDA.ones(Float32, N)
    bytes = N * sizeof(Float32)
    threads = 256

    # 1. naive atomic: every thread hits one global accumulator
    out1 = CUDA.zeros(Float32, 1)
    blocks1 = cld(N, threads)
    t_atomic = best_time(() -> @cuda(threads = threads, blocks = blocks1,
                                     reduce_atomic!(out1, input)))

    # 2. tree reduction in shared memory, one partial per block
    out2 = CUDA.zeros(Float32, blocks1)
    t_tree = best_time(() -> @cuda(threads = threads, blocks = blocks1,
                                   shmem = threads * sizeof(Float32),
                                   reduce_tree!(out2, input)))
    fill!(out2, 0.0f0)
    @cuda threads=threads blocks=blocks1 shmem=threads*sizeof(Float32) reduce_tree!(out2, input)
    CUDA.synchronize()
    isapprox(sum(Array(out2)), Float32(N); rtol = 1e-3) ||
        error("tree reduction wrong: $(sum(Array(out2))) vs $N")

    # 3. the book's warp-shuffle + shared-memory grid-stride kernel.
    #    It is a grid-stride kernel, so the block count is a free parameter and
    #    the driver in the book picks 64 only to keep the listing short. Timing
    #    it at 64 blocks against a tree reduction that gets N/256 blocks would
    #    measure the launch configuration, not the algorithm, so sweep the grid
    #    and report its best.
    t_shfl = Inf
    blocks3 = 64
    for b in (64, 256, 1024, 4096, 16384, 65536)
        o = CUDA.zeros(Float32, b)
        t = best_time(() -> @cuda(threads = threads, blocks = b,
                                  shmem = 32 * sizeof(Float32),
                                  full_reduce_sum!(o, input)))
        @printf("      warp shuffle @ %6d blocks: %7.2f GB/s\n", b, bytes / t / 1e9)
        if t < t_shfl
            t_shfl = t; blocks3 = b
        end
        CUDA.unsafe_free!(o)
    end
    println("      -> best grid: $blocks3 blocks")
    out3 = CUDA.zeros(Float32, blocks3)
    fill!(out3, 0.0f0)
    @cuda threads=threads blocks=blocks3 shmem=32*sizeof(Float32) full_reduce_sum!(out3, input)
    CUDA.synchronize()
    sum(Array(out3)) == Float32(N) ||
        error("warp-shuffle reduction wrong: $(sum(Array(out3))) vs $N")

    # 4. CUDA.jl's own sum
    t_builtin = best_time(() -> sum(input))

    times = [t_atomic, t_tree, t_shfl, t_builtin]
    names = ["naive atomic", "tree (shmem)", "warp shuffle", "CUDA.jl sum()"]
    rows = String[]
    for (i, (nm, t)) in enumerate(zip(names, times))
        gbps = bytes / t / 1e9
        push!(rows, @sprintf("%d  %10.2f  # %s", i, gbps, nm))
        @printf("  %-15s %8.2f GB/s\n", nm, gbps)
    end
    write_dat("reduction-levels.dat", "impl_idx  effective_gbps", rows)
end

# =================================================================================

function main()
    CUDA.functional() || error("no functional CUDA device")
    dev = CUDA.device()
    println("Device : ", CUDA.name(dev))
    println("Julia  : ", VERSION)
    println("Runtime: ", CUDA.runtime_version())
    println("Driver : ", CUDA.driver_version())
    # Theoretical peak bandwidth, for the reference lines in the figures.
    bus = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
    clk = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)   # kHz
    peak = 2 * clk * 1e3 * (bus / 8) / 1e9
    @printf("Peak memory bandwidth (theoretical): %.1f GB/s\n\n", peak)

    println("[1/4] coalescing bandwidth vs stride");  bench_coalescing();  println()
    println("[2/4] block-size sensitivity");          bench_blocksize();   println()
    println("[3/4] matmul optimization levels");      bench_matmul();      println()
    println("[4/4] reduction levels");                bench_reduction();   println()

    open(joinpath(OUT_DIR, "bench-environment.txt"), "w") do io
        println(io, "device=", CUDA.name(dev))
        println(io, "julia=", VERSION)
        println(io, "cuda_runtime=", CUDA.runtime_version())
        println(io, "cuda_driver=", CUDA.driver_version())
        @printf(io, "peak_bandwidth_gbps=%.1f\n", peak)
    end
    println("done")
end

main()
