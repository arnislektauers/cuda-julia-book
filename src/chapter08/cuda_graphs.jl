# CUDA graphs: capturing a repeated kernel sequence and replaying it.
#
# Definitions only: the three run_iterations_* variants are called from the
# chapter text and from benchmarks, not at load time, so this file defines the
# comparison (plain launches vs. explicit capture/instantiate vs. @captured)
# without executing an iteration loop.
# --- begin:cuda_graph_capture ---
using CUDA
using BenchmarkTools

function step_kernel!(u, v)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(u)
        @inbounds u[i] = 0.5f0 * (u[i] + v[i])
    end
    return nothing
end

function run_iterations_plain!(u, v, iters)
    threads = 256
    blocks = cld(length(u), threads)
    for _ in 1:iters
        @cuda threads=threads blocks=blocks step_kernel!(u, v)
    end
    return u
end

function run_iterations_graph!(u, v, iters)
    threads = 256
    blocks = cld(length(u), threads)

    # Compile without executing before capturing, so JIT compilation
    # happens outside the capture and does not add an extra iteration.
    # `launch=false` takes no launch-time keywords; threads and blocks are
    # supplied by the launches below.
    @cuda launch=false step_kernel!(u, v)

    # Capture one iteration into a graph.
    graph = CUDA.capture() do
        @cuda threads=threads blocks=blocks step_kernel!(u, v)
    end
    exec = CUDA.instantiate(graph)

    # Replay the captured graph `iters` times.
    for _ in 1:iters
        CUDA.launch(exec)
    end
    return u
end

# Convenience form: @captured caches and reuses the instantiated graph
# across calls, which is ideal for hot loops.
function run_iterations_captured!(u, v, iters)
    threads = 256
    blocks = cld(length(u), threads)
    for _ in 1:iters
        CUDA.@captured begin
            @cuda threads=threads blocks=blocks step_kernel!(u, v)
        end
    end
    return u
end
# --- end:cuda_graph_capture ---

# ---------------------------------------------------------------------------
# Driver, outside the tagged region so the book is unaffected.
#
# The three forms differ only in how the launches are issued, so all three must
# leave the same result. Each starts from its own copy: the kernel updates u in
# place, so sharing one buffer would make the second run start where the first
# stopped and the comparison would be meaningless.
#
# u <- (u + v)/2 repeated `iters` times drives u geometrically toward v, so the
# expected value is closed-form and a dropped or duplicated launch shifts it by
# a factor of two: far outside the tolerance.
let N = 1 << 16, iters = 20
    v = CUDA.fill(1.0f0, N)
    u0 = CUDA.zeros(Float32, N)

    a = run_iterations_plain!(copy(u0), v, iters)
    b = run_iterations_graph!(copy(u0), v, iters)
    c = run_iterations_captured!(copy(u0), v, iters)
    CUDA.synchronize()

    # run_iterations_graph! compiles without executing before capturing, so
    # all three variants perform exactly `iters` updates.
    want_plain = 1.0f0 - 0.5f0^iters
    want_graph = want_plain
    @assert all(isapprox.(Array(a), want_plain; atol = 1f-6)) "plain: $(Array(a)[1])"
    @assert all(isapprox.(Array(c), want_plain; atol = 1f-6)) "captured: $(Array(c)[1])"
    @assert all(isapprox.(Array(b), want_graph; atol = 1f-6)) "graph: $(Array(b)[1])"
    println("cuda graphs: plain, explicit graph, and @captured agree")
end
