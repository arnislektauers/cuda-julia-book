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
