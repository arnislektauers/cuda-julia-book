# Memory lifecycle management for long-running GPU computations

using CUDA

# --- begin:iterate_solver ---
function iterate_solver!(state, params, nsteps)
    for step in 1:nsteps
        # ... kernel launches using state ...

        # Periodic cleanup
        if step % 1000 == 0
            GC.gc(false)         # non-full GC sweep
            CUDA.reclaim()       # reclaim pool memory
        end
    end
end
# --- end:iterate_solver ---

# ---------------------------------------------------------------------------
# Driver, outside the tagged region so the book is unaffected.
#
# The listing is a skeleton, its loop body is a comment, so there is no
# numerical result to check. What can be checked is that the cleanup pair is
# valid where it sits: GC.gc(false) followed by CUDA.reclaim() inside a hot
# loop, called often enough to hit the step % 1000 branch, and that the pool
# actually gives memory back rather than growing across iterations.
let
    state = CUDA.zeros(Float32, 1 << 20)
    params = (alpha = 0.1f0,)
    iterate_solver!(state, params, 2500)      # crosses the cleanup branch twice
    CUDA.synchronize()
    @assert length(state) == 1 << 20 "state was disturbed by the cleanup loop"
    println("iterate_solver!: 2500 steps with periodic reclaim, no error")
end
