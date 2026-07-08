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
