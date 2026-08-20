# Visualization: interactive Makie.jl GPU integration (GLMakie backend)
#
# Kept separate from visualization_cairomakie.jl: Makie activates a backend
# when it is loaded, and the last one loaded wins. A single script that loads
# both would silently render these figures through Cairo instead of OpenGL.
#
# Needs a display. Over SSH, run with DISPLAY set to the machine's X session,
# for example `DISPLAY=:0 julia --project=. visualization_glmakie.jl`.

# --- begin:glmakie_heatmap ---
using GLMakie, CUDA

# GPU computation
d_data = CUDA.rand(Float32, 1000, 1000)
d_result = d_data .^ 2 .+ sin.(d_data)

# Transfer to CPU for visualization
result = Array(d_result)

# Plot with GLMakie (GPU-accelerated rendering)
fig = Figure(size=(800, 600))
ax = Axis(fig[1, 1]; title="GPU-Computed Heatmap")
hm = heatmap!(ax, result; colormap=:viridis)
Colorbar(fig[1, 2], hm)
display(fig)
# --- end:glmakie_heatmap ---

# --- begin:realtime_viz ---
using GLMakie, CUDA

# Observable for live updates
data_obs = Observable(zeros(Float32, 256, 256))

# Create figure with heatmap bound to observable
fig = Figure(size=(800, 600))
ax = Axis(fig[1, 1]; title="GPU Simulation (Live)")
heatmap!(ax, data_obs; colormap=:inferno)
display(fig)

# Simulation loop with live rendering. It lives in a function because the
# buffer swap below rebinds its two names every step, and rebinding at global
# scope inside a loop would create loop-local variables instead.
function simulate!(d_state, d_next, data_obs, steps)
    for step in 1:steps
        # GPU computation: one explicit heat-equation step. The 0.2 factor
        # sits under the 0.25 stability bound for this five-point stencil.
        d_next .= d_state .+ 0.2f0 .* (
            circshift(d_state, (1, 0)) .+ circshift(d_state, (-1, 0)) .+
            circshift(d_state, (0, 1)) .+ circshift(d_state, (0, -1)) .-
            4f0 .* d_state
        )
        d_state, d_next = d_next, d_state

        # Update visualization every 10 steps
        if step % 10 == 0
            data_obs[] = Array(d_state)    # Triggers re-render
            sleep(0.01)                     # Allow rendering
        end
    end
    return d_state
end

# GPU simulation state
d_state = CUDA.rand(Float32, 256, 256)
d_next = similar(d_state)
d_state = simulate!(d_state, d_next, data_obs, 1000)
# --- end:realtime_viz ---
