# Visualization -- interactive Makie.jl GPU integration (GLMakie backend)
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
heatmap!(ax, result; colormap=:viridis)
Colorbar(fig[1, 2])
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

# GPU simulation state
d_state = CUDA.rand(Float32, 256, 256)

# Simulation loop with live rendering
for step in 1:1000
    # GPU computation: one explicit heat-equation step. The 0.2 factor sits
    # under the 0.25 stability bound for this five-point stencil.
    d_state .+= 0.2f0 .* (
        circshift(d_state, (1, 0)) .+ circshift(d_state, (-1, 0)) .+
        circshift(d_state, (0, 1)) .+ circshift(d_state, (0, -1)) .- 4f0 .* d_state
    )

    # Update visualization every 10 steps
    if step % 10 == 0
        data_obs[] = Array(d_state)    # Triggers re-render
        sleep(0.01)                     # Allow rendering
    end
end
# --- end:realtime_viz ---
