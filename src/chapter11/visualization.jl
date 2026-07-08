# Visualization -- Makie.jl GPU integration

using GLMakie, CairoMakie, CUDA, Statistics

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
    # GPU computation (e.g., heat equation step)
    d_state .= 0.25f0 .* (
        circshift(d_state, (1, 0)) .+ circshift(d_state, (-1, 0)) .+
        circshift(d_state, (0, 1)) .+ circshift(d_state, (0, -1))
    )

    # Update visualization every 10 steps
    if step % 10 == 0
        data_obs[] = Array(d_state)    # Triggers re-render
        sleep(0.01)                     # Allow rendering
    end
end
# --- end:realtime_viz ---

# --- begin:publication_figures ---
using CairoMakie, CUDA, Statistics

# GPU computation: parameter sweep results
n_params = 50
n_samples = 100_000

d_params = CuArray(range(0.0f0, 5.0f0; length=n_params))
results = Vector{Float32}(undef, n_params)

for (i, α) in enumerate(Array(d_params))
    d_samples = CUDA.randn(Float32, n_samples)
    d_output = α .* d_samples .^ 2 .+ sin.(d_samples)
    results[i] = mean(Array(d_output))
end

# Publication figure with CairoMakie
fig = Figure(size=(500, 350), fontsize=12)
ax = Axis(fig[1, 1];
    xlabel=L"\alpha",
    ylabel=L"\mathbb{E}[\alpha x^2 + \sin(x)]",
    title="GPU-Computed Parameter Sweep"
)
lines!(ax, Array(d_params), results; linewidth=2, color=:blue)
save("parameter_sweep.pdf", fig)
# --- end:publication_figures ---
