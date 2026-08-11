# Visualization -- publication figures from GPU results (CairoMakie backend)
#
# Kept separate from visualization_glmakie.jl: Makie activates a backend when
# it is loaded, and the last one loaded wins. Splitting the two also keeps
# this file headless, so it runs anywhere without an X display.

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
