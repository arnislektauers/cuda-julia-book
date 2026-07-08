using CairoMakie

# Representative bandwidth for different reduction approaches on H100
# N = 10M Float32 elements
# (Actual values should be measured on target hardware)
implementations = ["Naive\n(atomic)", "Tree\n(shmem)", "Warp shuffle\n+ shmem", "CUDA.jl\nsum()"]
bandwidth = [150, 1800, 3000, 3200]
colors = ["#DC2626", "#EA580C", "#2563EB", "#16A34A"]

fig = Figure(size=(600, 400), fontsize=12)
ax = Axis(fig[1,1],
    xlabel="Reduction Approach",
    ylabel="Effective bandwidth (GB/s)",
    title="Parallel Reduction Performance (N=10M, FP32)",
    xticks=(1:4, implementations),
)

barplot!(ax, 1:4, bandwidth, color=colors, strokewidth=1, strokecolor=:black)

# Add speedup annotations
for (i, bw) in enumerate(bandwidth)
    speedup = round(bw / bandwidth[1], digits=1)
    text!(ax, i, bw + 50, text="$(speedup)×", fontsize=10, align=(:center, :bottom))
end

# Peak bandwidth reference line
hlines!(ax, [3350], color="#6B7280", linestyle=:dash, linewidth=1)
text!(ax, 3.5, 3400, text="H100 peak BW (3.35 TB/s)", fontsize=9, color="#6B7280")

save("plot-reduction-levels_en.svg", fig)
save("plot-reduction-levels_en.pdf", fig)
println("Saved plot-reduction-levels_en.{svg,pdf}")
