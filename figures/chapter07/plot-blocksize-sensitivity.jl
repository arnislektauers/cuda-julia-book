using CairoMakie

# Representative data for block size sensitivity on H100
# (Actual values should be measured on target hardware)
block_sizes = [32, 64, 128, 256, 512, 768, 1024]

# Vector addition (memory-bound, low register usage)
bw_vecadd = [1200, 2100, 2800, 3100, 3150, 3100, 3050]

# Stencil kernel (memory-bound, moderate register usage)
bw_stencil = [900, 1700, 2500, 2900, 2950, 2850, 2700]

# Matrix kernel (compute-bound, high register usage)
bw_matrix = [600, 1100, 1800, 2200, 2000, 1700, 1500]

fig = Figure(size=(600, 400), fontsize=12)
ax = Axis(fig[1,1],
    xlabel="Threads per block",
    ylabel="Effective bandwidth (GB/s)",
    title="Block Size Sensitivity for Different Kernel Types",
    xticks=block_sizes,
)

lines!(ax, block_sizes, bw_vecadd, color="#2563EB", linewidth=2, label="Vector addition")
scatter!(ax, block_sizes, bw_vecadd, color="#2563EB", markersize=8)

lines!(ax, block_sizes, bw_stencil, color="#16A34A", linewidth=2, linestyle=:dash, label="Stencil (3-pt)")
scatter!(ax, block_sizes, bw_stencil, color="#16A34A", markersize=8, marker=:rect)

lines!(ax, block_sizes, bw_matrix, color="#9558B2", linewidth=2, linestyle=:dot, label="Matrix kernel")
scatter!(ax, block_sizes, bw_matrix, color="#9558B2", markersize=8, marker=:diamond)

vlines!(ax, [256], color="#6B7280", linestyle=:dash, linewidth=1)
text!(ax, 270, 500, text="typical default (256)", fontsize=9, color="#6B7280")

axislegend(ax, position=:rb)

save("plot-blocksize-sensitivity_en.svg", fig)
save("plot-blocksize-sensitivity_en.pdf", fig)
println("Saved plot-blocksize-sensitivity_en.{svg,pdf}")
