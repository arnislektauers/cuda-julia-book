using CairoMakie

# Representative performance data for 1024x1024 FP32 matmul on H100
# (Actual values should be measured on target hardware)
implementations = ["Naive\n(global)", "Tiled\n16×16", "Tiled 32×32\n+padding", "cuBLAS\n(Tensor Cores)"]
gflops = [200, 2500, 5500, 22000]
colors = ["#DC2626", "#EA580C", "#2563EB", "#16A34A"]

fig = Figure(size=(600, 400), fontsize=12)
ax = Axis(fig[1,1],
    xlabel="Implementation",
    ylabel="GFLOP/s",
    title="Matrix Multiplication Performance (1024×1024, FP32)",
    xticks=(1:4, implementations),
)

barplot!(ax, 1:4, gflops, color=colors, strokewidth=1, strokecolor=:black)

# Add speedup annotations
for (i, g) in enumerate(gflops)
    speedup = round(g / gflops[1], digits=0)
    text!(ax, i, g + 500, text="$(Int(speedup))×", fontsize=10, align=(:center, :bottom))
end

save("plot-matmul-optimization_en.svg", fig)
save("plot-matmul-optimization_en.pdf", fig)
println("Saved plot-matmul-optimization_en.{svg,pdf}")
