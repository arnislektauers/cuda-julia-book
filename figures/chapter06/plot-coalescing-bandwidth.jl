#!/usr/bin/env julia
#
# Chapter 6: Coalescing Bandwidth Plot
# Generates representative data for memory coalescing impact on bandwidth.
# Run from this directory: julia plot-coalescing-bandwidth.jl
#
# Note: This script uses synthetic representative data based on published
# benchmarks. For actual measurements, run the stride_copy_kernel on a real GPU.

using CairoMakie

# ── Color palette (consistent across all chapters) ──────────────────────
BLUE   = colorant"#2563EB"
RED    = colorant"#DC2626"
GREEN  = colorant"#16A34A"
PURPLE = colorant"#9558B2"
GRAY   = colorant"#6B7280"
ORANGE = colorant"#EA580C"
BG     = colorant"#F8F9FA"
DARK   = colorant"#1F2937"

# ── Representative bandwidth data (GB/s) for stride access ──────────────
# Based on typical measurements from published CUDA benchmarks
strides = [1, 2, 4, 8, 16, 32, 64, 128]

# H100 SXM: ~3350 GB/s peak HBM3 bandwidth
h100_peak = 3350.0
h100_bw = [3015.0, 1650.0, 870.0, 445.0, 230.0, 120.0, 65.0, 35.0]

# A100 SXM: ~2039 GB/s peak HBM2e bandwidth
a100_peak = 2039.0
a100_bw = [1840.0, 1005.0, 530.0, 275.0, 142.0, 74.0, 40.0, 22.0]

# ── Create the plot ─────────────────────────────────────────────────────
fig = Figure(
    size = (600, 400),
    backgroundcolor = :white,
    fontsize = 12,
)

ax = Axis(fig[1, 1],
    xlabel = "Access Stride (elements)",
    ylabel = "Effective Bandwidth (GB/s)",
    title = "Memory Coalescing Impact on Effective Bandwidth",
    xscale = log2,
    xticks = (strides, string.(strides)),
    yticks = 0:500:3500,
    backgroundcolor = BG,
    titlesize = 14,
    xlabelsize = 12,
    ylabelsize = 12,
)

# Peak bandwidth horizontal dashed lines
hlines!(ax, [h100_peak], color = (BLUE, 0.3), linestyle = :dash, linewidth = 1)
hlines!(ax, [a100_peak], color = (PURPLE, 0.3), linestyle = :dash, linewidth = 1)

# Data lines
lines!(ax, strides, h100_bw, color = BLUE, linewidth = 2.5, label = "H100 SXM")
scatter!(ax, strides, h100_bw, color = BLUE, markersize = 8)

lines!(ax, strides, a100_bw, color = PURPLE, linewidth = 2.5,
       linestyle = :dash, label = "A100 SXM")
scatter!(ax, strides, a100_bw, color = PURPLE, markersize = 8,
         marker = :rect)

# Annotations
text!(ax, 1.15, h100_bw[1] + 100, text = "~90% peak",
      fontsize = 10, color = GREEN)
text!(ax, 40, 160, text = "~3% peak",
      fontsize = 10, color = RED)

# Peak labels (right side)
text!(ax, 100, h100_peak + 50, text = "H100 peak",
      fontsize = 9, color = (BLUE, 0.6))
text!(ax, 100, a100_peak + 50, text = "A100 peak",
      fontsize = 9, color = (PURPLE, 0.6))

axislegend(ax, position = :rt, framevisible = true, framecolor = GRAY,
           labelsize = 11)

ylims!(ax, 0, 3700)

# ── Save ────────────────────────────────────────────────────────────────
save("plot-coalescing-bandwidth_en.svg", fig)
save("plot-coalescing-bandwidth_en.pdf", fig)

println("Saved: plot-coalescing-bandwidth_en.{svg,pdf}")
