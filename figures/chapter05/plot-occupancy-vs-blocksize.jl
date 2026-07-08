# Plot: Occupancy vs Block Size for Chapter 5
# This script generates the occupancy-vs-blocksize figure.
# It uses representative data (not GPU-dependent) for illustration.
#
# To generate real data on a machine with a GPU, uncomment the CUDA section below.

using CairoMakie

# --- Representative occupancy data ---
# Based on H100 SM (compute capability 9.0):
#   Max warps per SM: 64
#   Max threads per SM: 2048
#   Max blocks per SM: 32
#   Max registers per SM: 65536
#   Max shared memory per SM: 228 KB

warp_size = 32
max_warps_per_sm = 64
max_threads_per_sm = 2048
max_blocks_per_sm = 32
max_regs_per_sm = 65536
max_smem_per_sm = 228 * 1024  # bytes

block_sizes = collect(32:32:1024)

# Scenario 1: 32 registers/thread, 0 shared memory
function occupancy_scenario1(block_size)
    regs_per_thread = 32
    smem_per_block = 0

    warps_per_block = block_size ÷ warp_size
    # Register limit
    threads_per_sm_by_regs = max_regs_per_sm ÷ regs_per_thread
    blocks_by_regs = threads_per_sm_by_regs ÷ block_size
    # Block limit
    blocks_by_limit = max_blocks_per_sm
    # Thread limit
    blocks_by_threads = max_threads_per_sm ÷ block_size

    active_blocks = min(blocks_by_regs, blocks_by_limit, blocks_by_threads)
    active_blocks = max(active_blocks, 0)
    active_warps = active_blocks * warps_per_block
    return min(active_warps / max_warps_per_sm * 100, 100.0)
end

# Scenario 2: 48 registers/thread, 0 shared memory
function occupancy_scenario2(block_size)
    regs_per_thread = 48
    smem_per_block = 0

    warps_per_block = block_size ÷ warp_size
    threads_per_sm_by_regs = max_regs_per_sm ÷ regs_per_thread
    blocks_by_regs = threads_per_sm_by_regs ÷ block_size
    blocks_by_limit = max_blocks_per_sm
    blocks_by_threads = max_threads_per_sm ÷ block_size

    active_blocks = min(blocks_by_regs, blocks_by_limit, blocks_by_threads)
    active_blocks = max(active_blocks, 0)
    active_warps = active_blocks * warps_per_block
    return min(active_warps / max_warps_per_sm * 100, 100.0)
end

# Scenario 3: 32 registers/thread, 32 KB shared memory per block
function occupancy_scenario3(block_size)
    regs_per_thread = 32
    smem_per_block = 32 * 1024  # 32 KB

    warps_per_block = block_size ÷ warp_size
    threads_per_sm_by_regs = max_regs_per_sm ÷ regs_per_thread
    blocks_by_regs = threads_per_sm_by_regs ÷ block_size
    blocks_by_limit = max_blocks_per_sm
    blocks_by_threads = max_threads_per_sm ÷ block_size
    blocks_by_smem = max_smem_per_sm ÷ smem_per_block

    active_blocks = min(blocks_by_regs, blocks_by_limit, blocks_by_threads, blocks_by_smem)
    active_blocks = max(active_blocks, 0)
    active_warps = active_blocks * warps_per_block
    return min(active_warps / max_warps_per_sm * 100, 100.0)
end

occ1 = [occupancy_scenario1(bs) for bs in block_sizes]
occ2 = [occupancy_scenario2(bs) for bs in block_sizes]
occ3 = [occupancy_scenario3(bs) for bs in block_sizes]

# --- Plot ---
fig = Figure(size=(500, 340), fontsize=11)

ax = Axis(fig[1, 1],
    xlabel="Threads per block",
    ylabel="Occupancy (%)",
    xticks=block_sizes[1:2:end],
    yticks=0:10:100,
    limits=(nothing, (0, 105)),
    xticklabelrotation=π/4,
)

# Recommended range shading
vspan!(ax, 128, 512; color=(:green, 0.06))

# Lines
stairs!(ax, block_sizes, occ1; color="#2563EB", linewidth=2.0, step=:pre,
    label="32 regs/thread, 0 smem")
stairs!(ax, block_sizes, occ2; color="#DC2626", linewidth=2.0, linestyle=:dash, step=:pre,
    label="48 regs/thread, 0 smem")
stairs!(ax, block_sizes, occ3; color="#9558B2", linewidth=2.0, linestyle=:dot, step=:pre,
    label="32 regs/thread, 32 KB smem")

# Typical default annotation
vlines!(ax, [256]; color="#6B7280", linestyle=:dashdot, linewidth=1.0)
text!(ax, 260, 105; text="typical default", fontsize=9, color="#6B7280")

# Legend
axislegend(ax; position=:rb, labelsize=9, framevisible=true, padding=(6, 6, 4, 4))

# Recommended range label
text!(ax, 320, 4; text="recommended range", fontsize=8, color="#16A34A", align=(:center, :bottom))

# Save
save("occupancy-vs-blocksize_en.svg", fig)
save("occupancy-vs-blocksize_en.pdf", fig)

println("Figures saved: occupancy-vs-blocksize_en.{svg,pdf}")
