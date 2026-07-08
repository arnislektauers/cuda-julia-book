# Plot: Cross-vendor performance comparison for portable KernelAbstractions.jl
# kernels (Chapter 12, Figure 12.10).
#
# Two panels:
#   (a) SAXPY throughput (GB/s) for native vendor libraries vs KernelAbstractions.jl
#       on NVIDIA H100, AMD MI250, and Intel PVC.
#   (b) Matrix multiplication GFLOPS normalized to vendor-specific library
#       performance (cuBLAS / rocBLAS / oneMKL).
#
# This script does NOT measure on local hardware — multi-vendor numbers cannot
# be obtained from a single machine. It emits the published reference values
# from the Juliana evaluation [DeLaCalle & García, FGCS 2025, doi:10.1016/
# j.future.2025.107813] to a `.dat` file, which the visutwin YAML spec at
# figures/spec/cross-vendor-benchmarks.yaml reads via `table:` references.
#
# Run with `julia plot_cross_vendor_benchmarks.jl` (no GPU required). Outputs
# `cross-vendor-benchmarks.dat` next to this script.

using Printf

const DATA_PATH = joinpath(@__DIR__, "cross-vendor-benchmarks.dat")

# --- Reference values (Juliana study, DeLaCalle & García 2025) ------------------
# One row per vendor. Columns:
#   1. vendor_idx        bar position on the x-axis (1 = NVIDIA, 2 = AMD, 3 = Intel)
#   2. saxpy_native_gbps SAXPY throughput with vendor-native (CUDA.jl / AMDGPU.jl /
#                        oneAPI.jl) implementation, in GB/s
#   3. saxpy_ka_gbps     SAXPY throughput with KernelAbstractions.jl, in GB/s
#   4. matmul_native     Matrix multiplication performance normalized to the
#                        vendor-specific library (cuBLAS / rocBLAS / oneMKL) — by
#                        construction this is 1.0
#   5. matmul_ka         Matrix multiplication with KernelAbstractions.jl,
#                        normalized to the same vendor's native library

const VENDORS = (
    (idx = 1, name = "NVIDIA H100", saxpy_native = 2900.0, saxpy_ka = 2750.0,
     matmul_native = 1.00, matmul_ka = 0.95),
    (idx = 2, name = "AMD MI250",   saxpy_native = 1900.0, saxpy_ka = 1800.0,
     matmul_native = 1.00, matmul_ka = 0.94),
    (idx = 3, name = "Intel PVC",   saxpy_native = 1400.0, saxpy_ka = 1320.0,
     matmul_native = 1.00, matmul_ka = 0.93),
)

# --- Write .dat -----------------------------------------------------------------
open(DATA_PATH, "w") do io
    println(io, "# vendor_idx  saxpy_native_gbps  saxpy_ka_gbps  matmul_native  matmul_ka")
    for v in VENDORS
        @printf(io, "%d  %.2f  %.2f  %.2f  %.2f\n",
                v.idx, v.saxpy_native, v.saxpy_ka, v.matmul_native, v.matmul_ka)
    end
end

println("Wrote $(DATA_PATH)")
println()
@printf "%-12s  %12s  %12s  %12s  %12s\n" "Vendor" "SAXPY native" "SAXPY KA" "Matmul nat." "Matmul KA"
println(repeat("-", 70))
for v in VENDORS
    @printf "%-12s  %9.0f GB/s  %9.0f GB/s  %12.2f  %12.2f\n" v.name v.saxpy_native v.saxpy_ka v.matmul_native v.matmul_ka
end
