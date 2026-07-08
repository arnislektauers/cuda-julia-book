# Enterprise deployment -- private registries and benchmarking

using LocalRegistry

# --- begin:local_registry ---
using LocalRegistry

# Create a new private registry (once)
create_registry("CompanyGPU",
    "git@github.com:company/CompanyGPU-Registry.git";
    description = "Internal GPU libraries"
)

# Register a private package
register("CUDAReductions";
    registry = "CompanyGPU",
    repo = "git@github.com:company/CUDAReductions.jl.git"
)
# --- end:local_registry ---

# --- begin:perf_benchmarks ---
using BenchmarkTools, CUDA, CUDAReductions

suite = BenchmarkGroup()
for N in [1_000, 100_000, 10_000_000]
    x = CUDA.rand(Float32, N)
    suite["reduce_sum_$N"] = @benchmarkable begin
        parallel_reduce(+, $x)
        CUDA.synchronize()
    end
end

results = run(suite; verbose=true)
# --- end:perf_benchmarks ---
