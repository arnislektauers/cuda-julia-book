# Testing and CI for GPU packages

using Test, CUDA

# --- begin:unit_tests ---
# test/runtests.jl
using Test
using CUDA
using CUDAReductions

@testset "CUDAReductions.jl" begin

    @testset "parallel_reduce" begin
        # Test with known values
        x = CuArray(Float32[1, 2, 3, 4, 5])
        @test parallel_reduce(+, x) ≈ 15.0f0
        @test parallel_reduce(max, x) ≈ 5.0f0
        @test parallel_reduce(min, x) ≈ 1.0f0
        @test parallel_reduce(*, x) ≈ 120.0f0

        # Test against CPU reference for large arrays
        N = 1_000_000
        x_cpu = rand(Float32, N)
        x_gpu = CuArray(x_cpu)

        @test parallel_reduce(+, x_gpu) ≈ sum(x_cpu) rtol=1e-4
        @test parallel_reduce(max, x_gpu) ≈ maximum(x_cpu)

        # Edge cases
        @test parallel_reduce(+, CuArray(Float32[42.0])) ≈ 42.0f0
        @test_throws ArgumentError parallel_reduce(+, CuArray(Float32[]))
    end

    @testset "type genericity" begin
        for T in (Float16, Float32, Float64)
            x = CuArray(T[1, 2, 3, 4])
            result = parallel_reduce(+, x)
            @test result isa T
            @test result ≈ T(10)
        end
    end

end
# --- end:unit_tests ---

# --- begin:cross_backend_tests ---
# test/runtests.jl
using Test, CUDA, CUDAReductions
import Pkg

# AMDGPU is optional: load it only when the test environment declares it
if haskey(Pkg.project().dependencies, "AMDGPU")
    using AMDGPU
end

@testset "CUDAReductions.jl" begin
    # CPU tests (always run)
    @testset "CPU fallback" begin
        x = [1.0f0, 2.0f0, 3.0f0, 4.0f0]
        @test parallel_reduce(+, x) ≈ 10.0f0
    end

    # CUDA tests (run only if GPU available)
    if CUDA.functional()
        @testset "CUDA backend" begin
            x = CuArray(Float32[1, 2, 3, 4])
            @test parallel_reduce(+, x) ≈ 10.0f0
        end
    else
        @warn "CUDA not available, skipping GPU tests"
    end

    # AMDGPU tests (run only if ROCm available)
    if @isdefined(AMDGPU) && AMDGPU.functional()
        @testset "AMDGPU backend" begin
            x = ROCArray(Float32[1, 2, 3, 4])
            @test parallel_reduce(+, x) ≈ 10.0f0
        end
    end
end
# --- end:cross_backend_tests ---
