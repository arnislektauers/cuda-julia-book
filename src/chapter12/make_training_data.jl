# Generates the synthetic dataset consumed by full_pipeline.jl.
#
# full_pipeline.jl opens with `CSV.read("training_data.csv", DataFrame)` to
# demonstrate a data-ingestion stage, but no such dataset ships with the book.
# This script writes one: two overlapping Gaussian blobs offset along both
# features, which is separable enough for the 2 -> 32 -> 16 -> 1 network to
# learn and still produces a visible decision boundary in the Stage 4 scatter.
#
# Run this before full_pipeline.jl; both use the working directory for I/O.
#
# Not included by any chapter -- a companion to retained, unpublished source.

using DataFrames, CSV, Random

Random.seed!(42)

n = 2000
label = rand(0:1, n)

# Class 1 is shifted +1.5 in both features; the overlap keeps the task
# non-trivial so the training loss curve actually has something to descend.
df = DataFrame(feature1 = randn(n) .+ 1.5 .* label,
               feature2 = randn(n) .+ 1.5 .* label,
               label    = label)

CSV.write("training_data.csv", df)
println("wrote training_data.csv ($(nrow(df)) rows, ",
        "$(count(==(1), label)) positive)")
