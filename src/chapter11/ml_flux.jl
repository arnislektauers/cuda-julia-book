# Machine learning -- Flux.jl GPU integration
#
# Kept separate from ml_lux.jl: Flux and Lux both export names such as
# `Chain` and `Dense`, so a single script that loads both leaves those
# names unbound and fails with an UndefVarError.

# --- begin:flux_cnn ---
# cuDNN is the trigger package that activates the CUDA backend: without it
# `gpu` silently returns the model unchanged and training runs on the CPU.
using Flux, CUDA, cuDNN

# Define model (CPU)
model = Chain(
    Conv((3, 3), 1 => 16, relu; pad=1),
    MaxPool((2, 2)),
    Conv((3, 3), 16 => 32, relu; pad=1),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(32 * 7 * 7, 128, relu),
    Dense(128, 10)
)

# Move model to GPU
model = model |> gpu    # All parameters become CuArray

# Data must also be on GPU
x = randn(Float32, 28, 28, 1, 64) |> gpu    # Batch of 64 images
y = Flux.onehotbatch(rand(0:9, 64), 0:9) |> gpu

# Forward pass executes entirely on GPU
ŷ = model(x)
loss = Flux.logitcrossentropy(ŷ, y)
# --- end:flux_cnn ---
