# Machine learning -- Lux.jl GPU integration
#
# Kept separate from ml_flux.jl: Flux and Lux both export names such as
# `Chain` and `Dense`, so a single script that loads both leaves those
# names unbound and fails with an UndefVarError.

# --- begin:lux_cnn ---
# cuDNN is the trigger package that activates the CUDA backend: without it
# `gpu_device()` returns a CPU device and the parameters never leave the host.
using Lux, CUDA, cuDNN, Random, Optimisers, Zygote, OneHotArrays

# Define model architecture (no parameters yet)
model = Chain(
    Conv((3, 3), 1 => 16, relu; pad=1),
    MaxPool((2, 2)),
    Conv((3, 3), 16 => 32, relu; pad=1),
    MaxPool((2, 2)),
    FlattenLayer(),
    Dense(32 * 7 * 7, 128, relu),
    Dense(128, 10)
)

# Initialize parameters and state
rng = Random.default_rng()
ps, st = Lux.setup(rng, model)

# Move to GPU using device API
gdev = gpu_device()
ps = ps |> gdev
st = st |> gdev

# Training data on GPU
x = randn(Float32, 28, 28, 1, 64) |> gdev
y = onehotbatch(rand(0:9, 64), 0:9) |> gdev

# Forward pass with explicit parameters
ŷ, st = Lux.apply(model, x, ps, st)
# --- end:lux_cnn ---
