# Machine learning -- Flux.jl and Lux.jl GPU integration

using Flux, Lux, CUDA, Random, Optimisers, Zygote

# --- begin:flux_cnn ---
using Flux, CUDA

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

# --- begin:lux_cnn ---
using Lux, CUDA, Random, Optimisers, Zygote, OneHotArrays

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

# --- begin:chainrules_gpu ---
using ChainRulesCore, CUDA

# Custom GPU operation: soft thresholding
function soft_threshold(x::CuArray, λ::Real)
    return sign.(x) .* max.(abs.(x) .- λ, 0)
end

# Define the backward rule for AD
function ChainRulesCore.rrule(::typeof(soft_threshold),
                               x::CuArray, λ::Real)
    y = soft_threshold(x, λ)
    function soft_threshold_pullback(ȳ)
        active = abs.(x) .> λ
        # Gradient for x: pass ȳ through active entries, 0 elsewhere
        dx = ȳ .* active
        # Gradient for λ: each active output shrinks by sign(x)
        dλ = -sum(ȳ .* sign.(x) .* active)
        return NoTangent(), dx, dλ
    end
    return y, soft_threshold_pullback
end
# --- end:chainrules_gpu ---

# --- begin:enzyme_gpu ---
using Enzyme, CUDA

function my_kernel!(y, x, α)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(x)
        @inbounds y[i] = α * x[i]^2 + sin(x[i])
    end
    return nothing
end

# Host wrapper that launches the kernel; Enzyme differentiates
# through the @cuda launch via CUDA.jl's Enzyme extension
function apply!(y, x, α)
    @cuda threads=256 blocks=cld(length(y), 256) my_kernel!(y, x, α)
    return nothing
end

# Enzyme generates the gradient kernel automatically
x = CUDA.rand(Float32, 1024)
y = CUDA.zeros(Float32, 1024)
dx = CUDA.zeros(Float32, 1024)
dy = CUDA.ones(Float32, 1024)

# Reverse-mode differentiation through the GPU kernel launch
Enzyme.autodiff(Reverse, apply!, Const, Duplicated(y, dy),
                Duplicated(x, dx), Const(2.0f0))
# --- end:enzyme_gpu ---

# --- begin:cnn_training ---
using Flux, CUDA, MLUtils, Optimisers, Statistics

# Model definition
model = Chain(
    Conv((5, 5), 1 => 6, relu),
    MaxPool((2, 2)),
    Conv((5, 5), 6 => 16, relu),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(256, 120, relu),
    Dense(120, 84, relu),
    Dense(84, 10)
) |> gpu

# Loss function
loss(model, x, y) = Flux.logitcrossentropy(model(x), y)

# Optimizer setup
opt_state = Optimisers.setup(Adam(0.001f0), model)

# Synthetic dataset: 512 grayscale 28x28 images with 10 class labels
X = randn(Float32, 28, 28, 1, 512)
Y = Flux.onehotbatch(rand(0:9, 512), 0:9)
train_loader = DataLoader((X, Y); batchsize=64)

# Training loop
for epoch in 1:10
    epoch_loss = 0.0f0
    n_batches = 0

    for (x, y) in train_loader    # DataLoader provides mini-batches
        x, y = x |> gpu, y |> gpu

        # Compute gradients
        grads = Flux.gradient(m -> loss(m, x, y), model)

        # Update parameters
        opt_state, model = Optimisers.update(opt_state, model, grads[1])

        epoch_loss += loss(model, x, y)
        n_batches += 1
    end

    println("Epoch $epoch: loss = $(epoch_loss / n_batches)")
end
# --- end:cnn_training ---
