# Machine learning -- end-to-end Flux.jl training loop on the GPU
#
# Not included by any chapter; retained as a worked reference.
# Kept apart from ml_flux.jl because that file binds `loss` as a variable
# while this one defines it as a function, which cannot coexist in one module.

# --- begin:cnn_training ---
using Flux, CUDA, cuDNN, MLUtils, Optimisers, Statistics

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

# Training loop. Keep it inside a function: a top-level `for` loop that
# reassigns `model` would create a new local instead of updating the global,
# and working through globals is slow in any case.
function train!(model, opt_state, train_loader, n_epochs)
    for epoch in 1:n_epochs
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
    return model, opt_state
end

model, opt_state = train!(model, opt_state, train_loader, 10)
# --- end:cnn_training ---
