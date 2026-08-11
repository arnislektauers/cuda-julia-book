# --- begin:nn_layers ---
using CUDA, Random

# A dense layer stores its parameters directly as CuArrays. No custom
# host/device matrix wrapper is needed: a CuArray already owns its device
# memory, and Array(x) copies back to the host only when we ask for it.
# The layer caches the values the backward pass needs (its input and its
# pre-activation z), following the (features x batch) column-major convention.
mutable struct Dense
    W::CuMatrix{Float32}          # weights, shape (out, in)
    b::CuVector{Float32}          # bias, shape (out,)
    σ::Function                   # activation
    dσ::Function                  # activation derivative w.r.t. the pre-activation
    input::CuMatrix{Float32}      # cached input, filled by forward
    z::CuMatrix{Float32}          # cached pre-activation, filled by forward
end

# He initialization for the weights; biases start at zero.
function Dense(nin::Int, nout::Int, σ, dσ)
    W = CUDA.randn(Float32, nout, nin) .* sqrt(2.0f0 / nin)
    b = CUDA.zeros(Float32, nout)
    Dense(W, b, σ, dσ, CUDA.zeros(Float32, nin, 0), CUDA.zeros(Float32, nout, 0))
end

# Activations and their derivatives are ordinary scalar functions; broadcasting
# (the dotted calls in the forward and backward passes) runs them on the GPU.
sigmoid(z) = 1.0f0 / (1.0f0 + exp(-z))
dsigmoid(z) = (s = sigmoid(z); s * (1.0f0 - s))
relu(z) = max(z, 0.0f0)
drelu(z) = z > 0.0f0 ? 1.0f0 : 0.0f0
# --- end:nn_layers ---

# --- begin:nn_forward ---
# One layer: Z = W*A .+ b (a cuBLAS GEMM plus a broadcast that adds the bias
# to every column), then the elementwise activation A = σ.(Z).
function forward!(layer::Dense, A::CuMatrix{Float32})
    layer.input = A
    layer.z = layer.W * A .+ layer.b
    return layer.σ.(layer.z)
end

# The whole network: thread the batch through each layer in turn.
function forward!(layers::Vector{Dense}, X::CuMatrix{Float32})
    A = X
    for layer in layers
        A = forward!(layer, A)
    end
    return A
end
# --- end:nn_forward ---

# --- begin:nn_backward ---
# Binary cross-entropy loss and its gradient. Predictions are clamped away from
# 0 and 1 so the logarithms stay finite; both are pure broadcast + reduction.
function bce_loss(pred::CuMatrix{Float32}, target::CuMatrix{Float32})
    m = size(pred, 2)
    p = clamp.(pred, 1.0f-7, 1.0f0 - 1.0f-7)
    return -sum(target .* log.(p) .+ (1.0f0 .- target) .* log.(1.0f0 .- p)) / m
end

bce_grad(pred::CuMatrix{Float32}, target::CuMatrix{Float32}) =
    (p = clamp.(pred, 1.0f-7, 1.0f0 - 1.0f-7);
     @. -(target / p - (1.0f0 - target) / (1.0f0 - p)))

# Backprop through one layer. Given dA (the gradient w.r.t. this layer's
# output) it applies the chain rule through the activation, propagates the
# gradient to the previous layer, and updates W and b in place by SGD.
function backward!(layer::Dense, dA::CuMatrix{Float32}, lr::Float32)
    m = size(dA, 2)
    dZ = dA .* layer.dσ.(layer.z)                  # through the activation
    dA_prev = layer.W' * dZ                         # to the previous layer
    layer.W .-= lr .* (dZ * layer.input') ./ m      # dW = dZ Aᵀ / m
    layer.b .-= lr .* vec(sum(dZ; dims = 2)) ./ m   # db = mean of dZ over the batch
    return dA_prev
end

# Backprop through the network, seeded by the loss gradient at the output.
function backward!(layers::Vector{Dense}, dY::CuMatrix{Float32}, lr::Float32)
    dA = dY
    for layer in Iterators.reverse(layers)
        dA = backward!(layer, dA, lr)
    end
    return nothing
end
# --- end:nn_backward ---

# --- begin:nn_training ---
# Synthetic quadrant-classification data: label 1 when the two coordinates
# share a sign (quadrants 1 and 3), 0 otherwise (quadrants 2 and 4). Data is
# built on the host and moved to the device once per batch.
function make_batch(batch::Int)
    X = rand(Float32, 2, batch) .- 0.5f0
    y = Float32.((X[1, :] .> 0) .== (X[2, :] .> 0))
    return CuMatrix(X), CuMatrix(reshape(y, 1, batch))
end

function neural_net_main()
    lr = 0.5f0
    layers = [Dense(2, 30, relu, drelu),
              Dense(30, 1, sigmoid, dsigmoid)]
    batches = [make_batch(100) for _ in 1:21]

    for epoch in 1:1000
        cost = 0.0f0
        for (X, y) in batches[1:end - 1]
            Ŷ = forward!(layers, X)
            backward!(layers, bce_grad(Ŷ, y), lr)
            cost += bce_loss(Ŷ, y)
        end
        epoch % 100 == 0 &&
            println("Epoch $epoch  cost $(cost / (length(batches) - 1))")
    end

    Xt, yt = batches[end]
    Ŷt = Array(forward!(layers, Xt))
    accuracy = sum((Ŷt .> 0.5f0) .== (Array(yt) .> 0.5f0)) / size(yt, 2)
    println("Accuracy: $accuracy")
end

neural_net_main()
# --- end:nn_training ---
