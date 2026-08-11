# Multi-package synergy -- end-to-end GPU workflow

using CUDA, DataFrames, Flux, CairoMakie, Statistics, Optimisers

# --- begin:time_series_pipeline ---
using CUDA, cuDNN, DataFrames, Flux, Optimisers, CairoMakie, Statistics

# -- Stage 1: Data Ingestion --
# Synthetic time series data (in practice: CSV.jl, Arrow.jl, etc.)
n = 100_000
t = Float32.(1:n)
signal = sin.(2π .* t ./ 1000) .+ 0.3f0 .* randn(Float32, n)
df = DataFrame(time=t, value=signal)

# -- Stage 2: GPU-Accelerated Preprocessing --
d_values = CuArray(df.value)

# Normalize on the GPU, then bring the result back for windowing
μ = mean(d_values)
σ = std(d_values)
normalized = Array((d_values .- μ) ./ σ)

# Create sliding windows (sequence length = 50, predict next value).
# Windowing is a host-side preprocessing step: element-by-element
# writes into a CuArray would require scalar indexing
seq_len = 50
n_windows = length(normalized) - seq_len

X = zeros(Float32, seq_len, n_windows)
y = zeros(Float32, 1, n_windows)

for i in 1:n_windows
    X[:, i] = normalized[i:i+seq_len-1]
    y[1, i] = normalized[i+seq_len]
end

# Move the windowed dataset to the GPU
d_X = cu(X)
d_y = cu(y)

# -- Stage 3: GPU Model Training --
# Train on the leading 80% of windows; hold out the trailing
# windows for evaluation
n_train = round(Int, 0.8 * n_windows)

model = Chain(
    Dense(seq_len, 64, relu),
    Dense(64, 32, relu),
    Dense(32, 1)
) |> gpu

opt_state = Optimisers.setup(Adam(0.001f0), model)
batch_size = 256
n_epochs = 20

# Training loop. Keep it inside a function: a top-level `for` loop that
# reassigns `model` would create a new local instead of updating the global,
# and working through globals is slow in any case.
function train!(model, opt_state, d_X, d_y, n_train, batch_size, n_epochs)
    for epoch in 1:n_epochs
        # Mini-batch training
        for start in 1:batch_size:n_train-batch_size
            idx = start:start+batch_size-1
            x_batch = d_X[:, idx]
            y_batch = d_y[:, idx]

            grads = Flux.gradient(model) do m
                Flux.mse(m(x_batch), y_batch)
            end

            opt_state, model = Optimisers.update(opt_state, model, grads[1])
        end
    end
    return model, opt_state
end

model, opt_state = train!(model, opt_state, d_X, d_y,
                          n_train, batch_size, n_epochs)

# -- Stage 4: Visualization --
# Predict on held-out windows after the training range
test_start = n_train + 1
test_X = d_X[:, test_start:test_start+999]
predictions = Array(model(test_X))'
actuals = Array(d_y[:, test_start:test_start+999])'

fig = Figure(size=(700, 400))
ax = Axis(fig[1, 1]; xlabel="Time Step", ylabel="Normalized Value",
          title="GPU-Trained Time Series Forecast")
lines!(ax, 1:1000, actuals[:]; label="Actual", color=:blue)
lines!(ax, 1:1000, predictions[:]; label="Predicted",
       color=:red, linestyle=:dash)
axislegend(ax; position=:lt)
save("forecast.pdf", fig)
# --- end:time_series_pipeline ---
