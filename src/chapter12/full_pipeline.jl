# End-to-end pipeline: Data ingestion -> GPU training -> Visualization

# --- begin:full_pipeline ---
using Flux, CUDA, cuDNN, DataFrames, CSV, CairoMakie, Statistics

# Stage 1: Data Ingestion
df = CSV.read("training_data.csv", DataFrame)
X = Float32.(hcat(df.feature1, df.feature2)')
y = Float32.(reshape(df.label, 1, :))

# Stage 2: GPU Transfer
X_gpu = CuArray(X)
y_gpu = CuArray(y)

# Stage 3: Model Definition and Training
model = Chain(
    Dense(2 => 32, relu),
    Dense(32 => 16, relu),
    Dense(16 => 1, sigmoid)
) |> gpu

loss(m, x, y) = Flux.binarycrossentropy(m(x), y)
opt = Flux.setup(Adam(0.001f0), model)

losses = Float32[]
for epoch in 1:100
    l, grads = Flux.withgradient(loss, model, X_gpu, y_gpu)
    Flux.update!(opt, model, grads[1])
    push!(losses, l)
end

# Stage 4: Visualization
fig = Figure(size=(800, 400))
ax1 = Axis(fig[1, 1], xlabel="Epoch", ylabel="Loss", title="Training Loss")
lines!(ax1, 1:length(losses), losses)

ax2 = Axis(fig[1, 2], xlabel="Feature 1", ylabel="Feature 2",
           title="Decision Boundary")
scatter!(ax2, Array(X_gpu[1, :]), Array(X_gpu[2, :]),
         color=vec(Array(y_gpu)), colormap=:viridis, markersize=3)
save("training_results.pdf", fig)
# --- end:full_pipeline ---
