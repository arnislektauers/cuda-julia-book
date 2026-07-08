# --- begin:nn_matrix ---
using CUDA

struct Shape
    x::Int
    y::Int
end

mutable struct NNMatrix
    shape::Shape
    dataDevice::Union{CuArray{Float32},Nothing}
    dataHost::Union{Array{Float32},Nothing}
    deviceAllocated::Bool
    hostAllocated::Bool
end

NNMatrix(xDim::Int = 1, yDim::Int = 1) = NNMatrix(
    Shape(xDim, yDim), nothing, nothing, false, false
)

Base.getindex(m::NNMatrix, i::Int) = (m.dataHost === nothing) ? error("Host not allocated") : m.dataHost[i]
Base.setindex!(matrix::NNMatrix, value::Float32, index::Int) = 
    (matrix.dataHost[index] = value)

function allocateCudaMemory!(matrix::NNMatrix)
    if !matrix.deviceAllocated
        shape = matrix.shape
        matrix.dataDevice = CuArray{Float32}(undef, shape.x * shape.y)
        matrix.deviceAllocated = true
    end
end

function allocateHostMemory!(matrix::NNMatrix)
    if !matrix.hostAllocated
        shape = matrix.shape
        matrix.dataHost = Array{Float32}(undef, shape.x * shape.y)
        matrix.hostAllocated = true
    end
end

function allocateMemory!(matrix::NNMatrix)
    allocateCudaMemory!(matrix)
    allocateHostMemory!(matrix)
end

function allocateMemoryIfNotAllocated!(matrix::NNMatrix, shape::Shape)
    matrix.shape = shape
    if !matrix.deviceAllocated
        allocateCudaMemory!(matrix)
    end
    if !matrix.hostAllocated
        allocateHostMemory!(matrix)
    end
end

function copyHostToDevice!(matrix::NNMatrix)
    if matrix.deviceAllocated && matrix.hostAllocated
        copyto!(matrix.dataDevice, matrix.dataHost)
    else
        error("Cannot copy host data: memory not allocated on host/device.")
    end
end

function copyDeviceToHost!(matrix::NNMatrix)
    if matrix.deviceAllocated && matrix.hostAllocated
        copyto!(matrix.dataHost, matrix.dataDevice)
    else
        error("Cannot copy device data: memory not allocated on host/device.")
    end
end
# --- end:nn_matrix ---

# --- begin:nn_sigmoid_layer ---
using Random

abstract type NNLayer end

@kwdef mutable struct SigmoidActivation <: NNLayer
    A::NNMatrix = NNMatrix()
    Z::NNMatrix = NNMatrix()
    dZ::NNMatrix = NNMatrix()
end

sigmoid(x::Float32) = 1.0f0 / (1.0f0 + exp(-x))

function sigmoidActivationForward!(Z::CuDeviceVector{Float32}, 
                                   A::CuDeviceVector{Float32}, 
                                   zxDim, zyDim)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds if index <= zxDim * zyDim
         A[index] = sigmoid(Z[index])
    end
    return
end

function sigmoidActivationBackprop!(Z::CuDeviceVector{Float32}, 
                                    dA::CuDeviceVector{Float32}, 
                                    dZ::CuDeviceVector{Float32}, zxDim, zyDim)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds if index <= zxDim * zyDim
        sigma = sigmoid(Z[index])
        dZ[index] = dA[index] * sigma * (1 - sigma)
    end
    return
end

function forward!(layer::SigmoidActivation, Z::NNMatrix)
    layer.Z = Z
    allocateMemoryIfNotAllocated!(layer.A, Z.shape)

    blockSize = 256
    numblocks = (Z.shape.y * Z.shape.x + blockSize - 1) ÷ blockSize

    @cuda(blocks = numblocks, threads = blockSize,
          sigmoidActivationForward!(
              Z.dataDevice, layer.A.dataDevice, Z.shape.x, Z.shape.y))

    return layer.A
end

function backprop!(layer::SigmoidActivation, dA::NNMatrix, 
                   learningRate::Float32=Float32(0.01))
    Z = layer.Z
    dZ = layer.dZ

    allocateMemoryIfNotAllocated!(dZ, Z.shape)

    blockSize = 256
    numblocks = (Z.shape.y * Z.shape.x + blockSize - 1) ÷ blockSize

    @cuda(blocks = numblocks, threads = blockSize,
          sigmoidActivationBackprop!(
              Z.dataDevice, dA.dataDevice, dZ.dataDevice,
              Z.shape.x, Z.shape.y))

    return dZ
end
# --- end:nn_sigmoid_layer ---

# --- begin:nn_relu_layer ---
@kwdef mutable struct ReLUActivation <: NNLayer
    A::NNMatrix = NNMatrix()
    Z::NNMatrix = NNMatrix()
    dZ::NNMatrix = NNMatrix()
end

function reluActivationForward!(Z::CuDeviceVector{Float32}, 
                                A::CuDeviceVector{Float32}, 
                                zxDim, zyDim)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds if index <= zxDim * zyDim
        A[index] = max(Z[index], 0.0f0)
    end
    return
end

function reluActivationBackprop!(Z::CuDeviceVector{Float32}, 
                                 dA::CuDeviceVector{Float32}, 
                                 dZ::CuDeviceVector{Float32}, zxDim, zyDim)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds if index <= zxDim * zyDim
        if Z[index] > 0
            dZ[index] = dA[index]
        else
            dZ[index] = 0
        end
    end
    return
end

function forward!(layer::ReLUActivation, Z::NNMatrix)
    layer.Z = Z

    A = layer.A
    allocateMemoryIfNotAllocated!(A, Z.shape)

    blockSize = 256
    numblocks = (Z.shape.y * Z.shape.x + blockSize - 1) ÷ blockSize

    @cuda(blocks = numblocks, threads = blockSize,
          reluActivationForward!(
              Z.dataDevice, A.dataDevice, Z.shape.x, Z.shape.y))

    return A
end

function backprop!(layer::ReLUActivation, dA::NNMatrix,
                   learningRate::Float32=Float32(0.01))
    Z = layer.Z
    dZ = layer.dZ
    allocateMemoryIfNotAllocated!(dZ, Z.shape)

    blockSize = 256
    numblocks = (Z.shape.y * Z.shape.x + blockSize - 1) ÷ blockSize

    @cuda(blocks = numblocks, threads = blockSize,
          reluActivationBackprop!(
              Z.dataDevice, dA.dataDevice, dZ.dataDevice,
              Z.shape.x, Z.shape.y))

    return dZ
end
# --- end:nn_relu_layer ---

# --- begin:nn_linear_layer ---
function initializeWeightsRandomly!(W::NNMatrix)
    randn!(W.dataHost)
    
    # Use Kaiming He initialization
    W.dataHost .*= sqrt(2.0f0 / W.shape.x)

    copyHostToDevice!(W)
end

mutable struct LinearLayer <: NNLayer
    W::NNMatrix
    b::NNMatrix

    Z::NNMatrix
    A::NNMatrix
    dA::NNMatrix

    function LinearLayer(shape::Shape)
        println("Linear layer $(shape.x)x$(shape.y)")
        W = NNMatrix(shape.x, shape.y)
        b = NNMatrix(shape.y, 1)

        allocateMemory!(W)
        allocateMemory!(b)

        initializeWeightsRandomly!(W)
        
        new(W, b, NNMatrix(), NNMatrix(), NNMatrix())
    end
end

@inline idx_cm(row, col, nrows) = (col - 1) * nrows + row  # 1-based

function linearLayerForward!(W::CuDeviceVector{Float32}, 
                             A::CuDeviceVector{Float32}, 
                             Z::CuDeviceVector{Float32},
                             b::CuDeviceVector{Float32},
                             wxDim, wyDim, axDim)

    col = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    row = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    zValue = 0.0f0
    @inbounds if row <= wyDim && col <= axDim
        for i in 1:wxDim
            zValue += W[idx_cm(i, row, wxDim)] * A[idx_cm(col, i, axDim)]
        end
        Z[idx_cm(col, row, axDim)] = zValue + b[row]
    end

    return
end

function linearLayerBackprop!(W::CuDeviceVector{Float32}, 
                              dZ::CuDeviceVector{Float32}, 
                              dA::CuDeviceVector{Float32},
							  wxDim, wyDim, batchSize) 

	col = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # batch index
    row = (blockIdx().y - 1) * blockDim().y + threadIdx().y  # input feature

    acc = 0.0f0
    @inbounds if col <= batchSize && row <= wxDim
        for j in 1:wyDim
            acc += dZ[idx_cm(col, j, batchSize)] *
                   W[idx_cm(row, j, wxDim)]
        end
        dA[idx_cm(col, row, batchSize)] = acc
    end

    return
end

function linearLayerUpdateWeights!(dZ::CuDeviceVector{Float32}, 
                                   A::CuDeviceVector{Float32}, 
                                   W::CuDeviceVector{Float32},
								   batchSize, outDim, inDim, 
                                   learningRate)
	col = (blockIdx().x - 1) * blockDim().x + threadIdx().x  # input feature
    row = (blockIdx().y - 1) * blockDim().y + threadIdx().y  # output neuron

	acc = 0.0f0
	@inbounds if col <= inDim && row <= outDim
		for i in 1:batchSize
			acc += A[idx_cm(i, col, batchSize)] * dZ[idx_cm(i, row, batchSize)]
        end
		W[idx_cm(col, row, inDim)] -= learningRate * (acc / batchSize)
    end

    return
end

function linearLayerUpdateBias!(dZ::CuDeviceVector{Float32}, 
                                b::CuDeviceVector{Float32},
                                batchSize, outDim, learningRate)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    acc = 0.0f0

    @inbounds if j <= outDim
        for i in 1:batchSize
            acc += dZ[idx_cm(i, j, batchSize)]
        end
        b[j] -= learningRate * (acc / batchSize)
    end

    return
end

function computeAndStoreLayerOutput!(layer::LinearLayer, A::NNMatrix)
    Z = layer.Z
    W = layer.W

    blockSize = (8, 8)
    numOfBlocks = ((Z.shape.x + blockSize[1] - 1) ÷ blockSize[1],
                   (Z.shape.y + blockSize[2] - 1) ÷ blockSize[2])

    @cuda(blocks = numOfBlocks, threads = blockSize,
          linearLayerForward!(
              W.dataDevice, A.dataDevice, Z.dataDevice,
              layer.b.dataDevice,
              W.shape.x, W.shape.y, A.shape.x))
end

function forward!(layer::LinearLayer, A::NNMatrix)
    @assert layer.W.shape.x == A.shape.y

    layer.A = A

    Z = layer.Z

    zShape = Shape(A.shape.x, layer.W.shape.y)
    allocateMemoryIfNotAllocated!(Z, zShape)

    computeAndStoreLayerOutput!(layer, A)

    return Z
end

function computeAndStoreBackpropError!(layer::LinearLayer, dZ::NNMatrix)
    A = layer.A
    W = layer.W

    blockSize = (8, 8)
    numOfBlocks = ((dZ.shape.x + blockSize[1] - 1) ÷ blockSize[1],
                   (W.shape.x  + blockSize[2] - 1) ÷ blockSize[2])

    @cuda(blocks = numOfBlocks, threads = blockSize,
          linearLayerBackprop!(
              W.dataDevice, dZ.dataDevice, layer.dA.dataDevice,
              W.shape.x, W.shape.y, dZ.shape.x))
end

function updateBias!(layer::LinearLayer, dZ::NNMatrix, learningRate)
    b = layer.b

    blockSize = 256
    numOfBlocks = (dZ.shape.y + 255) ÷ 256

    @cuda(blocks = numOfBlocks, threads = blockSize,
          linearLayerUpdateBias!(
              dZ.dataDevice, b.dataDevice, dZ.shape.x, dZ.shape.y,
              learningRate))
end

function updateWeights!(layer::LinearLayer, dZ::NNMatrix, learningRate)
    A = layer.A
    W = layer.W

    blockSize = (8, 8)
    numOfBlocks = ((W.shape.x + blockSize[1] - 1) ÷ blockSize[1],
                   (W.shape.y + blockSize[2] - 1) ÷ blockSize[2])

    @cuda(blocks = numOfBlocks, threads = blockSize,
          linearLayerUpdateWeights!(
              dZ.dataDevice, A.dataDevice, W.dataDevice,
              dZ.shape.x, dZ.shape.y,
              A.shape.y, learningRate))
end

function backprop!(layer::LinearLayer, dZ::NNMatrix, learningRate::Float32=0.01f0)
    dA = layer.dA

    allocateMemoryIfNotAllocated!(dA, layer.A.shape)

    computeAndStoreBackpropError!(layer, dZ)

    updateBias!(layer, dZ, learningRate)

    updateWeights!(layer, dZ, learningRate)

    return dA
end
# --- end:nn_linear_layer ---

# --- begin:nn_bce ---
@inline clamp_device(x::Float32, lo::Float32, hi::Float32) =
	ifelse(x < lo, lo, ifelse(x > hi, hi, x))

function binaryCrossEntropyCost!(predictions::CuDeviceArray{Float32}, 
                                 target::CuDeviceArray{Float32},
                                 size, 
                                 cost::CuDeviceArray{Float32})
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    @inbounds if index <= size
        p = clamp_device(predictions[index], 1.0f-7, 1.0f0 - 1.0f-7)
        partialCost = target[index] * log(p) +
                      (1.0f0 - target[index]) * log(1.0f0 - p)
        CUDA.@atomic cost[1] += -partialCost / size
    end
    return
end

function dBinaryCrossEntropyCost!(predictions::CuDeviceArray{Float32}, 
                                  target::CuDeviceArray{Float32},
                                  dY::CuDeviceArray{Float32}, size)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    @inbounds if index <= size
        p = clamp_device(predictions[index], 1.0f-7, 1.0f0 - 1.0f-7)
        dY[index] = -1.0f0 * (target[index] / p -
                              (1.0f0 - target[index]) / (1.0f0 - p))
    end
    return
end

function bceCost(predictions::NNMatrix, target::NNMatrix)
    @assert predictions.shape.x == target.shape.x

    cost = CuVector{Float32, CUDA.UnifiedMemory}([0.0])

    blockSize = 256
    numOfBlocks = (predictions.shape.x + blockSize - 1) ÷ blockSize

    @cuda blocks = numOfBlocks threads = blockSize binaryCrossEntropyCost!(
        predictions.dataDevice, target.dataDevice, predictions.shape.x, cost
    )

    CUDA.synchronize()  # ensure the atomic accumulation has completed

    costValue = cost[1]

    return costValue
end

function bceDCost(predictions::NNMatrix, target::NNMatrix, dY::NNMatrix)
    @assert predictions.shape.x == target.shape.x

    blockSize = 256
    numOfBlocks = (predictions.shape.x + blockSize - 1) ÷ blockSize
    
    @cuda blocks = numOfBlocks threads = blockSize dBinaryCrossEntropyCost!(
        predictions.dataDevice, target.dataDevice, dY.dataDevice, 
        predictions.shape.x
    )

    return dY
end
# --- end:nn_bce ---

# --- begin:nn_network ---
@kwdef mutable struct NeuralNetwork
    layers::Vector{NNLayer} = NNLayer[]
    Y::NNMatrix = NNMatrix()
    dY::NNMatrix = NNMatrix()
    learningRate::Float32 = 0.01f0
end

function forward!(network::NeuralNetwork, X::NNMatrix)
    Z = X

    for layer in network.layers
        Z = forward!(layer, Z)
    end

    network.Y = Z

    return network.Y
end

function backprop!(network::NeuralNetwork, predictions::NNMatrix,
                   target::NNMatrix)
    allocateMemoryIfNotAllocated!(network.dY, predictions.shape)
    error = bceDCost(predictions, target, network.dY)

    for layer in reverse(network.layers)
        error = backprop!(layer, error, network.learningRate)
    end

    CUDA.synchronize()
end

function addLayer!(nn::NeuralNetwork, layer::T) where {T<:NNLayer} 
    push!(nn.layers, layer)
end
# --- end:nn_network ---
# --- begin:nn_training ---

function computeAccuracy(predictions::NNMatrix, targets::NNMatrix)
	m = predictions.shape.x
	correctPredictions = 0

	for i in 1:m
		prediction = predictions[i] > 0.5 ? 1 : 0
		if prediction == targets[i]
			correctPredictions += 1
        end
    end

	return correctPredictions / m
end

struct CoordinatesDataset
    batchSize::Int
	numberOfBatches::Int
    batches::Array{NNMatrix}
    targets::Array{NNMatrix}

    function CoordinatesDataset(batchSize::Int, numberOfBatches::Int)
        batches = Array{NNMatrix}(undef, 0)
        targets = Array{NNMatrix}(undef, 0)

        for i in 1:numberOfBatches     
            push!(batches, NNMatrix(batchSize, 2))
            push!(targets, NNMatrix(batchSize, 1))

            allocateMemory!(batches[i])
            allocateMemory!(targets[i])
    
            for k in 1:batchSize
                batch = batches[i]
                batch[k] = Float32(rand() - 0.5)
                batch[batch.shape.x + k] = Float32(rand() - 0.5)
    
                if (batch[k] > 0 && batch[batch.shape.x + k] > 0) || 
                    (batch[k] < 0 && batch[batch.shape.x + k] < 0)
                    targets[i][k] = Float32(1.0)
                else 
                    targets[i][k] = Float32(0.0)
                end
            end
    
            copyHostToDevice!(batches[i])
            copyHostToDevice!(targets[i])
        end

        new(batchSize, numberOfBatches, batches, targets)
    end
end

function neural_net_main()
    dataset = CoordinatesDataset(100, 21)

    nn = NeuralNetwork()
    addLayer!(nn, LinearLayer(Shape(2, 30)))
    addLayer!(nn, ReLUActivation())
    addLayer!(nn, LinearLayer(Shape(30, 1)))
    addLayer!(nn, SigmoidActivation())

    for epoch in 1:1000
        cost = 0.0

        for batch in 1:(dataset.numberOfBatches - 1)
            Y = forward!(nn, dataset.batches[batch])
            backprop!(nn, Y, dataset.targets[batch])
            cost += bceCost(Y, dataset.targets[batch])
        end

        if epoch % 100 == 0
            println("Epoch: $(epoch), Cost: $(cost / dataset.numberOfBatches)")
        end
    end

    Y = forward!(nn, dataset.batches[dataset.numberOfBatches])
    copyDeviceToHost!(Y)

    copyDeviceToHost!(dataset.targets[dataset.numberOfBatches])
    accuracy = computeAccuracy(Y, dataset.targets[dataset.numberOfBatches])
    println("Accuracy: $(accuracy)")
end
# --- end:nn_training ---

neural_net_main()