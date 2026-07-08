# Multi-GPU deep learning and multi-node scaling

using CUDA

# --- begin:lux_distributed ---
using Lux, CUDA, MPI, NCCL, Random, Optimisers, Zygote

# Initialize distributed backend
DistributedUtils.initialize(NCCLBackend)
backend = DistributedUtils.get_distributed_backend(NCCLBackend)

# Create and replicate model
model = Chain(Dense(784, 256, relu), Dense(256, 10))
rng = Random.default_rng()
dev = gpu_device()  # From MLDataDevices, re-exported by Lux
ps, st = Lux.setup(rng, model) |> dev

# Synchronize initial parameters from rank 0
ps = DistributedUtils.synchronize!!(backend, ps)
st = DistributedUtils.synchronize!!(backend, st)

# Wrap optimizer for distributed gradient averaging
opt = DistributedUtils.DistributedOptimizer(backend, Adam(0.001f0))
opt_state = Optimisers.setup(opt, ps)

# Distribute dataset across ranks
data = DistributedUtils.DistributedDataContainer(backend, training_data)

# Training loop
for epoch in 1:num_epochs
    for (x, y) in data
        x, y = x |> dev, y |> dev
        function loss_fn(p)
            ŷ, st_ = Lux.apply(model, x, p, st)
            return sum(abs2, ŷ .- y)
        end
        loss, back = Zygote.pullback(loss_fn, ps)
        gs = back(one(loss))[1]
        opt_state, ps = Optimisers.update(opt_state, ps, gs)
    end
end

# Only save/log on the master rank
if DistributedUtils.local_rank(backend) == 0
    # Save model checkpoint
end
# --- end:lux_distributed ---

# --- begin:hybrid_mpi_nccl ---
using MPI, CUDA

MPI.Init()
comm = MPI.COMM_WORLD
global_rank = MPI.Comm_rank(comm)
world_size = MPI.Comm_size(comm)

# Assign local GPU based on rank within the node
local_rank = global_rank % length(CUDA.devices())
CUDA.device!(local_rank)

println("Rank $global_rank / $world_size -> GPU $local_rank "
      * "($(CUDA.name(CUDA.device())))")

# Compute on local GPU
sendbuf = CUDA.rand(Float32, 100_000)
recvbuf = CUDA.zeros(Float32, 100_000)

# AllReduce across all nodes and GPUs
MPI.Allreduce!(sendbuf, recvbuf, +, comm)

MPI.Finalize()
# --- end:hybrid_mpi_nccl ---

# --- begin:hierarchical_allreduce ---
# One process per GPU: every MPI rank drives exactly one device
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# Node-local communicator groups the ranks sharing this node
node_comm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)
local_rank = MPI.Comm_rank(node_comm)
ngpus_per_node = MPI.Comm_size(node_comm)
CUDA.device!(local_rank)

# One NCCL communicator per node, bootstrapped over node_comm
uid = local_rank == 0 ? NCCL.UniqueID() : nothing
uid = MPI.bcast(uid, node_comm; root=0)
nccl_comm = NCCL.Communicator(ngpus_per_node, uid, local_rank)

function hierarchical_allreduce!(data, nccl_comm, comm,
                                 node_comm, local_rank, rank)
    chunk_size = length(data) ÷ NCCL.size(nccl_comm)

    # Step 1: Intra-node ReduceScatter via NCCL
    local_chunk = similar(data, chunk_size)
    NCCL.ReduceScatter!(data, local_chunk, +, nccl_comm)

    # Step 2: Inter-node AllReduce of the local chunk via MPI,
    # among the ranks with the same local rank on every node
    peer_comm = MPI.Comm_split(comm, local_rank, rank)
    MPI.Allreduce!(MPI.IN_PLACE, local_chunk, +, peer_comm)

    # Step 3: Intra-node AllGather via NCCL
    NCCL.Allgather!(local_chunk, data, nccl_comm)
end
# --- end:hierarchical_allreduce ---
