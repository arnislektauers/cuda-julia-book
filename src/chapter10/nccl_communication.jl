# NCCL collective communication patterns

using CUDA, NCCL

# --- begin:nccl_communicators ---
using CUDA, NCCL

# Create communicators for all visible GPUs
devs = collect(CUDA.devices())
comms = NCCL.Communicators(devs)

# Query communicator properties
for comm in comms
    println("Rank $(NCCL.rank(comm)), "
          * "Device $(NCCL.device(comm)), "
          * "Size $(NCCL.size(comm))")
end
# --- end:nccl_communicators ---

# --- begin:nccl_allreduce ---
using CUDA, NCCL

devs = collect(CUDA.devices())
comms = NCCL.Communicators(devs)

# Prepare per-device buffers
sendbufs = Vector{CuVector{Float32}}(undef, length(devs))
recvbufs = Vector{CuVector{Float32}}(undef, length(devs))
for (i, dev) in enumerate(devs)
    CUDA.device!(dev)
    sendbufs[i] = CUDA.fill(Float32(i), 1024)
    recvbufs[i] = CUDA.zeros(Float32, 1024)
end

# AllReduce: sum across all devices
NCCL.group() do
    for (i, dev) in enumerate(devs)
        CUDA.device!(dev)
        NCCL.Allreduce!(sendbufs[i], recvbufs[i], +, comms[i])
    end
end
# Every recvbufs[i] now contains the element-wise sum
# --- end:nccl_allreduce ---

# --- begin:nccl_ring_exchange ---
# Ring exchange: each GPU sends to the next and receives from the previous
NCCL.group() do
    for (i, dev) in enumerate(devs)
        CUDA.device!(dev)
        nranks = length(devs)
        dest = mod(i, nranks) + 1
        src  = mod(i - 2, nranks) + 1
        NCCL.Send(sendbufs[i], comms[i]; dest=dest - 1)
        NCCL.Recv!(recvbufs[i], comms[i]; source=src - 1)
    end
end
# --- end:nccl_ring_exchange ---
