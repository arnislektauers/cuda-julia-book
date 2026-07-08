# CUDA-aware MPI communication

using MPI, CUDA

# --- begin:mpi_send_recv ---
using MPI, CUDA

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# Assign one GPU per MPI rank
CUDA.device!(rank % length(CUDA.devices()))

if rank == 0
    send_buf = CUDA.rand(Float32, 10_000)
    MPI.Send(send_buf, comm; dest=1, tag=0)
elseif rank == 1
    recv_buf = CUDA.zeros(Float32, 10_000)
    MPI.Recv!(recv_buf, comm; source=0, tag=0)
end
# --- end:mpi_send_recv ---

# --- begin:mpi_collectives ---
# AllReduce with GPU buffers
sendbuf = CUDA.ones(Float32, 10_000) .* Float32(rank + 1)
recvbuf = CUDA.zeros(Float32, 10_000)
MPI.Allreduce!(sendbuf, recvbuf, +, comm)

# Broadcast from rank 0
data = rank == 0 ? CUDA.rand(Float32, 10_000) : CUDA.zeros(Float32, 10_000)
MPI.Bcast!(data, comm; root=0)
# --- end:mpi_collectives ---
