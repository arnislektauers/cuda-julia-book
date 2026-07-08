# install the package
# --- begin:cuda_setup ---
using Pkg
Pkg.add("CUDA")
# --- end:cuda_setup ---

# smoke test (this will download the CUDA toolkit)
# --- begin:using_cuda ---
using CUDA
CUDA.versioninfo()
# --- end:using_cuda ---