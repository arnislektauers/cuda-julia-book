#!/usr/bin/env bash
# Runs book source programs on a GPU host, one per fresh Julia process,
# capturing PASS / FAIL / TIMEOUT.
#
#   Usage: run.sh <T1|T2|T2-display> [timeout_seconds]
#
# T2-display holds the listings that open a window. It reuses env-T2 rather
# than building a third environment -- the dependency set is identical and the
# depot is several GB -- but runs only where a real display is reachable, and
# refuses to run at all under software GL (see the preflight below).
set -uo pipefail
cd "$(dirname "$0")"
TIER="${1:-T1}"; TMO="${2:-360}"
JULIA="$HOME/.juliaup/bin/julia"

# env-T2-display would be a pointless copy of env-T2.
ENVTIER="${TIER%-display}"

# A system CUDA install on LD_LIBRARY_PATH shadows CUDA.jl's own artifacts and
# can mix incompatible library versions (CUDA.jl warns about this at startup).
# On a host with /usr/local/cuda in the path that surfaces as spurious failures
# such as "libcublas.so: undefined symbol: cublasLtSetEnvironmentMode", so run
# every listing against the artifacts CUDA.jl selected for itself.
export LD_LIBRARY_PATH=""
PROJ="$PWD/env-$ENVTIER"
LOGS="$PWD/logs-$TIER"; mkdir -p "$LOGS"
RES="$PWD/results-$TIER.tsv"; : > "$RES"

# ------------------------------------------------------------------ display
# Only for the display tier. Two distinct failures to separate: no display at
# all (skip, this host cannot check these listings), and a display backed by
# software rasterization (also skip -- llvmpipe renders the figure perfectly
# well, so the run would PASS while proving nothing about the OpenGL path,
# which is the entire reason these listings exist separately from CairoMakie).
if [ "$TIER" != "$ENVTIER" ]; then
  export DISPLAY="${DISPLAY:-:0}"
  echo ">>> display tier: DISPLAY=$DISPLAY"
  if ! xdpyinfo >/dev/null 2>&1; then
    echo ">>> SKIP: no X display reachable at $DISPLAY. Set DISPLAY to the"
    echo "    machine's own session (over SSH this is usually :0) and retry."
    exit 0
  fi
  GLREND="$(glxinfo -B 2>/dev/null | grep -i 'OpenGL renderer' | cut -d: -f2-)"
  if [ -z "$GLREND" ]; then
    echo ">>> WARNING: no glxinfo, cannot tell hardware GL from software."
  elif echo "$GLREND" | grep -qiE 'llvmpipe|softpipe|swrast'; then
    echo ">>> SKIP: software GL ($GLREND ). A PASS here would not exercise"
    echo "    the GPU. xvfb-run is software-only and cannot substitute."
    exit 0
  else
    echo ">>> OpenGL renderer:$GLREND"
  fi
fi

echo ">>> Julia: $($JULIA --version)   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ">>> Instantiating $ENVTIER environment at $PROJ ..."
mkdir -p "$PROJ"
if [ "$ENVTIER" = "T1" ]; then
  DEPS='["CUDA","Adapt","BenchmarkTools","KernelAbstractions","StaticArrays","GPUArraysCore","LinearAlgebra","SparseArrays","Statistics","Random","Distributed","Test"]'
else
  DEPS='["CUDA","Adapt","BenchmarkTools","KernelAbstractions","StaticArrays","GPUArraysCore","LinearAlgebra","SparseArrays","Statistics","Random","Distributed","Test","Flux","Lux","Zygote","Enzyme","DiffEqGPU","OrdinaryDiffEq","DataFrames","CSV","MLUtils","OneHotArrays","Optimisers","ChainRulesCore","cuDNN","cuTENSOR","CairoMakie","GLMakie","Images","TestImages"]'
fi
$JULIA --project="$PROJ" -e "using Pkg; for p in $DEPS; try; Pkg.add(p); catch e; @warn \"add failed\" p e; end; end" > "$LOGS/_instantiate.log" 2>&1
echo ">>> Add exit: $?  (log: $LOGS/_instantiate.log)"

# Pin the CUDA runtime rather than letting CUDA.jl pick the newest one the
# driver admits. The book's tested configuration is CUDA 12, and a host whose
# driver is new enough for CUDA 13 (580+) would otherwise silently resolve
# 13.x and stop reproducing what the text documents. Setting the preference
# invalidates precompilation, so it has to happen before Pkg.precompile() and
# in its own process. Override for a deliberate experiment with
# CUDA_RUNTIME_VERSION=13.3 ./run.sh T1
CUDA_RUNTIME_VERSION="${CUDA_RUNTIME_VERSION:-12.9}"
echo ">>> pinning CUDA runtime to $CUDA_RUNTIME_VERSION"
$JULIA --project="$PROJ" -e "using CUDA; CUDA.set_runtime_version!(v\"$CUDA_RUNTIME_VERSION\")" >> "$LOGS/_instantiate.log" 2>&1
echo ">>> Pin exit: $?"

$JULIA --project="$PROJ" -e "using Pkg; Pkg.precompile()" >> "$LOGS/_instantiate.log" 2>&1
echo ">>> Precompile exit: $?  (log: $LOGS/_instantiate.log)"

# A program that never reaches the GPU still exits 0, so an exit code alone
# cannot tell "ran on the GPU" from "silently fell back to the CPU". Flux and
# Lux reach the device through MLDataDevices, which only activates the CUDA
# backend when its trigger package (cuDNN) is loaded; without it these
# frameworks train on the host behind this one warning. Treat it as a failure.
FALLBACK_RE='No functional GPU backend found'

pass=0; fail=0; to=0; cpu=0; n=0
while IFS=$'\t' read -r tier exec f; do
  [ "$tier" = "$TIER" ] || continue
  n=$((n+1))
  name=$(echo "$f" | tr '/' '_' | sed 's/\.jl$//')
  log="$LOGS/$name.log"
  timeout "$TMO" "$JULIA" --project="$PROJ" "$f" > "$log" 2>&1
  rc=$?
  if   [ $rc -eq 0 ] && grep -q "$FALLBACK_RE" "$log"; then
    st=CPU-ONLY; cpu=$((cpu+1))
  elif [ $rc -eq 0 ];   then st=PASS; pass=$((pass+1))
  elif [ $rc -eq 124 ]; then st=TIMEOUT; to=$((to+1))
  else st=FAIL; fail=$((fail+1)); fi
  if [ "$st" = CPU-ONLY ]; then
    err="GPU backend inactive (missing trigger package?) -- ran on CPU"
  else
    err=$(grep -m1 -iE "error|exception|not defined|no method|failed" "$log" | head -c 200 | tr '\t' ' ')
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$st" "$exec" "$f" "$rc" "$err" >> "$RES"
  printf '[%2d] %-8s %-6s %s\n' "$n" "$st" "$exec" "$f"
done < runlist.txt

echo "==== $TIER SUMMARY: PASS=$pass FAIL=$fail TIMEOUT=$to CPU-ONLY=$cpu (of $n) ===="
echo "Results: $RES   Logs: $LOGS"
