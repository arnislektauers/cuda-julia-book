#!/usr/bin/env bash
# Runs book source programs on a GPU host, one per fresh Julia process,
# capturing PASS / FAIL / TIMEOUT. Usage: run.sh <T1|T2> [timeout_seconds]
set -uo pipefail
cd "$(dirname "$0")"
TIER="${1:-T1}"; TMO="${2:-360}"
JULIA="$HOME/.juliaup/bin/julia"
PROJ="$PWD/env-$TIER"
LOGS="$PWD/logs-$TIER"; mkdir -p "$LOGS"
RES="$PWD/results-$TIER.tsv"; : > "$RES"

echo ">>> Julia: $($JULIA --version)   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ">>> Instantiating $TIER environment at $PROJ ..."
mkdir -p "$PROJ"
if [ "$TIER" = "T1" ]; then
  DEPS='["CUDA","Adapt","BenchmarkTools","KernelAbstractions","StaticArrays","GPUArraysCore","LinearAlgebra","SparseArrays","Statistics","Random","Distributed","Test"]'
else
  DEPS='["CUDA","Adapt","BenchmarkTools","KernelAbstractions","StaticArrays","GPUArraysCore","LinearAlgebra","SparseArrays","Statistics","Random","Distributed","Test","Flux","Lux","Zygote","Enzyme","DiffEqGPU","OrdinaryDiffEq","DataFrames","CSV","MLUtils","OneHotArrays","Optimisers","ChainRulesCore","CairoMakie","Images","TestImages"]'
fi
$JULIA --project="$PROJ" -e "using Pkg; for p in $DEPS; try; Pkg.add(p); catch e; @warn \"add failed\" p e; end; end; Pkg.precompile()" > "$LOGS/_instantiate.log" 2>&1
echo ">>> Instantiate exit: $?  (log: $LOGS/_instantiate.log)"

pass=0; fail=0; to=0; n=0
while IFS=$'\t' read -r tier exec f; do
  [ "$tier" = "$TIER" ] || continue
  n=$((n+1))
  name=$(echo "$f" | tr '/' '_' | sed 's/\.jl$//')
  log="$LOGS/$name.log"
  timeout "$TMO" "$JULIA" --project="$PROJ" "$f" > "$log" 2>&1
  rc=$?
  if   [ $rc -eq 0 ];   then st=PASS; pass=$((pass+1))
  elif [ $rc -eq 124 ]; then st=TIMEOUT; to=$((to+1))
  else st=FAIL; fail=$((fail+1)); fi
  err=$(grep -m1 -iE "error|exception|not defined|no method|failed" "$log" | head -c 200 | tr '\t' ' ')
  printf '%s\t%s\t%s\t%s\t%s\n' "$st" "$exec" "$f" "$rc" "$err" >> "$RES"
  printf '[%2d] %-8s %-6s %s\n' "$n" "$st" "$exec" "$f"
done < runlist.txt

echo "==== $TIER SUMMARY: PASS=$pass FAIL=$fail TIMEOUT=$to (of $n) ===="
echo "Results: $RES   Logs: $LOGS"
