# Reference measurement data

Every file here is an output of a script in [`../scripts/`](../scripts), recorded
on the book's reference GPU. They are committed so you can compare your own
benchmark run against the numbers plotted in the book without first having to
reproduce them.

Nothing in this directory is hand-tuned or modelled: each file is written
verbatim by the script named below. Regenerate any of them by running that
script on your own hardware — your numbers will differ, which is the point.

| File | Written by | What it records |
|:---|:---|:---|
| `bench-environment.txt` | `bench_chapter_figures.jl` | GPU, Julia, CUDA runtime/driver versions and peak bandwidth for the four files below |
| `coalescing-bandwidth.dat` | `bench_chapter_figures.jl` | Effective read bandwidth against access stride (Chapter 6, coalescing) |
| `blocksize-sensitivity.dat` | `bench_chapter_figures.jl` | Achieved bandwidth against block size for three kernels (Chapter 7, launch configuration) |
| `matmul-optimization.dat` | `bench_chapter_figures.jl` | GFLOP/s for four matmul variants, from naive uncoalesced to tiled and coarsened (Chapter 7) |
| `reduction-levels.dat` | `bench_chapter_figures.jl` | Effective bandwidth for four reduction strategies, from naive atomics to warp shuffle (Chapter 7) |
| `monte-carlo-convergence.dat` | `plot_monte_carlo_convergence.jl` | Absolute error and its standard error against sample count, with CPU and GPU timings |
| `nn-batch-scaling.dat` | `bench_nn_batch_scaling.jl` | Per-epoch time and training throughput, CPU vs GPU, across batch sizes (Chapter 12) |
| `nn-batch-environment.txt` | `bench_nn_batch_scaling.jl` | Hardware, BLAS and thread settings for the run above, plus the measured crossover batch size |

Each `.dat` carries its own column header and a comment block describing what
was inside the clock; read those before interpreting the numbers.

## Reference hardware

The committed numbers were measured on an NVIDIA GeForce RTX 4070 SUPER (Ada,
compute capability 8.9) with Julia 1.12.6 and CUDA 12.9. The environment stamps
above carry the exact versions. Figures in the book that were measured on other
hardware say so in their captions.

## License

These measurements are released under the
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/)
(CC BY 4.0) — separate from the MIT License covering the repository's code.
Reuse them freely, including commercially, with credit:

> Arnis Lektauers, *CUDA Programming with Julia* companion measurements, 2026.
> Licensed under CC BY 4.0.

## Third-party data, not included here

Two figures in the book are plotted from datasets this repository does not
redistribute. Both are CC BY 4.0 at their source; fetch them there if you want
to reproduce those figures.

| Dataset | Source | Used for |
|:---|:---|:---|
| Data on Machine Learning Hardware | Epoch AI — <https://epoch.ai/data/machine-learning-hardware> | ML accelerator peak-performance trend by precision. Save the CSV export as `ml_hardware.csv` in this directory, then run `../scripts/prepare_ml_hardware_precision.py`, which writes the per-precision series it needs. The dataset is revised continuously; the book plots the snapshot retrieved 2026-02-09, so a fresh download will not match it exactly. |
| 50 Years of Microprocessor Trend Data | Karl Rupp — <https://github.com/karlrupp/microprocessor-trend-data> | Long-run CPU transistor count, clock frequency, power and single-thread performance. Consumed directly by the book's figure specs, which are not part of this repository. |
