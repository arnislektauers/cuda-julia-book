# Traveling Salesman Problem -- GPU tour length computation

# --- begin:tsp_tour_length ---
using CUDA

function compute_tour_length!(tour_lengths, tours, distances, n_cities, n_tours)
    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if tid <= n_tours
        total = 0.0f0
        offset = (tid - 1) * n_cities
        @inbounds for i in 1:n_cities-1
            c1 = tours[offset + i]
            c2 = tours[offset + i + 1]
            total += distances[(c1 - 1) * n_cities + c2]
        end
        # Return to start
        c1 = tours[offset + n_cities]
        c2 = tours[offset + 1]
        total += distances[(c1 - 1) * n_cities + c2]
        tour_lengths[tid] = total
    end
    return nothing
end
# --- end:tsp_tour_length ---

# ---------------------------------------------------------------------------
# Driver, outside the tagged region so the book is unaffected. Two tours over
# four cities on a unit square: the perimeter tour 1-2-3-4 measures 4.0, and
# the diagonal tour 1-3-2-4 measures 2 + 2*sqrt(2). Checking the value matters
# more than completing here -- the kernel indexes a flattened distance matrix
# by hand, which is exactly the kind of arithmetic that runs without erroring
# and returns nonsense.
let n = 4, ntours = 2
    pts = Float32[0 0; 1 0; 1 1; 0 1]
    d = Float32[sqrt((pts[i,1]-pts[j,1])^2 + (pts[i,2]-pts[j,2])^2)
                for i in 1:n for j in 1:n]          # row-major flatten
    tours = Int32[1, 2, 3, 4,  1, 3, 2, 4]
    lengths = CUDA.zeros(Float32, ntours)
    @cuda threads=32 blocks=1 compute_tour_length!(
        lengths, CuArray(tours), CuArray(d), n, ntours)
    CUDA.synchronize()
    got = Array(lengths)
    @assert isapprox(got[1], 4.0f0; atol = 1f-4) "perimeter tour: $(got[1])"
    @assert isapprox(got[2], 2 + 2*sqrt(2f0); atol = 1f-4) "diagonal tour: $(got[2])"
    println("compute_tour_length!: both tours CORRECT")
end
