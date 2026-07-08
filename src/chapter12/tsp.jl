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
