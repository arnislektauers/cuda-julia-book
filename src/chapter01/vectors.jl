# --- begin:vec_add ---
function vec_add!(c, a, b)
    @assert length(a) == length(b) == length(c)
    @inbounds for i in eachindex(a)
        c[i] = a[i] + b[i]
    end
end
# --- end:vec_add ---