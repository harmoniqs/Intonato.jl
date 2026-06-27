"""
    partial_trace_B(ρ_iso::AbstractVector, dims::Tuple{Int,Int})

Trace out subsystem B from density matrix iso-vec.

`dims = (d_A, d_B)` where d_A × d_B = total Hilbert space dimension.
Returns the reduced density matrix of subsystem A in iso-vec form.
"""
function partial_trace_B(ρ_iso::AbstractVector, dims::Tuple{Int,Int})
    ρ = iso_vec_to_density(ρ_iso)
    d_A, d_B = dims

    # kron ordering: row = (i_A-1)*d_B + i_B, so B is the fast index.
    # Reshape ρ as ρ_tensor[i_B, i_A, j_B, j_A].
    ρ_tensor = reshape(ρ, d_B, d_A, d_B, d_A)

    # Trace over B: ρ_A[i_A, j_A] = Σ_b ρ_tensor[b, i_A, b, j_A]
    ρ_A = zeros(eltype(ρ), d_A, d_A)
    for b = 1:d_B
        ρ_A .+= ρ_tensor[b, :, b, :]
    end

    return density_to_iso_vec(ρ_A)
end

# ──── Tests ──────────────────────────────────────────────────────────────────
