# ──── Populations ─────────────────────────────────────────────────────────────

"""
    populations(x::AbstractVector)

Compute level populations |ψ_j|² from iso-vec ket (Re(ψ), Im(ψ)).
"""
function populations(x::AbstractVector)
    n = length(x) ÷ 2
    x_re = @view x[1:n]
    x_im = @view x[n+1:2n]
    return x_re .^ 2 .+ x_im .^ 2
end

# ──── Full state (identity measurement) ──────────────────────────────────────

"""
    full_state(x::AbstractVector)

Identity measurement function — returns the iso-vec state as-is.
"""
full_state(x::AbstractVector) = Vector(x)

# ──── Density matrix measurement ─────────────────────────────────────────────

"""
    density_matrix_measurement(x::AbstractVector)

Phase-free measurement: g(x) = density_to_iso_vec(|ψ⟩⟨ψ|) from iso-vec ket.
Removes global phase without requiring density matrix dynamics.
"""
function density_matrix_measurement(x::AbstractVector)
    ψ = iso_to_ket(x)
    ρ = ψ * ψ'
    return density_to_iso_vec(ρ)
end

# ──── Density-native measurement functions ────────────────────────────────────
# These operate on compact density iso-vec (n² elements), used when the NLP
# state is a density matrix (DensityTrajectory with state name :ρ⃗̃).

"""
    populations_density(ρ̃::AbstractVector)

Compute level populations diag(ρ) from compact density iso-vec.
For density matrix dynamics where the NLP state is ρ̃ (compact iso, n² elements).

See also [`populations`](@ref) for the ket iso-vec version.
"""
function populations_density(ρ̃::AbstractVector)
    ρ = compact_iso_to_density(ρ̃)
    return real.(diag(ρ))
end

# ──── Observable expectations ────────────────────────────────────────────────

"""
    observable_expectation(x_iso, O_iso)

Compute ⟨O⟩ = x̃ᵀ Õ x̃ where Õ is the iso-form of O.
Returns a length-1 vector for consistent interface.
"""
function observable_expectation(x_iso::AbstractVector, O_iso::AbstractMatrix)
    return [dot(x_iso, O_iso * x_iso)]
end

"""
    observable_expectations(x_iso, Os_iso)

Compute expectations for multiple observables.
"""
function observable_expectations(
    x_iso::AbstractVector,
    Os_iso::Vector{<:AbstractMatrix},
)
    return [dot(x_iso, O * x_iso) for O in Os_iso]
end

"""
    expect(O::AbstractMatrix)

Create a closure g(x) → [⟨O⟩] for use in MeasurementModel.
The operator O should be in iso-vec form.
"""
expect(O_iso::AbstractMatrix) = x -> observable_expectation(x, O_iso)

"""
    expect(Os::Vector{<:AbstractMatrix})

Create a closure g(x) → [⟨O₁⟩, ⟨O₂⟩, …] for use in MeasurementModel.
The operators should be in iso-vec form.
"""
expect(Os_iso::Vector{<:AbstractMatrix}) = x -> observable_expectations(x, Os_iso)

# ──── Tests ──────────────────────────────────────────────────────────────────
