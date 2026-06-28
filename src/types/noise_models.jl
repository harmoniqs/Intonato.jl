# ============================================================================ #
#                      AbstractMeasurement hierarchy
# ============================================================================ #

"""
    AbstractMeasurement

A measurement that is both callable (computes measurement values from a state)
and carries noise information for MAP weighting.

All subtypes must be callable: `m(x)` returns `Vector{Float64}`.
"""
abstract type AbstractMeasurement end

"""
    DeterministicMeasurement(g)

Wraps a callable measurement function with no noise model.
Equivalent to the bare `Function` used before Phase 4.
"""
struct DeterministicMeasurement <: AbstractMeasurement
    g::Function
end
(m::DeterministicMeasurement)(x) = m.g(x)

"""
    ShotNoiseMeasurement(g, n_shots, covariance_fn)

A measurement with shot-noise statistics. The covariance is computed analytically
from the predicted measurement output:

    Σ = covariance_fn(y_predicted, n_shots)

# Fields
- `g::Function`: measurement function `x → Vector{Float64}`
- `n_shots::Int`: number of measurement shots
- `covariance_fn::Function`: `(y_predicted, n_shots) → AbstractMatrix{Float64}`
"""
struct ShotNoiseMeasurement <: AbstractMeasurement
    g::Function
    n_shots::Int
    covariance_fn::Function
end
(m::ShotNoiseMeasurement)(x) = m.g(x)

"""
    KnownCovarianceMeasurement(g, Σ)

A measurement with a fixed, user-provided covariance matrix.
"""
struct KnownCovarianceMeasurement <: AbstractMeasurement
    g::Function
    Σ::Matrix{Float64}
end
(m::KnownCovarianceMeasurement)(x) = m.g(x)

# ============================================================================ #
#                      Covariance formulas
# ============================================================================ #

"""
    pauli_covariance(y, n_shots) → Diagonal

Covariance for Pauli expectation measurements `⟨σ_j⟩`.
`Var[⟨σ_j⟩] = (1 - ⟨σ_j⟩²) / n_shots` (since σ_j² = I for Pauli matrices).
"""
pauli_covariance(y, n) = Diagonal((1 .- y .^ 2) ./ n)

"""
    population_covariance(y, n_shots) → Matrix

Multinomial covariance for population measurements.
`Σ = (diag(p) - p*p') / n_shots` where `p` is the predicted probability vector.
"""
population_covariance(y, n) = (diagm(y) - y * y') / n

"""
    wigner_covariance(y, n_shots) → Diagonal

Covariance for Wigner / displaced parity measurements.
`Var[W(α)] = (1 - W(α)²) / n_shots` (Rademacher mean).
Same formula as Pauli, different physics.
"""
wigner_covariance(y, n) = Diagonal((1 .- y .^ 2) ./ n)

# ============================================================================ #
#                      Measurement presets
# ============================================================================ #

"""
    pauli(operators; n_shots=nothing) → AbstractMeasurement

Pauli expectation measurement preset. Returns `DeterministicMeasurement` when
`n_shots` is omitted, `ShotNoiseMeasurement` with `pauli_covariance` otherwise.
"""
function pauli(ops; n_shots::Union{Nothing,Int} = nothing)
    g = expect(ops)
    isnothing(n_shots) ? DeterministicMeasurement(g) :
    ShotNoiseMeasurement(g, n_shots, pauli_covariance)
end

"""
    pop(; n_shots=nothing) → AbstractMeasurement

Population measurement preset. Returns `DeterministicMeasurement` when
`n_shots` is omitted, `ShotNoiseMeasurement` with `population_covariance` otherwise.
"""
function pop(; n_shots::Union{Nothing,Int} = nothing)
    isnothing(n_shots) ? DeterministicMeasurement(populations) :
    ShotNoiseMeasurement(populations, n_shots, population_covariance)
end

# ============================================================================ #
#                      Tests
# ============================================================================ #
