# ============================================================================ #
#     Propagated (prior-predictive) covariance of measurement rows
# ============================================================================ #
#
# When the device model carries parameter uncertainty Σ_θ (a calibration
# posterior), the measurement rows a model predicts are themselves uncertain.
# Linearizing the rows in the parameters, r(θ) ≈ r(θ̂) + S (θ − θ̂) with
# S = ∂r/∂θ, the prior-predictive covariance of the rows is
#
#     Σ_r = S Σ_θ Sᵀ + Σ_meas                                     (1)
#
# — the delta-method covariance, equivalently the Kalman innovation covariance
# H P Hᵀ + R. This is the object that calibrates model-mismatch alarms and
# acceptance margins: whitening an innovation by Σ_r (not the shot-only Σ_meas)
# gives a unit-covariance statistic, so a χ² alarm fires at its nominal rate and
# an acceptance margin does not false-accept model-uncertainty fluctuations.
#
# The advantage is real only where the row sensitivity S is large — e.g. off the
# fidelity optimum, where first-order sensitivity does not vanish.

using LinearAlgebra

"""
    propagated_covariance(S, Σθ, Σmeas) -> Symmetric

The prior-predictive row covariance `Σ_r = S Σθ Sᵀ + Σ_meas` (Eq. 1). `S` is the
row-sensitivity matrix `∂r/∂θ` (`n_rows × n_params`), `Σθ` the parameter
covariance (`n_params × n_params`), and `Σmeas` the measurement (e.g. shot) noise
covariance (`n_rows × n_rows`, or a `Diagonal`). Returns a `Symmetric` matrix.
"""
function propagated_covariance(S::AbstractMatrix, Σθ::AbstractMatrix, Σmeas::AbstractMatrix)
    size(S, 2) == size(Σθ, 1) == size(Σθ, 2) ||
        error("propagated_covariance: S is $(size(S)), Σθ is $(size(Σθ)) — inner dims must match")
    size(S, 1) == size(Σmeas, 1) == size(Σmeas, 2) ||
        error("propagated_covariance: S has $(size(S,1)) rows, Σmeas is $(size(Σmeas))")
    return Symmetric(S * Σθ * S' + Σmeas)   # Σmeas may be Diagonal (AbstractMatrix)
end

"""
    whiten_block(Σ; W_task=nothing) -> M::Matrix

Full (block) whitening transform for a row covariance `Σ`: returns `M` such that
`M * ν` has identity covariance when `Cov(ν) = Σ`, i.e. `M = W_task · Σ^{-1/2}`
via the Cholesky factor (`Σ = L Lᵀ ⇒ Σ^{-1/2}·ᵀ = L⁻¹`, so `M = L⁻¹` and
`Cov(Mν) = I`). This is the block-covariance path named as future work in
`whiten` — it keeps the off-diagonal correlations that `S Σθ Sᵀ` induces, which
the diagonal fast path drops.

`W_task` (default all-ones) scales rows by task importance after statistical
whitening, composing multiplicatively (as in `whiten`).

Use with [`propagated_covariance`](@ref): `M = whiten_block(Σ_r)`, then a
model-mismatch alarm is `‖M ν‖²  ~  χ²_{n_rows}` under the null.
"""
function whiten_block(Σ::AbstractMatrix; W_task::Union{Nothing,AbstractVector} = nothing)
    n = size(Σ, 1)
    size(Σ, 2) == n || error("whiten_block: Σ must be square, got $(size(Σ))")
    L = cholesky(Symmetric(Matrix(Σ))).L
    Minv = L \ Matrix{Float64}(I, n, n)        # L⁻¹  (lower-triangular solve)
    if W_task !== nothing
        length(W_task) == n || error("whiten_block: W_task has $(length(W_task)) entries for $n rows")
        Minv = Diagonal(collect(Float64, W_task)) * Minv
    end
    return Minv
end

"""
    mahalanobis(ν, Σ) -> Float64

The whitened squared norm `νᵀ Σ⁻¹ ν` (a model-mismatch χ² statistic). Uses a
Cholesky solve for numerical stability. Under `ν ~ N(0, Σ)` this is `χ²`
distributed with `length(ν)` degrees of freedom — the calibrated falsifier.
"""
function mahalanobis(ν::AbstractVector, Σ::AbstractMatrix)
    C = cholesky(Symmetric(Matrix(Σ)))
    return dot(ν, C \ collect(Float64, ν))
end
