# ============================================================================ #
#     Binomial Fisher / IRLS measurement weighting (exact-likelihood GLS)
# ============================================================================ #
#
# For shot-noise readout the underlying count statistic is binomial, and its
# Fisher information is known in closed form — so the exact-likelihood
# weighting of a least-squares cost needs no Gaussian plug-in approximation.
# This is the textbook iteratively-reweighted-least-squares (IRLS) weighting
# for binomial GLMs (McCullagh & Nelder 1989): weight the residual of element
# i by the Fisher information evaluated at the current fit, then refit, then
# re-evaluate the weighting at the new fit. The Gaussian predecessor on this
# surface is noise/whitening.jl (plug-in GLS); this module is its
# exact-likelihood counterpart for binomial/count shot statistics.

# Per-element Fisher weights from one measurement's shot statistics, evaluated
# at the fit `y` (the IRLS evaluation point). For binomial/count statistics
# the covariance diagonal has the exact form c·p(1-p)/n_shots, whose inverse
# is the Fisher information in the measured coordinate — the p↔y Jacobian is
# absorbed by the covariance function's own convention. Mirrors whitening.jl's
# `_element_variances` dispatch, inverted.
function _fisher_elements(m::ShotNoiseMeasurement, y::AbstractVector, var_floor::Real)
    σ2 = LinearAlgebra.diag(m.covariance_fn(y, m.n_shots))
    floored = max.(σ2, var_floor / m.n_shots)
    return 1 ./ sqrt.(floored)
end

# A user-known covariance is Fisher-exact only when it is binomial-form; the
# diagonal plug-in value is returned either way (documented, not enforced).
# A zero diagonal variance yields Inf — the exact zero-variance limit.
_fisher_elements(m::KnownCovarianceMeasurement, ::AbstractVector, ::Real) =
    1 ./ sqrt.(LinearAlgebra.diag(m.Σ))

# No shot statistics: no Fisher weighting — identity, like whiten's.
_fisher_elements(::DeterministicMeasurement, y::AbstractVector, ::Real) = ones(length(y))

"""
    binomial_fisher_information(p, n_shots) -> Float64

Exact Fisher information of a binomial count statistic — `n_shots` shots with
success probability `p` per Bernoulli outcome:

    I(p) = n_shots / (p * (1 - p))

The asymptotic variance of the MLE `p̂ = k/n` is its inverse, `p(1-p)/n_shots`.
For count data (multinomial outcomes) the same closed form applies per outcome
through the marginal binomial statistic — the repo's diagonal fast path.

`p ∉ [0, 1]` or `n_shots ≤ 0` is an `ArgumentError`. The saturated limits
`p → 0, 1` return `Inf` — the exact limit; floor the weight at the surface
level via [`binomial_fisher_weights`](@ref).

**A1 boundary:** generic exact-likelihood statistics for shot-noise readout —
public surface; error-channel attribution machinery is internal.
"""
function binomial_fisher_information(p::Real, n_shots::Real)
    0 <= p <= 1 ||
        throw(ArgumentError("binomial_fisher_information: p must be in [0, 1], got $p"))
    n_shots > 0 || throw(
        ArgumentError(
            "binomial_fisher_information: n_shots must be positive, got $n_shots",
        ),
    )
    return n_shots / (p * (1 - p))
end

"""
    binomial_fisher_weights(model::MeasurementModel, y_fit::Vector{Measurement};
                            var_floor::Float64 = 0.05) -> Vector{Float64}

Closed-form Fisher weights for binomial/count shot-noise data, assembled from
the measurement model's shot statistics **alone** — `n_shots` and the
per-element success probability carried by each measurement's covariance
function, evaluated at the model-side fit `y_fit`. No device knowledge, no
physical channel names: a lab that brings only their own measurement model
(and their own binomial-form `covariance_fn`) gets the exact-likelihood
weighting.

Per element, flattened in model element order (measurement-major — the same
convention as [`whiten`](@ref)):

- `ShotNoiseMeasurement` → `w = 1/√σ²` with `σ² = diag(covariance_fn(y_fit,
  n_shots))` floored at `var_floor/n_shots`. For binomial/count statistics the
  covariance diagonal has the exact form `c·p(1-p)/n_shots`, so `w² = I(y_fit)`
  is the Fisher information of the count statistic **in the measured
  coordinate** — the p↔y Jacobian is absorbed by the covariance function's
  convention (`pauli_covariance`/`wigner_covariance`: Rademacher assignment,
  `y = 2p-1`; `population_covariance`: proportions, `y = p`). `var_floor`
  engages only for saturated fits (`|y| ≳ 0.975` at the default); pass
  `var_floor = 0` for the raw closed form (`Inf` at saturation).
- `DeterministicMeasurement` → `w = 1` (no shot statistics; identity).
- `KnownCovarianceMeasurement` → `w = 1/√diag(Σ)` — the Fisher weight when Σ
  is binomial-form; the plug-in value otherwise (documented, not enforced).

**IRLS discipline.** This is the weighting half of the standard
iteratively-reweighted-least-squares update for binomial GLMs (McCullagh &
Nelder 1989): weight the residual by the Fisher information evaluated at the
current *fit* — not at the measured values, which is `whiten`'s plug-in
evaluation point — then refit, then re-evaluate the weighting at the new fit.
The re-evaluation loop belongs to the caller (the lab's tuning loop, or the
chassis through `W_task` per iteration).

**Chassis composition.** Pass the weights as `PulseTuningProblem`'s `W_task`:
on a deterministic-stub model the chassis cost is then exactly the
Fisher-weighted SSR `Σ w²r²`. On a model whose elements already carry shot
statistics, the chassis multiplies `W_task` with its own plug-in whitening
(`W = W_task·Σ^{-1/2}`) — the two statistical weightings then both act;
compose one, not both, when you want the exact-likelihood weighting alone.

**A1 boundary:** generic exact-likelihood statistics for shot-noise readout —
public surface; error-channel attribution machinery is internal.
"""
function binomial_fisher_weights(
    model::MeasurementModel,
    y_fit::Vector{Measurement};
    var_floor::Float64 = 0.05,
)
    length(model.measurements) == length(y_fit) || error(
        "binomial_fisher_weights: model has $(length(model.measurements)) measurements, " *
        "y_fit has $(length(y_fit))",
    )
    w = Float64[]
    for (m, y) in zip(model.measurements, y_fit)
        append!(w, _fisher_elements(m, y.data, var_floor))
    end
    return w
end
