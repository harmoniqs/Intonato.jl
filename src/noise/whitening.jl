# ============================================================================ #
#            whiten — GLS weights + variances from the measurement model
# ============================================================================ #

# Per-element variances contributed by one measurement's noise model, evaluated
# at the measured values `y` (plug-in GLS). Diagonal fast path: off-diagonal
# covariance is dropped (see `whiten` docstring).
function _element_variances(m::ShotNoiseMeasurement, y::AbstractVector, var_floor::Real)
    Σ = m.covariance_fn(y, m.n_shots)
    floor = var_floor / m.n_shots
    return max.(LinearAlgebra.diag(Σ), floor)
end

_element_variances(m::KnownCovarianceMeasurement, ::AbstractVector, ::Real) =
    LinearAlgebra.diag(m.Σ)

_element_variances(::DeterministicMeasurement, y::AbstractVector, ::Real) =
    zeros(length(y))

"""
    whiten(model::MeasurementModel, y_exp::Vector{Measurement};
           W_task=nothing, var_floor=0.05) -> (w::Vector{Float64}, σ2::Vector{Float64})

Assemble the GLS whitening weights `W = W_task · Σ^{-1/2}` and the per-element
noise variances `Σ` from the measurement model's noise types, evaluated at the
measured values `y_exp` (plug-in GLS — re-evaluate each iteration). Both are
returned flattened in model element order (measurement-major).

Per noise type:
- `ShotNoiseMeasurement` → `σ² = diag(covariance_fn(y, n))`, floored at
  `var_floor/n` so weights stay finite as `|y| → 1` (where e.g. the Wigner
  variance `(1−y²)/n` vanishes exactly where parity is most saturated);
  `w = W_task/σ`. The default `var_floor = 0.05` engages only for
  `|y| ≳ 0.975` on `(1−y²)`-type variances — raise it to cap weights earlier.
- `KnownCovarianceMeasurement` → `σ² = diag(Σ)`; `w = W_task/σ`. **The
  off-diagonals of `Σ` are dropped** — this is the diagonal fast path; a
  block-covariance path is future work per the spec.
- `DeterministicMeasurement` → `σ² = 0` and `w = W_task` (identity scaling;
  these elements contribute no noise floor and no σ_J).

`W_task` (default all-ones) is the task-importance weight vector — explicit
and composable with statistical whitening (e.g. the σ_z×40 leakage-defense
weight is task importance, not noise statistics). Statistical whitening does
NOT subsume it.
"""
function whiten(
    model::MeasurementModel,
    y_exp::Vector{Measurement};
    W_task::Union{Nothing,AbstractVector} = nothing,
    var_floor::Float64 = 0.05,
)
    length(model.measurements) == length(y_exp) || error(
        "whiten: model has $(length(model.measurements)) measurements, " *
        "y_exp has $(length(y_exp))",
    )
    σ2 = Float64[]
    for (m, y) in zip(model.measurements, y_exp)
        append!(σ2, _element_variances(m, y.data, var_floor))
    end
    n = length(σ2)
    w_task = isnothing(W_task) ? ones(n) : collect(Float64, W_task)
    length(w_task) == n ||
        error("whiten: W_task has $(length(w_task)) entries for $n elements")
    w = [σ2[i] > 0 ? w_task[i] / sqrt(σ2[i]) : w_task[i] for i = 1:n]
    return w, σ2
end
