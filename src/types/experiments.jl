abstract type AbstractExperiment end

# ──── SimulatedExperiment ────────────────────────────────────────────────────

"""
    SimulatedExperiment{QT}

Simulated experiment using a "true" (mismatched) quantum system.

The `qtraj_template` carries the true system, initial state, and goal.
`run_experiment` rolls out the given pulse through this system and evaluates
the measurement model at the specified knot indices.

# Noise sampling contract
`rng` (constructor kwarg, default `nothing`) controls measurement-noise
sampling:
- `rng === nothing` → deterministic evaluation, exactly the legacy behavior.
- an `AbstractRNG` → after evaluating each measurement, draw
  `ε ~ N(0, Σ(y))` from the element's noise model and add it (Gaussian
  approximation to shot statistics; variances floored as in [`whiten`](@ref)).
  Deterministic elements get no noise. Seeding the rng gives reproducible
  fixtures.
"""
struct SimulatedExperiment{QT<:AbstractQuantumTrajectory} <: AbstractExperiment
    qtraj_template::QT
    measurement_model::MeasurementModel
    rng::Union{Nothing,AbstractRNG}
end

function SimulatedExperiment(
    qtraj_template::AbstractQuantumTrajectory,
    measurement_model::MeasurementModel;
    rng::Union{Nothing,AbstractRNG} = nothing,
)
    return SimulatedExperiment(qtraj_template, measurement_model, rng)
end

# Value-type copies of shot-noise measurements with a boosted shot count.
# Non-shot measurements pass through unchanged. Used by the `n_shots` override
# (e.g. TopKRemeasure re-measuring finalists at higher precision).
_with_n_shots(m::ShotNoiseMeasurement, n::Int) =
    ShotNoiseMeasurement(m.g, n, m.covariance_fn)
_with_n_shots(m::AbstractMeasurement, ::Int) = m

function _with_n_shots(model::MeasurementModel, n::Int)
    return MeasurementModel(
        model.state_name,
        AbstractMeasurement[_with_n_shots(m, n) for m in model.measurements],
        model.indices,
    )
end

# Add ε ~ N(0, Σ(y)) to each noisy measurement element in-place (in the vector;
# Measurement itself is immutable — replaced). Variances come from each
# measurement's noise model at the evaluated y, floored as in `whiten`.
# Deterministic elements draw nothing, so the rng stream is stable across
# layout-identical runs (the n_shots-scaling contract depends on this).
function _sample_measurement_noise!(
    measurements::Vector{Measurement},
    model::MeasurementModel,
    rng::AbstractRNG,
)
    for (j, m) in enumerate(model.measurements)
        σ2 = _element_variances(m, measurements[j].data, 0.05)
        any(>(0.0), σ2) || continue
        noisy = measurements[j].data .+ sqrt.(σ2) .* randn(rng, length(σ2))
        measurements[j] = Measurement(noisy, measurements[j].index)
    end
    return measurements
end

"""
    run_experiment(exp::SimulatedExperiment, pulse::AbstractPulse;
                   logger = NullExperimentLogger(), n_shots = nothing)

Rollout pulse through the true system and evaluate measurements at each knot index.

The positional contract is unchanged — this returns a `Vector{Measurement}`.
The additive `logger` keyword (default [`NullExperimentLogger`](@ref), a no-op)
builds an [`ExperimentRecord`](@ref) from the run and calls `record!(logger, …)`.

The additive `n_shots` keyword (default `nothing` = passthrough) rebuilds the
model's `ShotNoiseMeasurement`s with the boosted shot count before evaluation —
a value-type copy, never a mutation of `exp`. With a sampling `rng` set on the
experiment, the injected noise then scales as `1/√n_shots`.
"""
function run_experiment(
    exp::SimulatedExperiment,
    pulse::AbstractPulse;
    logger::AbstractExperimentLogger = NullExperimentLogger(),
    n_shots::Union{Nothing,Int} = nothing,
)
    qtraj = rollout(exp.qtraj_template, pulse)
    knot_times = get_knot_times(pulse)
    model =
        isnothing(n_shots) ? exp.measurement_model :
        _with_n_shots(exp.measurement_model, n_shots)

    measurements = Vector{Measurement}(undef, length(model.indices))
    for (j, k) in enumerate(model.indices)
        t_k = knot_times[k]
        x_k = _state_at_time(qtraj, t_k, model.state_name)
        measurements[j] = Measurement(model.measurements[j](x_k), k)
    end

    if !isnothing(exp.rng)
        _sample_measurement_noise!(measurements, model, exp.rng)
    end

    _maybe_record!(logger, exp, pulse, model, measurements; device = "sim")

    return measurements
end

# ──── _state_at_time dispatch ────────────────────────────────────────────────

function _state_at_time(qtraj::KetTrajectory, t::Float64, ::Symbol)
    return ket_to_iso(qtraj(t))
end

function _state_at_time(qtraj::UnitaryTrajectory, t::Float64, ::Symbol)
    return operator_to_iso_vec(qtraj(t))
end

function _state_at_time(qtraj::DensityTrajectory, t::Float64, ::Symbol)
    return density_to_compact_iso(qtraj(t))
end

# ──── HardwareExperiment ─────────────────────────────────────────────────────

"""
    HardwareExperiment

Hardware experiment with a user-provided callable.

The `run` function is responsible for executing the pulse on hardware,
collecting raw measurement data, and constructing `Measurement` objects
with the correct knot indices.
"""
struct HardwareExperiment <: AbstractExperiment
    run::Function   # (pulse::AbstractPulse) → Vector{Measurement}
    # Optional measurement model used only to annotate logged ExperimentRecords.
    # `nothing` when the user's `run` closure owns the measurement structure.
    measurement_model::Union{Nothing,MeasurementModel}
end

HardwareExperiment(run::Function) = HardwareExperiment(run, nothing)

"""
    run_experiment(exp::HardwareExperiment, pulse::AbstractPulse;
                   logger = NullExperimentLogger(), n_shots = nothing)

Execute the user-provided `run` closure on `pulse` and return its
`Vector{Measurement}`. The additive `logger` keyword (default
[`NullExperimentLogger`](@ref), a no-op) builds an [`ExperimentRecord`](@ref)
and calls `record!(logger, …)`; the record's `measurement_model` is `exp`'s
annotation if set, otherwise a trivial placeholder derived from the returned
measurement indices.

The additive `n_shots` keyword forwards to the `run` closure when it accepts
an `n_shots` keyword; otherwise it is ignored with a one-time warning (the
closure owns the hardware shot budget).
"""
function run_experiment(
    exp::HardwareExperiment,
    pulse::AbstractPulse;
    logger::AbstractExperimentLogger = NullExperimentLogger(),
    n_shots::Union{Nothing,Int} = nothing,
)
    measurements = if isnothing(n_shots)
        exp.run(pulse)
    elseif hasmethod(exp.run, Tuple{typeof(pulse)}, (:n_shots,))
        exp.run(pulse; n_shots = n_shots)
    else
        @warn "HardwareExperiment run closure does not accept n_shots; override ignored" maxlog =
            1
        exp.run(pulse)
    end
    if !(logger isa NullExperimentLogger)
        model =
            isnothing(exp.measurement_model) ?
            _placeholder_measurement_model(measurements) : exp.measurement_model
        _maybe_record!(logger, exp, pulse, model, measurements; device = "hardware")
    end
    return measurements
end

# Build a minimal MeasurementModel from a returned measurement vector so a
# HardwareExperiment with no declared model can still produce a well-formed
# ExperimentRecord. Uses DeterministicMeasurement identity stand-ins keyed by
# the measurement knot indices (the data is already in `measurements`).
function _placeholder_measurement_model(measurements::Vector{Measurement})
    indices = [m.index for m in measurements]
    funcs = AbstractMeasurement[DeterministicMeasurement(identity) for _ in measurements]
    return MeasurementModel(:unknown, funcs, indices)
end

# ──── ExperimentRecord construction (data-collection seam) ───────────────────

# Per-measurement shot counts read off the model's measurements. A
# ShotNoiseMeasurement carries `n_shots`; deterministic / known-covariance
# measurements have no shot count (recorded as `missing`).
function _shots_per_measurement(model::MeasurementModel)
    return [m isa ShotNoiseMeasurement ? m.n_shots : missing for m in model.measurements]
end

# Deterministic, dependency-free hash of a pulse's knot data, rendered as a
# hex string. Uses Julia's `hash` (no SHA stdlib dep on the public surface).
function _pulse_hash(pulse::AbstractPulse)
    h = hash(get_knot_times(pulse))
    h = hash(get_knot_values(pulse), h)
    if pulse isa CubicSplinePulse
        h = hash(get_knot_derivatives(pulse), h)
    end
    return string(h; base = 16)
end

# Build an ExperimentRecord and hand it to the logger. No-op fast path for the
# default NullExperimentLogger so the common case allocates nothing.
function _maybe_record!(
    logger::AbstractExperimentLogger,
    ::AbstractExperiment,
    pulse::AbstractPulse,
    model::MeasurementModel,
    measurements::Vector{Measurement};
    device::AbstractString,
)
    logger isa NullExperimentLogger && return nothing
    # `time()` (epoch seconds) keeps the public surface dependency-free (no
    # Dates stdlib dep); a richer ISO-8601 timestamp can be supplied by
    # backends in their own metadata before/after this call.
    metadata = (;
        device = device,
        shots = _shots_per_measurement(model),
        timestamp = string(time()),
        pulse_hash = _pulse_hash(pulse),
    )
    rec = ExperimentRecord(;
        pulse = pulse,
        measurement_model = model,
        measurements = measurements,
        raw = nothing,
        metadata = metadata,
    )
    record!(logger, rec)
    return nothing
end

# ──── Free-phase fidelity helper ─────────────────────────────────────────────

"""
    phase_max_fidelity(ψ_T, ψ_goal; n_grid=128) → Float64

Maximum of `|⟨ψ_goal | exp(iφ n̂) | ψ_T⟩|²` over a 1D grid of `n_grid`
phases φ ∈ [0, 2π), where `n̂ = diag(0, 1, …, d−1)` is the number operator
on the full Hilbert space (dim `d = length(ψ_T)`).

**Why this exists:** Piccolo's QCP objectives for cavity / multi-level
systems learn one (or more) free-phase globals (`:φ_1`, …) that
maximize `|⟨ψ_goal | e^{iφ_1 G_1} ⋯ | ψ_T⟩|²` over physically-irrelevant
frame rotations. The naive `abs2(dot(ψ_goal, ψ_T))` ignores this and
returns a fidelity that's much lower than the optimization actually
achieved (typical cat-state runs: raw overlap ~0.4, free-phase fidelity
~0.9+). This helper recovers a strict lower bound on the free-phase
fidelity using a single number-operator generator on the full space.

**Coverage:**
- Pure cavity / pure qubit / single multi-level system: exact (matches
  the cavity grid-search formula `F = max_φ |∑_n e^{iφn} o_n|²`).
- Pure-Z multi-qubit with one shared phase: exact.
- Tensor-product systems with two independent phase generators (e.g.
  transmon × cavity, where both `e^{iφ₁ n̂_q}` and `e^{iφ₂ n̂_c}` are
  free): catches the dominant phase only; *still a lower bound* on the
  canonical free-phase fidelity. Users in that case should compute the
  authoritative value via a per-subsystem free-phase fidelity that grid-searches
  each independent phase generator separately.
"""
function phase_max_fidelity(ψ_T::AbstractVector, ψ_goal::AbstractVector; n_grid::Int = 128)
    d = length(ψ_T)
    @assert length(ψ_goal) == d "ψ_T and ψ_goal must have the same length"
    F_max = 0.0
    @inbounds for i = 0:(n_grid-1)
        φ = 2π * i / n_grid
        s = zero(ComplexF64)
        for k = 0:(d-1)
            s += conj(cis(k * φ) * ψ_goal[k+1]) * ψ_T[k+1]
        end
        F = abs2(s)
        F > F_max && (F_max = F)
    end
    return F_max
end
