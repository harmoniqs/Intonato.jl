# ──── Tuning-problem + strategy interfaces ────────────────────────────────────

"""Supertype for closed-loop pulse-tuning problems (chassis types)."""
abstract type AbstractPulseTuningProblem end

"""
    AbstractTuningStrategy

The inner optimization step plugged into a `PulseTuningProblem` chassis.

Concrete strategies implement:

    step(strategy, ctx) -> candidate_pulse

`ctx` is a NamedTuple the chassis assembles each iteration:
`(; pulse, y_exp, y_goal, device_model, tr_scale, iter, qcp, z_ref)`.
The chassis owns the experiment call, convergence check, line search,
acceptance, the trust-region *scalar schedule*, and recording. The strategy
owns its trust-region *representation* and its inner solve.
"""
abstract type AbstractTuningStrategy end

function step end

# ──── Generic strategy interface (chassis ⇄ strategy boundary) ────────────────
#
# The `solve!(::PulseTuningProblem)` chassis is strategy-agnostic: it owns the
# experiment / convergence / line-search / trust-region-scalar / record loop and
# delegates the inner step (and everything inner-specific) to the strategy via
# the generic hooks below. Concrete strategies override these; the defaults make
# `IdentityStrategy` a no-op tuner (no candidate trajectory ⇒ the chassis leaves
# the pulse unchanged).

"""
    prepare_strategy(strategy, ptp, z_ref; solve_kwargs...) -> strategy

Build any per-solve state the strategy needs (subproblem handles, calibration
setup, goal measurements, …) from the parent problem `ptp` and its reference
trajectory `z_ref`, given the solve-time kwargs. Called once by `solve!` before
the outer loop. The default returns the strategy unchanged (stateless
strategies such as `IdentityStrategy` need no preparation).
"""
prepare_strategy(strategy::AbstractTuningStrategy, ptp, z_ref; kwargs...) = strategy

"""
    tuning_goal(strategy, ptp, z_ref) -> Vector{Measurement}

The goal measurements the chassis checks convergence against. Default:
`model_predict(z_ref, ptp.measurement_model)` — the model prediction at the
reference (nominal) trajectory.
"""
tuning_goal(::AbstractTuningStrategy, ptp, z_ref) =
    model_predict(z_ref, ptp.measurement_model)

"""
    candidate_trajectory(strategy) -> Union{Nothing, NamedTrajectory}

The trajectory whose `data` (and, if [`accepts_global_data`](@ref), `global_data`)
the chassis interpolates `z_ref` toward on an accepted step. Returning `nothing`
means "no trajectory-level update" — the chassis leaves `z_ref` unchanged, which
is exactly the no-op tuning behavior of `IdentityStrategy`.
"""
candidate_trajectory(::AbstractTuningStrategy) = nothing

"""
    last_timings(strategy) -> NamedTuple

Per-phase wall-clock timings stashed by the strategy's most recent `step`
(keys `:sysid`, `:nlp`). Default: zeros (stateless strategies do no inner solve).
"""
last_timings(::AbstractTuningStrategy) = (sysid = 0.0, nlp = 0.0)

"""
    last_f_model(strategy) -> Union{Nothing, Float64}

Model-fidelity diagnostic stashed by the strategy's most recent `step`
(e.g. the free-phase model fidelity a QP-ILC strategy computes per
iteration). Recorded in `IterationRecord.F_model`; powers the
`f_proxy_slack` regression warning. Default: `nothing` (no diagnostic).
"""
last_f_model(::AbstractTuningStrategy) = nothing

"""
    accepts_global_data(strategy) -> Bool

Whether the chassis should also interpolate `z_ref.global_data` toward the
candidate trajectory's globals on an accepted step. Default `false` — a
strategy that manages global parameters itself writes them directly into
`z_ref`, so the candidate's phantom globals must NOT overwrite them.
"""
accepts_global_data(::AbstractTuningStrategy) = false

"""
    IdentityStrategy <: AbstractTuningStrategy

Trivial strategy: return the current pulse unchanged. With the generic strategy
interface defaults (no candidate trajectory), a `PulseTuningProblem` carrying an
`IdentityStrategy` does no tuning — `solve!` runs the experiment / convergence /
record loop but leaves the pulse unchanged. This is the default stand-in; a
tuning run is configured by passing a concrete `strategy` that provides the
inner step.
"""
struct IdentityStrategy <: AbstractTuningStrategy end

step(::IdentityStrategy, ctx) = ctx.pulse

@testitem "AbstractTuningStrategy contract + IdentityStrategy" begin
    using Intonato
    using Intonato: step, AbstractTuningStrategy, AbstractPulseTuningProblem
    @test IdentityStrategy <: AbstractTuningStrategy
    # IdentityStrategy returns the candidate pulse unchanged given a context NamedTuple.
    ctx = (; pulse = :sentinel)
    @test step(IdentityStrategy(), ctx) == :sentinel
end
