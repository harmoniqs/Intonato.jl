# ============================================================================ #
#                          Iteration history types
# ============================================================================ #

"""
    IterationRecord

Per-iteration snapshot from `solve!(::PulseTuningProblem)`.

# Fields
- `y_exp::Vector{Measurement}`: experimental measurements
- `J_exp::Float64`: **debiased** whitened cost `J̃ = Ĵ − tr(WΣWᵀ)` vs y_goal
  (equal to the raw SSR for all-deterministic measurement models)
- `J_hat::Float64`: raw whitened cost `Ĵ = ‖W r‖²` (biased up by the noise
  floor; kept for diagnostics and selection studies)
- `step_size::Float64`: applied fraction α (β for one-shot policies, the
  armijo α for line-search policies, 0.0 if rejected)
- `tr_scale::Float64`: trust region scaling factor after this iteration's
  acceptance decision
- `accepted::Bool`: whether the measured trial was accepted as the new base
- `F_model::Union{Nothing,Float64}`: strategy-reported model-fidelity
  diagnostic for this iteration (`last_f_model` hook; `nothing` if the
  strategy reports none)
- `pulse::AbstractPulse`: pulse used for this iteration's experiment
- `θ::Union{Nothing,NamedTuple}`: measurement-time trajectory snapshot
  `(; data, global_data)` — restoring it into `z_ref` reproduces the iterate
  this record measured (selectors use this)
"""
struct IterationRecord
    y_exp::Vector{Measurement}
    J_exp::Float64
    J_hat::Float64
    step_size::Float64
    tr_scale::Float64
    accepted::Bool
    F_model::Union{Nothing,Float64}
    pulse::AbstractPulse
    θ::Union{Nothing,NamedTuple}
    # Per-phase wall-clock timings (seconds) for this outer iteration.
    # `nlp` is the inner subproblem solve; `sysid` includes device-model
    # adapt! plus any strategy-internal calibration; `total` excludes
    # post-iter rollout/record overhead. Phases that didn't run stay 0.0.
    timing::NamedTuple{(:experiment, :sysid, :nlp, :armijo, :total),NTuple{5,Float64}}
end

# Loggers opt into chassis iteration records by adding a method; the default
# is a no-op so ExperimentRecord-only loggers keep working unchanged.
record!(::AbstractExperimentLogger, ::IterationRecord) = nothing
record!(lg::InMemoryExperimentLogger, rec::IterationRecord) =
    push!(lg.iteration_records, rec)

"""
    TuningResult

Result of `solve!(::PulseTuningProblem)`.

# Fields
- `history::Vector{IterationRecord}`: per-iteration snapshots
- `converged::Bool`: whether measurement error reached tolerance
- `n_experiments::Int`: total experiment calls (including line search)
"""
struct TuningResult
    history::Vector{IterationRecord}
    converged::Bool
    n_experiments::Int
end

# ============================================================================ #
#                              Helpers
# ============================================================================ #

# The chassis keeps only `_check_nominal_fidelity` (a generic precondition used
# by every closed-loop path). Everything strategy-specific — the inner step, its
# trust-region representation, and any parameter calibration — lives behind the
# strategy interface.

# Flattened residual r = y_goal − y_exp in model element order (the vector the
# whitened cost Ĵ = ‖W r‖² and the acceptance statistics are built from).
_flat_residual(y_goal::Vector{Measurement}, y_exp::Vector{Measurement}) =
    reduce(vcat, Vector{Float64}[g.data .- e.data for (g, e) in zip(y_goal, y_exp)])

"""
    _check_nominal_fidelity(qcp, threshold; verbose, path_label)

Refuse to run the outer loop if `Piccolo.fidelity(qcp)` is below the
threshold. The closed-loop paths call this before starting; running on a
poorly-converged warm-start is wasteful. Silently passes if `fidelity` isn't
defined for the QCP type. Pass `threshold = 0` to skip. `path_label` is used
only in log messages.
"""
function _check_nominal_fidelity(
    qcp,
    threshold::Real;
    verbose::Bool,
    path_label::AbstractString,
)
    threshold > 0 || return nothing
    F_nom = try
        Piccolo.fidelity(qcp)
    catch
        nothing
    end
    isnothing(F_nom) && return nothing
    if F_nom < threshold
        error(
            "Nominal QCP fidelity ($(round(F_nom, digits=4))) is below " *
            "min_nominal_fidelity ($threshold). The QCP likely did not " *
            "converge — running $path_label on this pulse is wasteful. " *
            "Either solve the QCP more carefully (more iterations, better " *
            "init, or curriculum) or pass min_nominal_fidelity=0 to skip.",
        )
    end
    verbose && @info "$path_label: nominal fidelity check passed" F_nom threshold
    return nothing
end

# ============================================================================ #
#                          PulseTuningProblem
# ============================================================================ #

"""
    PulseTuningProblem

Strategy-generic closed-loop pulse-tuning chassis. Wraps a
`QuantumControlProblem` together with an experiment, a measurement model, a
trust-region representation, a tuning `strategy`, and a predictive
`device_model`.

After `solve!`, the QCP's trajectory and qtraj are updated in-place with the
tuned result.

# Fields
- `qcp::QuantumControlProblem`: the original (solved) optimization problem
- `experiment::AbstractExperiment`: hardware or simulated experiment
- `measurement_model::MeasurementModel`: measurement functions and knot indices
- `R_tr::NamedTuple`: trust-region weights per component, e.g. `(u=1e-2,)`
- `Q_meas::Union{Float64, Vector{Float64}}`: measurement-matching weight(s)
- `strategy::S`: the inner tuning strategy (the chassis/strategy split). The
  strategy provides the inner step and owns its own configuration; the default
  `IdentityStrategy` does no tuning (the loop runs but leaves the pulse unchanged).
- `device_model::M`: the predictive device model the loop plans against
  (`AbstractDeviceModel`). Defaults to a `NominalModel` wrapping the QCP's
  nominal system; `solve!` calls `adapt!(device_model, …)` each iteration
  (a no-op for `NominalModel`).
- `y_goal::Union{Nothing, Vector{Measurement}}`: explicit goal measurements,
  FIXED for the whole solve. `nothing` (default) → `solve!` resolves the goal
  once at solve start from the strategy's `tuning_goal` hook (back-compat).
  Supplied → used verbatim. Either way the resolved goal is computed **once**
  and never recomputed mid-solve — recomputing it from the current command
  makes the loop chase its own tail (the chained-loop-drift invariant).
- `result::Union{Nothing, TuningResult}`: populated by `solve!`
"""
mutable struct PulseTuningProblem{S<:AbstractTuningStrategy,M<:AbstractDeviceModel} <:
               AbstractPulseTuningProblem
    qcp::QuantumControlProblem
    experiment::AbstractExperiment
    measurement_model::MeasurementModel
    R_tr::NamedTuple
    Q_meas::Union{Float64,Vector{Float64}}
    # Chassis/strategy split: the inner step and the predictive device model.
    strategy::S
    device_model::M
    # Acceptance policy (nothing ⇒ solve! builds a LineSearchAcceptance from
    # its legacy γ / line_search kwargs).
    acceptance::Union{Nothing,AcceptancePolicy}
    # End-of-run iterate selector (nothing ⇒ FinalIterate, i.e. legacy).
    selector::Union{Nothing,IterateSelector}
    # Explicit fixed goal (nothing ⇒ resolved once from tuning_goal in solve!).
    y_goal::Union{Nothing,Vector{Measurement}}
    result::Union{Nothing,TuningResult}
end

"""
    PulseTuningProblem(qcp, experiment, model; R_tr, Q_meas, strategy, device_model)

Construct a strategy-generic pulse tuning problem from a `QuantumControlProblem`.

**Strategy + device model (chassis/strategy split).** The problem is
parametrized as `PulseTuningProblem{S,M}` on its tuning `strategy` and
`device_model`. The chassis is strategy-agnostic — it owns the experiment /
convergence / line-search / trust-region-scalar / record loop and delegates the
inner step (and everything inner-specific, including any parameter calibration)
to the strategy via the generic strategy interface. The `strategy` field defaults
to the lightweight `IdentityStrategy` (no tuning). `device_model` defaults to a
`NominalModel` wrapping the QCP's nominal system. Pass `strategy=`/`device_model=`
to override.
"""
function PulseTuningProblem(
    qcp::QuantumControlProblem,
    experiment::AbstractExperiment,
    measurement_model::MeasurementModel;
    R_tr::NamedTuple = (;),
    Q_meas::Union{Float64,Vector{Float64}} = 1.0,
    strategy::Union{Nothing,AbstractTuningStrategy} = nothing,
    device_model::Union{Nothing,AbstractDeviceModel} = nothing,
    acceptance::Union{Nothing,AcceptancePolicy} = nothing,
    selector::Union{Nothing,IterateSelector} = nothing,
    y_goal::Union{Nothing,Vector{Measurement}} = nothing,
)
    # Default strategy: a lightweight no-op placeholder. A tuning strategy is
    # provided by passing `strategy=`; the strategy carries its own config and
    # `solve!` calls `prepare_strategy` to build per-solve state from it.
    strat = isnothing(strategy) ? IdentityStrategy() : strategy
    # Default device model: a NominalModel wrapping the QCP's nominal system.
    # `adapt!` is a no-op for NominalModel.
    devmodel = if isnothing(device_model)
        NominalModel(get_system(qcp.qtraj), qcp.qtraj.initial, qcp.qtraj.goal)
    else
        device_model
    end

    return PulseTuningProblem(
        qcp,
        experiment,
        measurement_model,
        R_tr,
        Q_meas,
        strat,
        devmodel,
        acceptance,
        selector,
        y_goal,
        nothing,
    )
end

# ============================================================================ #
#                              solve!
# ============================================================================ #

"""
    solve!(ptp::PulseTuningProblem; kwargs...)

Run the outer closed-loop tuning loop. Updates `ptp.qcp` in-place with the
tuned result. The chassis is strategy-generic: it owns the experiment call,
convergence check, line search, acceptance, the trust-region scalar schedule,
and recording, and delegates the inner step to `ptp.strategy`.

# Keyword Arguments
- `max_iter::Int`: maximum outer iterations (default: 5)
- `tol::Float64`: convergence tolerance on measurement error (default: 1e-3)
- `ipopt_options::NamedTuple`: options forwarded to the inner solve (default: `(;)`)
- `line_search::Bool`: enable Armijo backtracking (default: true). Together
  with `γ` this configures the default `LineSearchAcceptance` policy; both are
  ignored when the problem carries an explicit `acceptance` policy.
- `verbose::Bool`: print iteration info (default: true)
- `γ::Float64`: trust region schedule factor (default: 0.8)
- `max_rejections::Union{Nothing, Int}`: max consecutive line search rejections
  before early stop. `nothing` = never stop early (default), `0` = stop on first
  rejection, `n` = stop after n consecutive rejections.
- `min_nominal_fidelity::Float64`: refuse to run if the nominal QCP's simulated
  fidelity is below this threshold (default: 0.8). Tuning on a poorly-converged
  pulse is wasteful. Set to 0 to skip the check. Silently skips if
  `fidelity(ptp.qcp)` is not defined for this QCP type.
- `polyak_avg::Int`: average the trajectory over the last `polyak_avg` iterates
  before syncing (default: 0, disabled). Reduces variance of the final iterate.
"""
function Piccolo.solve!(
    ptp::PulseTuningProblem;
    max_iter::Int = 5,
    tol::Float64 = 1e-3,
    ipopt_options::NamedTuple = (;),
    line_search::Bool = true,
    verbose::Bool = true,
    γ::Float64 = 0.8,
    max_rejections::Union{Nothing,Int} = nothing,
    min_nominal_fidelity::Float64 = 0.8,
    polyak_avg::Int = 0,
    logger::AbstractExperimentLogger = NullExperimentLogger(),
)
    # Sanity check: refuse to run on a poorly-converged nominal QCP.
    _check_nominal_fidelity(ptp.qcp, min_nominal_fidelity; verbose, path_label = "QILC")

    qcp = ptp.qcp
    z_ref = qcp.prob.trajectory

    # Prepare the inner tuning strategy. The chassis is strategy-agnostic: it
    # owns the experiment/convergence/line-search/tr-scalar/record loop and
    # delegates the inner step (and everything inner-specific) to the generic
    # strategy interface (`prepare_strategy`, `tuning_goal`,
    # `candidate_trajectory`, `last_timings`, `accepts_global_data`,
    # `step`). `prepare_strategy` builds any per-solve state the strategy needs
    # from `ptp`, `z_ref`, and the strategy's own config. For the default
    # `IdentityStrategy` this is a no-op and the loop leaves the pulse unchanged.
    strategy = prepare_strategy(ptp.strategy, ptp, z_ref; verbose)
    # Resolve the goal measurements ONCE for the whole solve: the explicit
    # y_goal kwarg verbatim if supplied, else the strategy's tuning_goal hook.
    # Never recomputed mid-solve (the chained-loop-drift invariant).
    y_goal = isnothing(ptp.y_goal) ? tuning_goal(strategy, ptp, z_ref) : ptp.y_goal

    # Resolve the acceptance policy: an explicit `ptp.acceptance` wins; else
    # the legacy γ / line_search kwargs configure a LineSearchAcceptance
    # (line_search=false ⇒ full-step, schedule-free — historical behavior).
    policy =
        isnothing(ptp.acceptance) ? LineSearchAcceptance(; γ, line_search) :
        ptp.acceptance
    reset_acceptance!(policy)

    history = IterationRecord[]
    tr_scale = 1.0
    n_experiments = 0
    consecutive_rejections = 0
    converged = false

    # Last-accepted-iterate snapshot, for reject-and-revert policies
    # (OneShotAcceptance): on an accepted trial the measured iterate is
    # snapshotted before the next candidate is applied on top; on a revert
    # the chassis restores it and discards the candidate.
    z_accepted_data = copy(z_ref.data)
    z_accepted_global = z_ref.global_dim > 0 ? copy(z_ref.global_data) : nothing

    # Resolve the end-of-run iterate selector: an explicit `ptp.selector`
    # wins; the deprecated `polyak_avg` kwarg forwards to PolyakAverage;
    # default is the literal final iterate (legacy).
    selector = if !isnothing(ptp.selector)
        ptp.selector
    elseif polyak_avg > 0
        @warn "solve!(…; polyak_avg=n) is deprecated — pass selector = PolyakAverage(n) to PulseTuningProblem instead" maxlog =
            1
        PolyakAverage(polyak_avg)
    else
        FinalIterate()
    end
    reset_selector!(selector)

    # The ACTIVE goal + measurement model. These start as the resolved fixed
    # goal / declared model and are only ever RESTRICTED, mid-campaign, when
    # the experiment starts returning fewer measurements (e.g. the lab σ_z
    # fallback) — see the dimension-change guard in the loop.
    y_goal_active = y_goal
    model_active = ptp.measurement_model

    for i = 1:max_iter
        # Per-iter timing accumulator. Phases that don't run for an iter
        # stay at 0.0.
        t_experiment = 0.0;
        t_sysid = 0.0;
        t_nlp = 0.0;
        t_armijo = 0.0
        t_iter_start = time()

        # 1. Extract pulse from current trajectory and run experiment. θ is
        # the measurement-time trajectory snapshot — selectors restore it to
        # reproduce the iterate this iteration measured.
        pulse = extract_pulse(qcp.qtraj, z_ref)
        θ_snapshot = (
            data = copy(z_ref.data),
            global_data = z_ref.global_dim > 0 ? copy(z_ref.global_data) : nothing,
        )
        t_experiment = @elapsed begin
            y_exp = run_experiment(ptp.experiment, pulse)
        end
        n_experiments += 1

        # 1a. Measurement-dimension change (e.g. the lab stops returning a
        # requested σ_z): restrict the active goal + model to the returned
        # length, reset the acceptance base (costs are no longer comparable
        # across the change), and warn.
        if length(y_exp) != length(y_goal_active)
            length(y_exp) < length(y_goal_active) || error(
                "experiment returned $(length(y_exp)) measurements, more than " *
                "the $(length(y_goal_active)) the goal declares",
            )
            @warn "Measurement dimension changed mid-campaign: experiment returned " *
                  "$(length(y_exp)) of $(length(y_goal_active)) measurements — " *
                  "restricting the goal and resetting the acceptance base." iter = i
            y_goal_active = y_goal_active[1:length(y_exp)]
            model_active = MeasurementModel(
                model_active.state_name,
                model_active.measurements[1:length(y_exp)],
                model_active.indices[1:length(y_exp)],
            )
            reset_acceptance!(policy)
        end

        # 2. Whitened cost statistics + convergence. With an all-deterministic
        # measurement model w ≡ 1 and σ² ≡ 0, so Ĵ equals the legacy raw SSR
        # and J̃ = Ĵ (behavior-preserving). Records and the convergence check
        # use the debiased J̃ — chassis, strategies, and logs speak one unit.
        w, σ2 = whiten(model_active, y_exp)
        r = _flat_residual(y_goal_active, y_exp)
        J_hat = sum(abs2, w .* r)
        J_exp = debiased_cost(J_hat, w, σ2)
        verbose && @info "QILC iter $i" J_exp tr_scale

        if J_exp ≤ tol
            t_total = time() - t_iter_start
            rec = IterationRecord(
                y_exp,
                J_exp,
                J_hat,
                1.0,
                tr_scale,
                true,
                nothing,
                pulse,
                θ_snapshot,
                (
                    experiment = t_experiment,
                    sysid = 0.0,
                    nlp = 0.0,
                    armijo = 0.0,
                    total = t_total,
                ),
            )
            push!(history, rec)
            record!(logger, rec)
            converged = true
            break
        end

        # 2a. Refine the device model from the latest experiment data BEFORE
        # the strategy step — recalibration must precede Jacobian assembly to
        # be useful (this inverts the pre-seam order). For a strategy that
        # owns model adaptation internally, `device_model` is a `NominalModel`
        # and `adapt!` is a no-op dispatched away at compile time.
        t_adapt = @elapsed begin
            adapt!(ptp.device_model, pulse, y_exp)
        end

        # 2b–6. Inner step delegated to the strategy. The strategy may mutate
        # z_ref's global data (e.g. parameter calibration) and leaves its
        # candidate trajectory populated for acceptance; it stashes the
        # per-phase sysid/NLP timings for this iteration's IterationRecord.
        ctx = (;
            pulse,
            y_exp,
            J_exp,
            z_ref,
            iter = i,
            tr_scale,
            ipopt_options,
            verbose,
            qcp,
            y_goal = y_goal_active,
            device_model = ptp.device_model,
        )
        pulse_cand = step(strategy, ctx)
        timings = last_timings(strategy)
        t_sysid = t_adapt + timings.sysid
        t_nlp = timings.nlp
        F_model = last_f_model(strategy)

        # The strategy's candidate trajectory holds the iterate to accept
        # (`nothing` for a no-op strategy like IdentityStrategy ⇒ no update).
        cand_traj = candidate_trajectory(strategy)

        # 7. Acceptance decision (policy seam): applied fraction α, accept /
        # revert, extra experiment evals, and the trust-scale evolution — all
        # policy-owned (LineSearchAcceptance reproduces the historical Armijo
        # + γ-schedule behavior verbatim).
        local decision
        t_armijo = @elapsed begin
            decision = decide(
                policy,
                (;
                    experiment = ptp.experiment,
                    pulse,
                    pulse_cand,
                    y_goal = y_goal_active,
                    J_hat,
                    J_tilde = J_exp,
                    r,
                    w,
                    σ2,
                    tr_scale,
                    iter = i,
                    verbose,
                ),
            )
        end
        α = decision.α
        n_experiments += decision.n_evals

        # 8. Apply the decision. A revert restores the last accepted iterate
        # and discards the candidate (reject-and-revert policies); an accepted
        # trial is snapshotted BEFORE the next candidate is applied on top.
        if decision.revert
            z_ref.data .= z_accepted_data
            if !isnothing(z_accepted_global)
                z_ref.global_data .= z_accepted_global
            end
        elseif decision.accepted
            z_accepted_data .= z_ref.data
            if !isnothing(z_accepted_global)
                z_accepted_global .= z_ref.global_data
            end
        end

        # Interpolate toward the candidate trajectory in-place. A no-op
        # strategy (IdentityStrategy) returns no candidate trajectory, so
        # there is nothing to interpolate and the pulse is left unchanged.
        if !decision.revert && α > 0.0 && !isnothing(cand_traj)
            z_ref.data .= (1 - α) .* z_ref.data .+ α .* cand_traj.data
            # Whether to also accept the candidate's globals is strategy-owned.
            # A strategy that owns its globals internally (e.g. via a separate
            # calibration step that writes them directly into z_ref) reports
            # `accepts_global_data == false` so the candidate's phantom globals
            # don't overwrite the owned value; a strategy that co-optimizes
            # globals with the controls reports `true`.
            if z_ref.global_dim > 0 && accepts_global_data(strategy)
                z_ref.global_data .=
                    ((1 - α) .* z_ref.global_data .+ α .* cand_traj.global_data)
            end
        end

        if decision.accepted && α > 0.0
            consecutive_rejections = 0
        else
            consecutive_rejections += 1
            if !isnothing(max_rejections) && consecutive_rejections > max_rejections
                t_total = time() - t_iter_start
                rec = IterationRecord(
                    y_exp,
                    J_exp,
                    J_hat,
                    α,
                    tr_scale,
                    decision.accepted,
                    F_model,
                    pulse,
                    θ_snapshot,
                    (
                        experiment = t_experiment,
                        sysid = t_sysid,
                        nlp = t_nlp,
                        armijo = t_armijo,
                        total = t_total,
                    ),
                )
                push!(history, rec)
                record!(logger, rec)
                verbose &&
                    @info "QILC: stopping after $consecutive_rejections consecutive rejections"
                break
            end
        end

        # 9. Trust-region scalar evolution — policy-owned (γ schedule for
        # LineSearchAcceptance; shrink-only halving for OneShotAcceptance).
        tr_scale = decision.tr_scale

        t_total = time() - t_iter_start
        rec = IterationRecord(
            y_exp,
            J_exp,
            J_hat,
            α,
            tr_scale,
            decision.accepted,
            F_model,
            pulse,
            θ_snapshot,
            (
                experiment = t_experiment,
                sysid = t_sysid,
                nlp = t_nlp,
                armijo = t_armijo,
                total = t_total,
            ),
        )
        push!(history, rec)
        record!(logger, rec)

        verbose &&
            @info "QILC iter $i timing (s)" experiment=t_experiment sysid=t_sysid nlp=t_nlp armijo=t_armijo total=t_total

        # Per-iteration selector hook (PolyakAverage accumulates post-step
        # iterates here).
        observe!(selector, z_ref, rec, i, max_iter)
    end

    # End-of-run iterate selection (best-J̃ / re-measure / Polyak / final).
    n_experiments += select_iterate!(
        selector,
        z_ref,
        history,
        (;
            experiment = ptp.experiment,
            measurement_model = model_active,
            y_goal = y_goal_active,
            verbose,
        ),
    )

    # Sync QCP: extract pulse + rollout to update qtraj.pulse and qtraj.solution
    sync_trajectory!(qcp)

    ptp.result = TuningResult(history, converged, n_experiments)

    return nothing
end
