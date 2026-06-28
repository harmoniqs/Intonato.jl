# ============================================================================ #
#                          Iteration history types
# ============================================================================ #

"""
    IterationRecord

Per-iteration snapshot from `solve!(::PulseTuningProblem)`.

# Fields
- `y_exp::Vector{Measurement}`: experimental measurements
- `J_exp::Float64`: measurement error vs y_goal
- `step_size::Float64`: α from line search (1.0 if no line search, 0.0 if rejected)
- `tr_scale::Float64`: trust region scaling factor at this iteration
- `pulse::AbstractPulse`: pulse used for this iteration's experiment
"""
struct IterationRecord
    y_exp::Vector{Measurement}
    J_exp::Float64
    step_size::Float64
    tr_scale::Float64
    pulse::AbstractPulse
    # Per-phase wall-clock timings (seconds) for this outer iteration.
    # `nlp` is the inner subproblem solve; `total` excludes post-iter
    # rollout/record overhead. Phases that didn't run for a given iter stay 0.0.
    timing::NamedTuple{(:experiment, :sysid, :nlp, :armijo, :total),NTuple{5,Float64}}
end

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
- `line_search::Bool`: enable Armijo backtracking (default: true)
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
    y_goal = tuning_goal(strategy, ptp, z_ref)

    history = IterationRecord[]
    tr_scale = 1.0
    n_experiments = 0
    consecutive_rejections = 0
    converged = false

    # Polyak-Ruppert averaging buffers — accumulate the last `polyak_avg` iters'
    # trajectory data + global data, then write the running average into z_ref
    # at the end of solve!. Reduces variance of the final iterate vs picking
    # the literal final or "best-J" iterate, which can chase shot noise on
    # hardware. polyak_avg=0 disables (current behavior preserved).
    polyak_data_acc = polyak_avg > 0 ? zero(z_ref.data) : nothing
    polyak_global_acc =
        polyak_avg > 0 && z_ref.global_dim > 0 ? zero(z_ref.global_data) : nothing
    polyak_count = 0

    for i = 1:max_iter
        # Per-iter timing accumulator. Phases that don't run for an iter
        # stay at 0.0.
        t_experiment = 0.0;
        t_sysid = 0.0;
        t_nlp = 0.0;
        t_armijo = 0.0
        t_iter_start = time()

        # 1. Extract pulse from current trajectory and run experiment
        pulse = extract_pulse(qcp.qtraj, z_ref)
        t_experiment = @elapsed begin
            y_exp = run_experiment(ptp.experiment, pulse)
        end
        n_experiments += 1

        # 2. Check convergence
        J_exp = measurement_error(y_exp, y_goal)
        verbose && @info "QILC iter $i" J_exp tr_scale

        if J_exp ≤ tol
            t_total = time() - t_iter_start
            push!(
                history,
                IterationRecord(
                    y_exp,
                    J_exp,
                    1.0,
                    tr_scale,
                    pulse,
                    (
                        experiment = t_experiment,
                        sysid = 0.0,
                        nlp = 0.0,
                        armijo = 0.0,
                        total = t_total,
                    ),
                ),
            )
            converged = true
            break
        end

        # 2a–6. Inner step delegated to the strategy. The strategy may mutate
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
            y_goal,
            device_model = ptp.device_model,
        )
        pulse_cand = step(strategy, ctx)
        timings = last_timings(strategy)
        t_sysid = timings.sysid
        t_nlp = timings.nlp

        # Refine the device model from the latest experiment data. For a
        # strategy that owns model adaptation internally, `device_model` is a
        # `NominalModel` and `adapt!` is a no-op dispatched away at compile time.
        adapt!(ptp.device_model, pulse, y_exp)

        # The strategy's candidate trajectory holds the iterate to accept
        # (`nothing` for a no-op strategy like IdentityStrategy ⇒ no update).
        cand_traj = candidate_trajectory(strategy)

        # 7. Line search (optional)
        if line_search
            local α_value
            t_armijo = @elapsed begin
                α_value, ls_evals =
                    armijo_line_search(ptp.experiment, pulse, pulse_cand, y_goal, J_exp)
            end
            α = α_value
            n_experiments += ls_evals
        else
            α = 1.0
        end

        # 8. Accept step — interpolate trajectory in-place. A no-op strategy
        # (IdentityStrategy) returns no candidate trajectory, so there is
        # nothing to interpolate and the pulse is left unchanged.
        if α > 0.0 && !isnothing(cand_traj)
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
            consecutive_rejections = 0
        elseif α > 0.0
            # Accepted step but no candidate trajectory (no-op tuning) — count
            # as a non-rejection so the rejection-based early stop isn't tripped.
            consecutive_rejections = 0
        else
            consecutive_rejections += 1
            if !isnothing(max_rejections) && consecutive_rejections > max_rejections
                t_total = time() - t_iter_start
                push!(
                    history,
                    IterationRecord(
                        y_exp,
                        J_exp,
                        α,
                        tr_scale,
                        pulse,
                        (
                            experiment = t_experiment,
                            sysid = t_sysid,
                            nlp = t_nlp,
                            armijo = t_armijo,
                            total = t_total,
                        ),
                    ),
                )
                verbose &&
                    @info "QILC: stopping after $consecutive_rejections consecutive rejections"
                break
            end
        end

        # 9. Trust region schedule
        # The schedule grows/shrinks `tr_scale` based on the line-search outcome
        # (α). With `line_search=false`, α is hardcoded to 1.0, so the schedule
        # would unconditionally shrink `R_tr_effective = R_tr · tr_scale` by γ
        # every iteration — driving the trust region to ~0 and leaving the NLP
        # effectively unregularized in `u`. That makes `line_search=false` unsafe
        # at large mismatch. Skip the adaptive scaling when line search is off;
        # the user's `R_tr` then holds verbatim across iterations.
        if line_search
            tr_scale *= (α ≥ 0.5 ? γ : 1 / γ)
        end

        t_total = time() - t_iter_start
        push!(
            history,
            IterationRecord(
                y_exp,
                J_exp,
                α,
                tr_scale,
                pulse,
                (
                    experiment = t_experiment,
                    sysid = t_sysid,
                    nlp = t_nlp,
                    armijo = t_armijo,
                    total = t_total,
                ),
            ),
        )

        verbose &&
            @info "QILC iter $i timing (s)" experiment=t_experiment sysid=t_sysid nlp=t_nlp armijo=t_armijo total=t_total

        # Polyak-Ruppert averaging: accumulate the last polyak_avg iters'
        # trajectory data into a running mean. Skips iters that line search
        # rejected (those z_ref values weren't promoted).
        if polyak_avg > 0 && i > max_iter - polyak_avg
            polyak_count += 1
            polyak_data_acc .+= z_ref.data
            if !isnothing(polyak_global_acc)
                polyak_global_acc .+= z_ref.global_data
            end
        end
    end

    # Polyak averaging: write averaged trajectory into z_ref before sync.
    if polyak_avg > 0 && polyak_count > 0
        z_ref.data .= polyak_data_acc ./ polyak_count
        if !isnothing(polyak_global_acc)
            z_ref.global_data .= polyak_global_acc ./ polyak_count
        end
        verbose && @info "QILC: Polyak-averaged final iterate over last $polyak_count iters"
    end

    # Sync QCP: extract pulse + rollout to update qtraj.pulse and qtraj.solution
    sync_trajectory!(qcp)

    ptp.result = TuningResult(history, converged, n_experiments)

    return nothing
end
