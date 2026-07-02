# ============================================================================ #
#                       Iterate-selector seam (chassis)
# ============================================================================ #
#
# One end-of-run iterate policy. Under measurement noise the literal final
# iterate is a random draw; the selector decides what the solve RETURNS:
#
#   FinalIterate       — the literal final iterate (legacy default)
#   NoiseCorrectedBestJ — argmin J̃ over accepted iterates (debiased selection)
#   TopKRemeasure      — re-measure the top-k at boosted shots, pick the winner
#   PolyakAverage      — average the last-n iterates (variance reduction)

"""
    IterateSelector

End-of-run iterate policy for `solve!(::PulseTuningProblem)`. Concrete
selectors implement

    select_iterate!(selector, z_ref, history, ctx) -> n_evals::Int

called once after the outer loop (before the QCP sync); it writes the selected
iterate into `z_ref` in-place and returns the number of extra experiment
evaluations it spent. `history` is the `Vector{IterationRecord}`; `ctx`
carries `(; experiment, measurement_model, y_goal, verbose)`.

Selectors that need per-iteration state additionally implement

    observe!(selector, z_ref, rec, i, max_iter)

called at the end of every completed outer iteration, and
`reset_selector!(selector)` to clear per-solve state.
"""
abstract type IterateSelector end

"""
    select_iterate!(selector, z_ref, history, ctx) -> n_evals::Int

Write the selected final iterate into `z_ref`. Default: no-op (the literal
final iterate stands). Returns the number of extra experiment evaluations.
"""
select_iterate!(::IterateSelector, z_ref, history, ctx) = 0

"""
    observe!(selector, z_ref, rec, i, max_iter)

Per-iteration hook (end of each completed outer iteration, after acceptance
application). Default: no-op.
"""
observe!(::IterateSelector, z_ref, rec, i, max_iter) = nothing

"""
    reset_selector!(selector)

Clear per-solve selector state. Called by `solve!` before the loop.
"""
reset_selector!(::IterateSelector) = nothing

# Restore a recorded θ snapshot (measurement-time trajectory state) into z_ref.
function _restore_iterate!(z_ref, θ::NamedTuple)
    z_ref.data .= θ.data
    if !isnothing(θ.global_data) && z_ref.global_dim > 0
        z_ref.global_data .= θ.global_data
    end
    return nothing
end

_accepted_records(history) = [r for r in history if r.accepted && !isnothing(r.θ)]

"""
    FinalIterate <: IterateSelector

The literal final iterate — the legacy behavior, and the default.
"""
struct FinalIterate <: IterateSelector end

"""
    NoiseCorrectedBestJ <: IterateSelector

Select the accepted iterate with the smallest **debiased** cost
`J̃ = Ĵ − tr(WΣWᵀ)`. Selecting on raw `Ĵ` under heteroscedastic shot noise is
biased low by the per-iterate noise floor (the documented 0.3–0.7 % best-J
selection bias); `J̃` removes the bias at the source.
"""
struct NoiseCorrectedBestJ <: IterateSelector end

function select_iterate!(::NoiseCorrectedBestJ, z_ref, history, ctx)
    acc = _accepted_records(history)
    isempty(acc) && return 0
    best = argmin(r -> r.J_exp, acc)
    _restore_iterate!(z_ref, best.θ)
    ctx.verbose && @info "NoiseCorrectedBestJ: selected iterate with J̃ = $(best.J_exp)"
    return 0
end

"""
    TopKRemeasure(; k = 3, reps_factor = 4)

Re-measure the top-`k` accepted iterates (ranked by recorded `J̃`) at boosted
shots — `run_experiment(exp, pulse; n_shots = n·reps_factor)`, with `n` the
largest shot count declared on the measurement model — and select the
re-measured winner. If fewer than `k` iterates were accepted, all accepted
iterates are re-measured. Costs `min(k, n_accepted)` extra experiment rounds.
"""
Base.@kwdef struct TopKRemeasure <: IterateSelector
    k::Int = 3
    reps_factor::Int = 4
end

# Largest declared shot count on the model (nothing when no shot-noise
# measurement is declared — the re-measure then runs at the default budget).
function _max_declared_shots(model::MeasurementModel)
    ns = [m.n_shots for m in model.measurements if m isa ShotNoiseMeasurement]
    return isempty(ns) ? nothing : maximum(ns)
end

function select_iterate!(s::TopKRemeasure, z_ref, history, ctx)
    acc = _accepted_records(history)
    isempty(acc) && return 0
    ranked = sort(acc; by = r -> r.J_exp)
    finalists = length(ranked) ≤ s.k ? ranked : ranked[1:s.k]

    n_base = _max_declared_shots(ctx.measurement_model)
    boost = isnothing(n_base) ? nothing : n_base * s.reps_factor
    model_boost =
        isnothing(boost) ? ctx.measurement_model :
        _with_n_shots(ctx.measurement_model, boost)

    best = nothing
    best_J = Inf
    n_evals = 0
    for rec in finalists
        y_re =
            isnothing(boost) ? run_experiment(ctx.experiment, rec.pulse) :
            run_experiment(ctx.experiment, rec.pulse; n_shots = boost)
        n_evals += 1
        w, σ2 = whiten(model_boost, y_re; W_task = get(ctx, :W_task, nothing))
        r = _flat_residual(ctx.y_goal, y_re)
        J̃ = debiased_cost(sum(abs2, w .* r), w, σ2)
        if J̃ < best_J
            best_J = J̃
            best = rec
        end
    end
    _restore_iterate!(z_ref, best.θ)
    ctx.verbose &&
        @info "TopKRemeasure: re-measured $(n_evals) finalists; winner J̃ = $best_J"
    return n_evals
end

"""
    PolyakAverage(n)

Polyak–Ruppert averaging: the returned iterate is the mean of the last `n`
post-acceptance iterates. Reduces variance of the final iterate vs picking
the literal final or a best-J iterate, which can chase shot noise on
hardware. `n = 0` disables (equivalent to [`FinalIterate`](@ref)).

This selector reproduces the chassis's former `polyak_avg` kwarg behavior;
that kwarg now forwards here with a deprecation note.
"""
mutable struct PolyakAverage <: IterateSelector
    n::Int
    data_acc::Any
    global_acc::Any
    count::Int
end

PolyakAverage(n::Int) = PolyakAverage(n, nothing, nothing, 0)

function reset_selector!(p::PolyakAverage)
    p.data_acc = nothing
    p.global_acc = nothing
    p.count = 0
    return nothing
end

function observe!(p::PolyakAverage, z_ref, rec, i, max_iter)
    (p.n > 0 && i > max_iter - p.n) || return nothing
    if isnothing(p.data_acc)
        p.data_acc = zero(z_ref.data)
        p.global_acc = z_ref.global_dim > 0 ? zero(z_ref.global_data) : nothing
    end
    p.count += 1
    p.data_acc .+= z_ref.data
    if !isnothing(p.global_acc)
        p.global_acc .+= z_ref.global_data
    end
    return nothing
end

function select_iterate!(p::PolyakAverage, z_ref, history, ctx)
    p.count > 0 || return 0
    z_ref.data .= p.data_acc ./ p.count
    if !isnothing(p.global_acc)
        z_ref.global_data .= p.global_acc ./ p.count
    end
    ctx.verbose && @info "PolyakAverage: averaged final iterate over last $(p.count) iters"
    return 0
end
