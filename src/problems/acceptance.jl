# ============================================================================ #
#                       Acceptance-policy seam (chassis)
# ============================================================================ #
#
# The chassis owns: run experiment → compute Ĵ, J̃ → decide → apply the
# α-interpolation (or revert) → record. The DECISION — how much of the
# strategy's candidate to apply, whether the measured trial is accepted, and
# how the trust-region scalar evolves — is a policy object:
#
#   LineSearchAcceptance — today's Armijo backtracking + γ schedule (sim path)
#   OneShotAcceptance    — β-damped apply, noise-aware reject-and-revert,
#                          shrink-only radius (the hardware-hardened path)

"""
    AcceptancePolicy

Chassis seam deciding step acceptance, the applied fraction α, and the
trust-region scalar evolution. Concrete policies implement

    decide(policy, ctx) -> (; α, accepted, revert, tr_scale, n_evals)

where `ctx` is a NamedTuple the chassis assembles after the measurement and
the strategy step: `(; experiment, pulse, pulse_cand, y_goal, J_hat, J_tilde,
r, w, σ2, tr_scale, iter, verbose)`. Semantics of the returned fields:

- `α`: fraction of the candidate trajectory to apply (0 ⇒ no move).
- `accepted`: whether the measured trial becomes the new acceptance base
  (drives best-iterate bookkeeping and the rejection counter).
- `revert`: the chassis restores the last accepted iterate before continuing
  (reject-and-revert policies); the candidate is discarded.
- `tr_scale`: the (possibly updated) trust-region scalar.
- `n_evals`: extra experiment evaluations the policy spent (line-search probes).

Policies may be stateful across a solve; [`reset_acceptance!`](@ref) is called
once by `solve!` before the loop.
"""
abstract type AcceptancePolicy end

function decide end

"""
    reset_acceptance!(policy)

Clear per-solve state (acceptance base, stall counters). Called by `solve!`
before the outer loop. Default: no-op for stateless policies.
"""
reset_acceptance!(::AcceptancePolicy) = nothing

# ──── LineSearchAcceptance ────────────────────────────────────────────────────

"""
    LineSearchAcceptance(; γ = 0.8, line_search = true)

Today's Armijo acceptance path, extracted behavior-preserving from the chassis:
backtracking line search on the experiment (`armijo_line_search`), acceptance
iff α > 0, and the γ trust-scale schedule (`tr_scale *= α ≥ 0.5 ? γ : 1/γ`).

With `line_search = false` the step is applied fully (α = 1) and the schedule
is skipped — matching the chassis's historical `line_search=false` behavior
(the user's `R_tr` holds verbatim across iterations).

Each armijo probe costs an experiment evaluation; on one-shot hardware use
[`OneShotAcceptance`](@ref) instead.
"""
Base.@kwdef struct LineSearchAcceptance <: AcceptancePolicy
    γ::Float64 = 0.8
    line_search::Bool = true
end

function decide(p::LineSearchAcceptance, ctx)
    if !p.line_search
        # No probing, full application, no schedule: R_tr holds verbatim.
        return (α = 1.0, accepted = true, revert = false, tr_scale = ctx.tr_scale, n_evals = 0)
    end
    # Trial costs must be in the SAME units as J_ref = Ĵ (the whitened cost):
    # plug-in GLS whitening at each trial's own y. With an all-deterministic
    # model this reduces exactly to the raw measurement_error (w ≡ 1) — a raw
    # trial cost against a whitened J_ref would silently neuter backtracking.
    whitened_cost = y -> begin
        w, _ = whiten(ctx.measurement_model, y)
        sum(abs2, w .* _flat_residual(ctx.y_goal, y))
    end
    α, n_evals = armijo_line_search(
        ctx.experiment,
        ctx.pulse,
        ctx.pulse_cand,
        ctx.y_goal,
        ctx.J_hat;
        cost = whitened_cost,
    )
    tr_scale = ctx.tr_scale * (α ≥ 0.5 ? p.γ : 1 / p.γ)
    return (α = α, accepted = α > 0, revert = false, tr_scale = tr_scale, n_evals = n_evals)
end

# ──── OneShotAcceptance ───────────────────────────────────────────────────────

"""
    OneShotAcceptance(; β = 0.5, ρ_rej = 2.0, k = 3.0, stall_patience = 0,
                      tr_scale_min = 2.0^-8)

The hardware-hardened acceptance policy: no probing (a probe costs a
measurement round). The candidate step is always applied at the fixed fraction
`α = β` (classic ILC relaxation — the no-line-search counter to 2-cycle
overshoot), and step-size control happens through measured-outcome tests on
the *next* trial:

- **Reject-and-revert** (catastrophe cascade): the measured trial is rejected
  iff `Ĵ − J_base > max(ρ_rej·J_base, k·σ_Δ)`, where `J_base` is the last
  accepted trial's cost and `σ_Δ = √(σ_J²(trial) + σ_J²(base))` is the
  difference scale (both costs are noisy). On rejection the chassis reverts to
  the previous accepted iterate and `tr_scale` is halved **permanently** —
  the cascade auto-finds the linearization-validity radius. The `k·σ_Δ` guard
  keeps the cascade from firing on a noise fluctuation; with an
  all-deterministic measurement model `σ_Δ = 0` and this reduces to the
  legacy pure ratio test.
- **Stall shrink** (stochastic-approximation anneal): after `stall_patience`
  consecutive accepted trials with improvement `< max(0.01·J̃, k·σ_Δ)`,
  `tr_scale` is halved. `stall_patience = 0` disables.
- `tr_scale` is **shrink-only** (never grows across any accepted sequence),
  floored at `tr_scale_min`.

The first measured trial always establishes the acceptance base — no
rejection is possible before a base exists.
"""
Base.@kwdef mutable struct OneShotAcceptance <: AcceptancePolicy
    β::Float64 = 0.5
    ρ_rej::Float64 = 2.0
    k::Float64 = 3.0
    stall_patience::Int = 0
    tr_scale_min::Float64 = 2.0^-8
    # ── per-solve state (reset by reset_acceptance!) ──
    J_base::Union{Nothing,Float64} = nothing   # Ĵ at the last accepted trial
    σ_base::Float64 = 0.0                      # σ_J at the last accepted trial
    J̃_base::Float64 = Inf                      # debiased cost at the base
    stall_count::Int = 0
end

function reset_acceptance!(p::OneShotAcceptance)
    p.J_base = nothing
    p.σ_base = 0.0
    p.J̃_base = Inf
    p.stall_count = 0
    return nothing
end

function decide(p::OneShotAcceptance, ctx)
    σ_trial = cost_std(ctx.r, ctx.w, ctx.σ2)

    # First measured trial establishes the base — no rejection possible.
    if isnothing(p.J_base)
        p.J_base = ctx.J_hat
        p.σ_base = σ_trial
        p.J̃_base = ctx.J_tilde
        return (α = p.β, accepted = true, revert = false, tr_scale = ctx.tr_scale, n_evals = 0)
    end

    σΔ = diff_std(σ_trial, p.σ_base)

    # Catastrophe test on the difference scale.
    if ctx.J_hat - p.J_base > max(p.ρ_rej * p.J_base, p.k * σΔ)
        tr_scale = max(ctx.tr_scale / 2, p.tr_scale_min)
        ctx.verbose &&
            @info "OneShotAcceptance: rejecting trial (Ĵ=$(ctx.J_hat) vs base $(p.J_base)); tr_scale → $tr_scale"
        return (α = 0.0, accepted = false, revert = true, tr_scale = tr_scale, n_evals = 0)
    end

    # Accepted: stall bookkeeping, then the trial becomes the new base.
    tr_scale = ctx.tr_scale
    if p.stall_patience > 0
        improvement = p.J̃_base - ctx.J_tilde
        if improvement < max(0.01 * ctx.J_tilde, p.k * σΔ)
            p.stall_count += 1
            if p.stall_count ≥ p.stall_patience
                tr_scale = max(tr_scale / 2, p.tr_scale_min)
                p.stall_count = 0
                ctx.verbose &&
                    @info "OneShotAcceptance: stall — halving tr_scale → $tr_scale"
            end
        else
            p.stall_count = 0
        end
    end
    p.J_base = ctx.J_hat
    p.σ_base = σ_trial
    p.J̃_base = ctx.J_tilde
    return (α = p.β, accepted = true, revert = false, tr_scale = tr_scale, n_evals = 0)
end
