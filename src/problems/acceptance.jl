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
    α, n_evals = armijo_line_search(
        ctx.experiment,
        ctx.pulse,
        ctx.pulse_cand,
        ctx.y_goal,
        ctx.J_hat,
    )
    tr_scale = ctx.tr_scale * (α ≥ 0.5 ? p.γ : 1 / p.γ)
    return (α = α, accepted = α > 0, revert = false, tr_scale = tr_scale, n_evals = n_evals)
end

