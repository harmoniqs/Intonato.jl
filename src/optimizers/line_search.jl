"""
    armijo_line_search(experiment, pulse_ref, pulse_cand, y_goal, J_ref; ρ, α_min)

Backtracking line search on pulse objects against the experiment.

Tries α = 1.0, ρ, ρ², … until the experimental cost decreases, or α < α_min.

# Arguments
- `experiment::AbstractExperiment`: the experiment to evaluate
- `pulse_ref::AbstractPulse`: current best pulse (α = 0)
- `pulse_cand::AbstractPulse`: candidate pulse from NLP solve (α = 1)
- `y_goal::Vector{Measurement}`: goal measurements (target for calibration)
- `J_ref::Float64`: experimental cost at pulse_ref (reused from outer loop)

# Keyword Arguments
- `ρ::Float64`: backtracking factor (default: 0.5)
- `α_min::Float64`: minimum step before rejecting (default: 0.01)

# Returns
- `α::Float64`: accepted step size (0.0 if rejected)
- `n_evals::Int`: number of experiment evaluations
"""
function armijo_line_search(
    experiment::AbstractExperiment,
    pulse_ref::AbstractPulse,
    pulse_cand::AbstractPulse,
    y_goal::Vector{Measurement},
    J_ref::Float64;
    ρ::Float64 = 0.5,
    α_min::Float64 = 0.01,
)
    α = 1.0
    n_evals = 0

    while α ≥ α_min
        pulse_trial = interpolate_pulse(pulse_ref, pulse_cand, α)
        y_trial = run_experiment(experiment, pulse_trial)
        J_trial = measurement_error(y_trial, y_goal)
        n_evals += 1

        if J_trial < J_ref
            return α, n_evals
        end

        α *= ρ
    end

    return 0.0, n_evals
end

# ============================================================================ #
