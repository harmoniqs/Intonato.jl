# ──── Fidelity measurement rows ───────────────────────────────────────────────
#
# Scalar fidelity as a first-class measurement row for MeasurementModel.
# This is the observable the low-rank Hessian strategy (Liu et al. 2026)
# measures by construction, and the "cheap sensor" arm of scalar-observable
# model-gradient tuning: one number per experiment, SPAM-lean, but carrying
# only the *magnitude* of the endpoint defect — no direction, and (at a
# calibrated fixed point) asymptotically no device-parameter information.

"""
    goal_fidelity(x_iso, ψ_goal_iso) -> Float64

Overlap fidelity ``F = |⟨ψ_goal|ψ⟩|²`` evaluated directly on ket iso-vecs
(`[Re(ψ); Im(ψ)]`), in pure real arithmetic:

``⟨g|ψ⟩ = (g_r·ψ_r + g_i·ψ_i) + i\\,(g_r·ψ_i − g_i·ψ_r)``

ForwardDiff-compatible (no complex intermediates, no branches). Invariant
under a *global* phase of either argument; local/relative phases are NOT
quotiented — use [`phase_max_fidelity_at`](@ref) when a free-phase frame
should be maximized over.
"""
function goal_fidelity(x_iso::AbstractVector, ψ_goal_iso::AbstractVector)
    d2 = length(ψ_goal_iso)
    @assert length(x_iso) == d2 "state and goal iso-vecs must have equal length"
    d = d2 ÷ 2
    re = zero(promote_type(eltype(x_iso), eltype(ψ_goal_iso)))
    im_ = zero(re)
    @inbounds for k = 1:d
        gr = ψ_goal_iso[k]
        gi = ψ_goal_iso[d+k]
        ψr = x_iso[k]
        ψi = x_iso[d+k]
        re += gr * ψr + gi * ψi
        im_ += gr * ψi - gi * ψr
    end
    return re^2 + im_^2
end

"""
    goal_fidelity_at(ψ_goal_iso) -> (x_iso -> [F])

Closure factory for `MeasurementModel`: a single-element fidelity measurement
against a fixed goal state (iso-vec form). ForwardDiff-compatible.

# Example
```julia
model = MeasurementModel(:ψ̃, [goal_fidelity_at(ket_to_iso(ψ_goal))], [N])
```
"""
goal_fidelity_at(ψ_goal_iso::AbstractVector) =
    x -> [goal_fidelity(x, collect(float.(ψ_goal_iso)))]

"""
    phase_max_fidelity_at(ψ_goal; n_grid = 128) -> (x_iso -> [F])

Closure factory for a *free-phase-maximized* fidelity measurement row:
``F = \\max_φ |⟨ψ_goal| e^{iφ n̂} |ψ⟩|²`` via [`phase_max_fidelity`](@ref)
(number-operator generator, grid search). Takes the goal as a **complex**
ket; converts the measured iso-vec internally.

Experiment-side only: the grid max is not smooth, and the implementation
uses complex arithmetic — do not differentiate through it (use
[`goal_fidelity_at`](@ref) for model-side Jacobians/Hessians and quotient
gauge with an explicit nuisance parameter instead).
"""
phase_max_fidelity_at(ψ_goal::AbstractVector{<:Complex}; n_grid::Int = 128) =
    x -> [phase_max_fidelity(iso_to_ket(x), ψ_goal; n_grid = n_grid)]

@testitem "goal_fidelity: values, invariances, ForwardDiff" begin
    using Intonato
    using LinearAlgebra
    using ForwardDiff

    ψg = ComplexF64[0.0, 1.0]
    g_iso = ket_to_iso(ψg)

    # Value tests: goal state → 1, orthogonal → 0, superposition → 1/2.
    @test goal_fidelity(ket_to_iso(ComplexF64[0.0, 1.0]), g_iso) ≈ 1.0
    @test goal_fidelity(ket_to_iso(ComplexF64[1.0, 0.0]), g_iso) ≈ 0.0
    @test goal_fidelity(ket_to_iso(ComplexF64[1.0, 1.0] / sqrt(2)), g_iso) ≈ 0.5

    # Global-phase invariance (both arguments).
    ψ = ComplexF64[0.6, 0.8im]
    @test goal_fidelity(ket_to_iso(cis(0.7) * ψ), g_iso) ≈
          goal_fidelity(ket_to_iso(ψ), g_iso)
    @test goal_fidelity(ket_to_iso(ψ), ket_to_iso(cis(1.3) * ψg)) ≈
          goal_fidelity(ket_to_iso(ψ), g_iso)

    # ForwardDiff gradient: finite, and matches central differences.
    f = x -> goal_fidelity(x, g_iso)
    x0 = ket_to_iso(ComplexF64[0.8, 0.6])
    grad = ForwardDiff.gradient(f, x0)
    @test all(isfinite, grad)
    ε = 1e-6
    for j in eachindex(x0)
        xp = copy(x0);
        xp[j] += ε
        xm = copy(x0);
        xm[j] -= ε
        @test grad[j] ≈ (f(xp) - f(xm)) / (2ε) atol = 1e-6
    end
end

@testitem "fidelity measurement rows in a MeasurementModel round-trip" begin
    using Intonato
    using LinearAlgebra

    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    sys = QuantumSystem(0.01 * σz, [σx], [1.0])
    N = 11
    times = collect(range(0.0, 5.0, length = N))
    pulse = LinearSplinePulse(0.1 * ones(1, N), times)
    ψ0 = ComplexF64[1.0, 0.0]
    ψg = ComplexF64[0.0, 1.0]

    # A one-row fidelity measurement model at the final knot.
    model = MeasurementModel(:ψ̃, [goal_fidelity_at(ket_to_iso(ψg))], [N])
    experiment = SimulatedExperiment(KetTrajectory(sys, pulse, ψ0, ψg), model)
    y = run_experiment(experiment, pulse)
    @test length(y) == 1
    @test length(y[1].data) == 1
    @test 0.0 ≤ y[1].data[1] ≤ 1.0 + 1e-12

    # Free-phase row agrees with the raw row up to the phase maximization
    # (≥ raw, ≤ 1) on the same experiment.
    model_pm = MeasurementModel(:ψ̃, [phase_max_fidelity_at(ψg)], [N])
    experiment_pm = SimulatedExperiment(KetTrajectory(sys, pulse, ψ0, ψg), model_pm)
    y_pm = run_experiment(experiment_pm, pulse)
    @test y_pm[1].data[1] ≥ y[1].data[1] - 1e-9
    @test y_pm[1].data[1] ≤ 1.0 + 1e-9
end
