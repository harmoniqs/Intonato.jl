# This file contains tests for the surrounding source directory: the Piccolo
# iso-operator contract and the armijo_line_search items (line_search.jl).

# ============================================================================ #
# Tests for the iso-operator contract (Piccolo helper)
# ============================================================================ #

@testitem "operator_to_iso_operator (Piccolo) gives iso(P·ψ) = iso(P)·iso(ψ)" begin
    # Assert the iso contract on Piccolo's existing helper (which we use instead
    # of rolling our own).
    using Intonato
    using LinearAlgebra

    P = randn(ComplexF64, 4, 4)
    ψ = randn(ComplexF64, 4)
    iso_ψ = vcat(real.(ψ), imag.(ψ))
    iso_out_expected = let v = P * ψ
        vcat(real.(v), imag.(v))
    end
    isoP = operator_to_iso_operator(P)
    @test size(isoP) == (8, 8)
    @test isoP * iso_ψ ≈ iso_out_expected

    Id = Matrix{ComplexF64}(I, 3, 3)
    @test operator_to_iso_operator(Id) ≈ Matrix{Float64}(I, 6, 6)
end

# ============================================================================ #
# Tests for line_search.jl
# ============================================================================ #

@testitem "armijo_line_search accepts full step when candidate is better" begin
    using Intonato
    using LinearAlgebra

    # Two-level system
    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]

    sys = QuantumSystem(0.01 * σz, [σx], [1.0])

    N = 11
    T = 5.0
    times = range(0.0, T, length = N) |> collect

    ψ_init = ComplexF64[1.0, 0.0]
    ψ_goal = ComplexF64[0.0, 1.0]

    # Create a "bad" pulse (near-zero controls, state stays at |0⟩)
    pulse_bad = LinearSplinePulse(0.01 * randn(1, N), times)

    # Create a "better" pulse (π rotation: Ω·T = π/2 → Ω = π/(2T))
    u_good = fill(π / (2 * T), 1, N)
    pulse_good = LinearSplinePulse(u_good, times)

    # Experiment
    model = MeasurementModel(:ψ̃, [populations], [N])
    qtraj = KetTrajectory(sys, pulse_bad, ψ_init, ψ_goal)
    experiment = SimulatedExperiment(qtraj, model)

    # Goal: populations at |1⟩
    y_goal = [Measurement([0.0, 1.0], N)]

    # J_ref = cost at pulse_bad (should be high — state stays near |0⟩)
    y_ref = run_experiment(experiment, pulse_bad)
    J_ref = measurement_error(y_ref, y_goal)

    # Line search: pulse_good should be accepted with α > 0
    α, n_evals = armijo_line_search(experiment, pulse_bad, pulse_good, y_goal, J_ref)

    @test α > 0.0
    @test n_evals ≥ 1
end

@testitem "armijo_line_search rejects when candidate is worse" begin
    using Intonato
    using LinearAlgebra

    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    sys = QuantumSystem(0.01 * σz, [σx], [1.0])

    N = 11
    T = 5.0
    times = range(0.0, T, length = N) |> collect

    ψ_init = ComplexF64[1.0, 0.0]
    ψ_goal = ComplexF64[0.0, 1.0]

    # Good pulse (π rotation: Ω·T = π/2 → Ω = π/(2T))
    u_good = fill(π / (2 * T), 1, N)
    pulse_good = LinearSplinePulse(u_good, times)

    # Bad pulse (random noise — state stays near |0⟩)
    pulse_bad = LinearSplinePulse(0.01 * randn(1, N), times)

    model = MeasurementModel(:ψ̃, [populations], [N])
    qtraj = KetTrajectory(sys, pulse_good, ψ_init, ψ_goal)
    experiment = SimulatedExperiment(qtraj, model)

    y_goal = [Measurement([0.0, 1.0], N)]

    # J_ref at the good pulse (should be low)
    y_ref = run_experiment(experiment, pulse_good)
    J_ref = measurement_error(y_ref, y_goal)

    # Candidate is worse — line search should reject
    α, n_evals = armijo_line_search(experiment, pulse_good, pulse_bad, y_goal, J_ref)

    @test α == 0.0
    @test n_evals ≥ 1
end
