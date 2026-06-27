# This file contains tests for the surrounding source directory: construction
# of the strategy-generic PulseTuningProblem chassis. End-to-end tuning items
# (which require a concrete tuning strategy) live with the strategies they
# exercise.

@testitem "PulseTuningProblem constructor" begin
    using Intonato
    using LinearAlgebra

    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]

    sys_nom = QuantumSystem(0.01 * σz, [σx], [1.0])
    sys_true = QuantumSystem(0.0105 * σz, [σx], [1.0])

    N = 11
    T = 5.0
    times = range(0.0, T, length = N) |> collect
    pulse = LinearSplinePulse(0.1 * randn(1, N), times)

    ψ_init = ComplexF64[1.0, 0.0]
    ψ_goal = ComplexF64[0.0, 1.0]

    qtraj_nom = KetTrajectory(sys_nom, pulse, ψ_init, ψ_goal)
    qcp = SplinePulseProblem(qtraj_nom, N; Q = 100.0, R = 1e-2)

    model = MeasurementModel(:ψ̃, [populations], [N])
    qtraj_true = KetTrajectory(sys_true, pulse, ψ_init, ψ_goal)
    experiment = SimulatedExperiment(qtraj_true, model)

    ptp = PulseTuningProblem(qcp, experiment, model; R_tr = (u = 1e-2,))

    @test ptp.qcp === qcp
    @test ptp.experiment === experiment
    @test ptp.R_tr == (u = 1e-2,)
    @test ptp.Q_meas == 1.0
    @test isnothing(ptp.result)
    # Default strategy is the no-op IdentityStrategy until a concrete strategy
    # is provided via `strategy=`.
    @test ptp.strategy isa IdentityStrategy
end
