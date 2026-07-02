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
    # Default y_goal is nothing (resolved once from tuning_goal at solve start).
    @test isnothing(ptp.y_goal)
end

@testitem "PulseTuningProblem explicit y_goal kwarg (fixed goal invariant)" begin
    using Intonato
    using Intonato: AbstractTuningStrategy
    using LinearAlgebra

    # Stub strategy that counts tuning_goal calls — the chassis must resolve
    # the goal exactly once per solve (the chained-loop-drift invariant), and
    # not at all when an explicit y_goal is supplied.
    mutable struct GoalCountingStrategy <: AbstractTuningStrategy
        goal_calls::Int
    end
    GoalCountingStrategy() = GoalCountingStrategy(0)
    function Intonato.tuning_goal(s::GoalCountingStrategy, ptp, z_ref)
        s.goal_calls += 1
        return model_predict(z_ref, ptp.measurement_model)
    end
    Intonato.step(::GoalCountingStrategy, ctx) = ctx.pulse

    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    sys_nom = QuantumSystem(0.01 * σz, [σx], [1.0])
    sys_true = QuantumSystem(0.0105 * σz, [σx], [1.0])

    N = 11
    times = collect(range(0.0, 5.0, length = N))
    pulse = LinearSplinePulse(0.1 * ones(1, N), times)
    ψ_init = ComplexF64[1.0, 0.0]
    ψ_goal = ComplexF64[0.0, 1.0]
    qtraj_nom = KetTrajectory(sys_nom, pulse, ψ_init, ψ_goal)
    qcp = SplinePulseProblem(qtraj_nom, N; Q = 100.0, R = 1e-2)
    model = MeasurementModel(:ψ̃, [populations], [N])
    qtraj_true = KetTrajectory(sys_true, pulse, ψ_init, ψ_goal)
    experiment = SimulatedExperiment(qtraj_true, model)

    # Default (y_goal = nothing): tuning_goal resolved exactly ONCE for the
    # whole solve, and J_exp is measured against that fixed goal.
    strat = GoalCountingStrategy()
    ptp = PulseTuningProblem(qcp, experiment, model; strategy = strat)
    solve!(
        ptp;
        max_iter = 3,
        line_search = false,
        verbose = false,
        min_nominal_fidelity = 0.0,
    )
    @test strat.goal_calls == 1
    y_goal_default = model_predict(qcp.prob.trajectory, model)
    y_exp = run_experiment(experiment, extract_pulse(qcp.qtraj, qcp.prob.trajectory))
    @test ptp.result.history[end].J_exp ≈ measurement_error(y_exp, y_goal_default)

    # Explicit y_goal: used verbatim, tuning_goal never called.
    ys = [Measurement([0.0, 1.0], N)]
    strat2 = GoalCountingStrategy()
    ptp2 = PulseTuningProblem(qcp, experiment, model; strategy = strat2, y_goal = ys)
    @test ptp2.y_goal === ys
    solve!(
        ptp2;
        max_iter = 2,
        line_search = false,
        verbose = false,
        min_nominal_fidelity = 0.0,
    )
    @test strat2.goal_calls == 0
    @test ptp2.result.history[end].J_exp ≈ measurement_error(y_exp, ys)
end
