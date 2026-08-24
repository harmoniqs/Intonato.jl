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

    # R_tr is DEPRECATED (removed in v0.4.0): stored for one-release source
    # compatibility but never consumed by solve!, so a non-default value must warn.
    ptp = @test_logs (:warn, r"R_tr.*not consumed") match_mode = :any PulseTuningProblem(
        qcp,
        experiment,
        model;
        R_tr = (u = 1e-2,),
    )

    @test ptp.qcp === qcp
    @test ptp.experiment === experiment
    @test ptp.R_tr == (u = 1e-2,)   # still stored this release (compat)
    @test ptp.Q_meas == 1.0
    @test isnothing(ptp.result)
    # Default strategy is the no-op IdentityStrategy until a concrete strategy
    # is provided via `strategy=`.
    @test ptp.strategy isa IdentityStrategy
    # Default y_goal is nothing (resolved once from tuning_goal at solve start).
    @test isnothing(ptp.y_goal)
end

@testitem "PulseTuningProblem W_task flows into the whitened chassis cost" begin
    using Intonato
    using LinearAlgebra

    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    sys_nom = QuantumSystem(0.01 * σz, [σx], [1.0])
    sys_rabi = QuantumSystem(0.01 * σz, [1.15 * σx], [1.0])
    N = 11
    times = collect(range(0.0, 5.0, length = N))
    pulse = LinearSplinePulse(0.1 * ones(1, N), times)
    ψ0 = ComplexF64[1.0, 0.0]
    ψg = ComplexF64[0.0, 1.0]
    qcp = SplinePulseProblem(KetTrajectory(sys_nom, pulse, ψ0, ψg), N; Q = 100.0, R = 1e-2)
    model = MeasurementModel(:ψ̃, [populations], [N])
    experiment = SimulatedExperiment(KetTrajectory(sys_rabi, pulse, ψ0, ψg), model)

    # Deterministic model + task weights: J̃ = Σ (w_task ⊙ r)² exactly
    # (task importance is first-class, not noise statistics).
    W_task = [1.0, sqrt(40)]
    ptp = PulseTuningProblem(qcp, experiment, model; W_task = W_task)
    solve!(
        ptp;
        max_iter = 1,
        line_search = false,
        verbose = false,
        min_nominal_fidelity = 0.0,
        tol = 0.0,
    )
    y_goal = model_predict(qcp.prob.trajectory, model)
    y_exp = run_experiment(experiment, pulse)
    r = y_goal[1].data .- y_exp[1].data
    @test ptp.result.history[1].J_exp ≈ sum(abs2, W_task .* r)
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

@testitem "nominal-fidelity gate: pass, fail, and unfaithful-fidelity paths" begin
    using Intonato
    using Intonato: _check_nominal_fidelity
    using LinearAlgebra

    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    sys_nom = QuantumSystem(0.01 * σz, [σx], [1.0])
    N = 11
    T = 5.0
    times = range(0.0, T, length = N) |> collect
    pulse = LinearSplinePulse(0.1 * randn(1, N), times)
    qtraj = KetTrajectory(sys_nom, pulse, ComplexF64[1.0, 0.0], ComplexF64[0.0, 1.0])
    qcp = SplinePulseProblem(qtraj, N; Q = 100.0, R = 1e-2)

    # threshold ≤ 0 disables the gate outright
    @test _check_nominal_fidelity(qcp, 0.0; verbose = false, path_label = "test") === nothing

    # A qcp whose fidelity throws is skipped, not fatal (nothing → gate passes)
    struct _Unfaithful end
    @test _check_nominal_fidelity(
        _Unfaithful(),
        0.9;
        verbose = false,
        path_label = "test",
    ) === nothing

    # Real qcp with an unsolved (random) pulse: fidelity is low → gate errors
    F = try
        Piccolo.fidelity(qcp)
    catch
        nothing
    end
    if !isnothing(F) && F < 0.5
        err = try
            _check_nominal_fidelity(qcp, 0.5; verbose = false, path_label = "test")
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("min_nominal_fidelity", err.msg)
    end

    # Fidelity above threshold passes (and logs when verbose)
    solve!(qcp; max_iter = 200, print_level = 0)
    F = Piccolo.fidelity(qcp)
    if !isnothing(F) && F ≥ 0.5
        @test_logs (:info, r"nominal fidelity check passed") match_mode = :any _check_nominal_fidelity(
            qcp,
            0.5;
            verbose = true,
            path_label = "test",
        )
    end
end

@testitem "PulseTuningProblem Q_meas deprecation warning fires" begin
    using Intonato

    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    sys_nom = QuantumSystem(0.01 * σz, [σx], [1.0])
    sys_true = QuantumSystem(0.0105 * σz, [σx], [1.0])
    N = 11
    times = range(0.0, 5.0, length = N) |> collect
    pulse = LinearSplinePulse(0.1 * randn(1, N), times)
    qtraj_nom = KetTrajectory(sys_nom, pulse, ComplexF64[1.0, 0.0], ComplexF64[0.0, 1.0])
    qcp = SplinePulseProblem(qtraj_nom, N; Q = 100.0, R = 1e-2)
    model = MeasurementModel(:ψ̃, [populations], [N])
    experiment = SimulatedExperiment(
        KetTrajectory(sys_true, pulse, ComplexF64[1.0, 0.0], ComplexF64[0.0, 1.0]),
        model,
    )

    ptp = @test_logs (:warn, r"Q_meas.*not consumed") match_mode = :any PulseTuningProblem(
        qcp,
        experiment,
        model;
        Q_meas = 2.0,
    )
    @test ptp.Q_meas == 2.0
end

@testitem "solve!: max_rejections early stop fires and is visible in history" begin
    using Intonato
    using Intonato: AbstractTuningStrategy

    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    sys_nom = QuantumSystem(0.01 * σz, [σx], [1.0])
    sys_rabi = QuantumSystem(0.01 * σz, [1.15 * σx], [1.0])
    N = 11
    times = collect(range(0.0, 5.0, length = N))
    pulse = LinearSplinePulse(0.1 * ones(1, N), times)
    ψ_init = ComplexF64[1.0, 0.0]
    ψ_goal = ComplexF64[0.0, 1.0]
    model = MeasurementModel(:ψ̃, [populations], [N])

    # Every step catastrophically rejects → consecutive rejections accumulate.
    mutable struct ExplodingStrategy <: AbstractTuningStrategy
        cand::Any
    end
    ExplodingStrategy() = ExplodingStrategy(nothing)
    function Intonato.step(s::ExplodingStrategy, ctx)
        cand = deepcopy(ctx.z_ref)
        cand.u .= 4.0 .* cand.u
        s.cand = cand
        return extract_pulse(ctx.qcp.qtraj, cand)
    end
    Intonato.candidate_trajectory(s::ExplodingStrategy) = s.cand

    qcp = SplinePulseProblem(
        KetTrajectory(sys_nom, pulse, ψ_init, ψ_goal),
        N;
        Q = 100.0,
        R = 1e-2,
    )
    experiment = SimulatedExperiment(KetTrajectory(sys_rabi, pulse, ψ_init, ψ_goal), model)
    ptp = PulseTuningProblem(
        qcp,
        experiment,
        model;
        strategy = ExplodingStrategy(),
    )
    solve!(
        ptp;
        max_iter = 10,
        max_rejections = 2,
        verbose = false,
        min_nominal_fidelity = 0.0,
        tol = 0.0,
    )
    # The early stop must fire BEFORE max_iter: 10 allowed, stopped after 3
    # (armijo rejects the quadrupled drive every iteration).
    @test length(ptp.result.history) < 10
    @test ptp.result.history[end].accepted == false
end

@testitem "chassis: W_task is trimmed when the measurement dimension shrinks mid-campaign" begin
    using Intonato

    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    sys = QuantumSystem(0.01 * σz, [σx], [1.0])
    N = 11
    times = collect(range(0.0, 5.0, length = N))
    pulse = LinearSplinePulse(0.1 * ones(1, N), times)
    ψ0 = ComplexF64[1.0, 0.0]
    ψg = ComplexF64[0.0, 1.0]
    qcp = SplinePulseProblem(KetTrajectory(sys, pulse, ψ0, ψg), N; Q = 100.0, R = 1e-2)

    g1 = x -> [sum(abs2, x[1:2])]
    model = MeasurementModel(:ψ̃, [g1, g1], [N, N])
    y_goal = [Measurement([1.0], N), Measurement([0.9], N)]

    calls = Ref(0)
    function lab_run(p)
        calls[] += 1
        full = [Measurement([0.4], N), Measurement([0.5], N)]
        return calls[] == 1 ? full : full[1:1]
    end
    experiment = HardwareExperiment(lab_run, model)

    # W_task sized for the FULL (2-block) measurement vector: the shrink must
    # trim it to the flat length of the surviving measurements (1 scalar).
    ptp = PulseTuningProblem(
        qcp,
        experiment,
        model;
        acceptance = OneShotAcceptance(β = 0.5),
        y_goal = y_goal,
        W_task = [2.0, 3.0],
    )
    @test_logs (:warn, r"[Mm]easurement dimension") match_mode = :any solve!(
        ptp;
        max_iter = 3,
        verbose = false,
        min_nominal_fidelity = 0.0,
        tol = 0.0,
    )
    h = ptp.result.history
    @test length(h) == 3
    @test length(h[2].y_exp) == 1
    # The trimmed W_task keeps whitening consistent: no bounds/shape errors.
    @test isfinite(h[2].J_exp)
    @test h[2].accepted
end

@testitem "chassis: free-phase globals ride accept/revert/interpolate through solve!" begin
    using Intonato

    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    sys_nom = QuantumSystem(0.01 * σz, [σx], [1.0])
    sys_true = QuantumSystem(0.01 * σz, [1.3 * σx], [1.0])
    N = 11
    times = collect(range(0.0, 5.0, length = N))
    pulse = LinearSplinePulse(0.1 * randn(1, N), times)
    ψ0 = ComplexF64[1.0, 0.0]
    ψg = ComplexF64[0.0, 1.0]

    # free-phase adds a φ global carried by the trajectory (and z_ref)
    qtraj = KetTrajectory(sys_nom, pulse, ψ0, ψg)
    qcp = SplinePulseProblem(
        qtraj,
        N;
        Q = 100.0,
        R = 1e-2,
        free_phase = true,
        subsystem_levels = [2],
    )
    z0 = qcp.prob.trajectory
    if z0.global_dim > 0
        model = MeasurementModel(:ψ̃, [populations], [N])
        experiment = SimulatedExperiment(KetTrajectory(sys_true, pulse, ψ0, ψg), model)

        # An oscillating strategy: strong step (likely accepted) then a
        # catastrophic one (rejected → revert path restores accepted globals).
        mutable struct OscillatingStrategy <: Intonato.AbstractTuningStrategy
            cand::Any
            i::Int
        end
        OscillatingStrategy() = OscillatingStrategy(nothing, 0)
        function Intonato.step(s::OscillatingStrategy, ctx)
            s.i += 1
            cand = deepcopy(ctx.z_ref)
            if isodd(s.i)
                cand.u .= 0.8 .* cand.u     # damping: the classic improving candidate
            else
                cand.u .= 4.0 .* cand.u     # catastrophe → reject + revert
            end
            if cand.global_dim > 0
                cand.global_data .+= 0.05   # candidate proposes a global shift
            end
            s.cand = cand
            return extract_pulse(ctx.qcp.qtraj, cand)
        end
        Intonato.candidate_trajectory(s::OscillatingStrategy) = s.cand
        # This strategy co-optimizes globals with the controls → the chassis
        # interpolates candidate globals alongside the data (line 569).
        Intonato.accepts_global_data(::OscillatingStrategy) = true

        # OneShotAcceptance: rejected trials REVERT z_ref to the last accepted
        # iterate — the branch that restores z_accepted_global (line 548).
        ptp = PulseTuningProblem(
            qcp,
            experiment,
            model;
            strategy = OscillatingStrategy(),
            acceptance = OneShotAcceptance(β = 0.5),
        )
        solve!(
            ptp;
            max_iter = 6,
            verbose = false,
            min_nominal_fidelity = 0.0,
            tol = 0.0,
        )
        h = ptp.result.history
        @test length(h) == 6
        # Both acceptance outcomes occurred: the revert and accept branches of
        # the global-data plumbing both ran.
        @test any(rec.accepted for rec in h)
        @test any(!rec.accepted for rec in h)
        # The solve never corrupted the trajectory's global slot.
        @test size(qcp.prob.trajectory.global_data) == size(z0.global_data)
    else
        @test_broken false   # free-phase globals unavailable in this Piccolo version
    end
end
