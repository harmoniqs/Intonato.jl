# Tests for the IterateSelector seam, the extended IterationRecord, the
# adapt!-before-step ordering, the solve! logger seam, and the
# measurement-dimension-change reset.

@testitem "selectors: NoiseCorrectedBestJ picks argmin J̃ over accepted iterates" begin
    using Intonato
    using Intonato: select_iterate!, NoiseCorrectedBestJ

    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    sys = QuantumSystem(0.01 * σz, [σx], [1.0])
    N = 11
    times = collect(range(0.0, 5.0, length = N))
    pulse = LinearSplinePulse(0.1 * ones(1, N), times)
    ψ0 = ComplexF64[1.0, 0.0]
    ψg = ComplexF64[0.0, 1.0]
    qcp = SplinePulseProblem(KetTrajectory(sys, pulse, ψ0, ψg), N; Q = 100.0, R = 1e-2)
    z_ref = qcp.prob.trajectory
    model = MeasurementModel(:ψ̃, [populations], [N])

    θ_of(x) = (data = fill(x, size(z_ref.data)), global_data = nothing)
    timing = (experiment = 0.0, sysid = 0.0, nlp = 0.0, armijo = 0.0, total = 0.0)
    mkrec(; J_hat, J_tilde, accepted, θ) = IterationRecord(
        Measurement[],
        J_tilde,
        J_hat,
        1.0,
        1.0,
        accepted,
        nothing,
        pulse,
        θ,
        timing,
    )

    # Heteroscedastic floors: raw Ĵ would pick B (0.9 < 1.0), the debiased J̃
    # picks A (0.2 < 0.8). A rejected record with the smallest J̃ of all must
    # be ignored.
    recA = mkrec(J_hat = 1.0, J_tilde = 0.2, accepted = true, θ = θ_of(1.0))
    recB = mkrec(J_hat = 0.9, J_tilde = 0.8, accepted = true, θ = θ_of(2.0))
    recC = mkrec(J_hat = 0.05, J_tilde = 0.05, accepted = false, θ = θ_of(3.0))
    history = [recB, recA, recC]

    ctx = (;
        experiment = nothing,
        measurement_model = model,
        y_goal = Measurement[],
        verbose = false,
    )
    n_evals = select_iterate!(NoiseCorrectedBestJ(), z_ref, history, ctx)
    @test n_evals == 0
    @test all(z_ref.data .== 1.0)     # A's θ restored — not B's, not rejected C's
end

@testitem "selectors: TopKRemeasure re-measures finalists at boosted shots" begin
    using Intonato
    using Intonato: select_iterate!, TopKRemeasure

    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    sys = QuantumSystem(0.01 * σz, [σx], [1.0])
    N = 11
    times = collect(range(0.0, 5.0, length = N))
    base_pulse = LinearSplinePulse(0.1 * ones(1, N), times)
    ψ0 = ComplexF64[1.0, 0.0]
    ψg = ComplexF64[0.0, 1.0]
    qcp =
        SplinePulseProblem(KetTrajectory(sys, base_pulse, ψ0, ψg), N; Q = 100.0, R = 1e-2)
    z_ref = qcp.prob.trajectory

    # Declared model: one shot-noise measurement at n=400 → boost = 400·4.
    m_shot = ShotNoiseMeasurement(populations, 400, wigner_covariance)
    model = MeasurementModel(:ψ̃, AbstractMeasurement[m_shot], [N])
    y_goal = [Measurement([0.0, 1.0], N)]

    # Spy experiment: records the n_shots override and returns a "true"
    # quality keyed on the pulse's first knot amplitude — pulse #2 is the TRUE
    # best even though its recorded J̃ ranks second.
    shot_calls = Int[]
    pulses = [LinearSplinePulse(a * ones(1, N), times) for a in (0.10, 0.20, 0.30, 0.40)]
    true_p1 = Dict(0.10 => 0.8, 0.20 => 0.99, 0.30 => 0.9, 0.40 => 0.7)
    function spy_run(p; n_shots = nothing)
        push!(shot_calls, something(n_shots, -1))
        p1 = true_p1[p(0.0)[1]]
        return [Measurement([1 - p1, p1], N)]
    end
    experiment = HardwareExperiment(spy_run, model)

    θ_of(x) = (data = fill(x, size(z_ref.data)), global_data = nothing)
    timing = (experiment = 0.0, sysid = 0.0, nlp = 0.0, armijo = 0.0, total = 0.0)
    mkrec(J̃, p, θ; accepted = true) =
        IterationRecord(Measurement[], J̃, J̃, 1.0, 1.0, accepted, nothing, p, θ, timing)

    # Recorded (noisy) J̃ ranking: p1 < p2 < p3 < p4; truth says p2 is best.
    history = [
        mkrec(0.10, pulses[1], θ_of(1.0)),
        mkrec(0.20, pulses[2], θ_of(2.0)),
        mkrec(0.30, pulses[3], θ_of(3.0)),
        mkrec(0.40, pulses[4], θ_of(4.0)),
    ]
    ctx = (; experiment, measurement_model = model, y_goal, verbose = false)

    n_evals = select_iterate!(TopKRemeasure(k = 3, reps_factor = 4), z_ref, history, ctx)
    @test n_evals == 3                       # top-3 by recorded J̃ re-measured
    @test shot_calls == [1600, 1600, 1600]   # n_shots = 400 · 4
    @test all(z_ref.data .== 2.0)            # true winner (p2) restored

    # Fewer than k accepted ⇒ re-measure all accepted.
    empty!(shot_calls)
    short_history = [
        mkrec(0.10, pulses[1], θ_of(1.0)),
        mkrec(0.20, pulses[2], θ_of(2.0)),
        mkrec(0.05, pulses[3], θ_of(3.0); accepted = false),
    ]
    n_evals2 =
        select_iterate!(TopKRemeasure(k = 3, reps_factor = 4), z_ref, short_history, ctx)
    @test n_evals2 == 2
    @test all(z_ref.data .== 2.0)
end

@testitem "selectors: PolyakAverage reproduces the legacy polyak_avg behavior" begin
    using Intonato
    using Intonato: AbstractTuningStrategy, PolyakAverage

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

    mutable struct DampingStrategy <: AbstractTuningStrategy
        cand::Any
    end
    DampingStrategy() = DampingStrategy(nothing)
    function Intonato.step(s::DampingStrategy, ctx)
        cand = deepcopy(ctx.z_ref)
        cand.u .= 0.5 .* cand.u
        s.cand = cand
        return extract_pulse(ctx.qcp.qtraj, cand)
    end
    Intonato.candidate_trajectory(s::DampingStrategy) = s.cand

    function run(; kwargs...)
        qcp = SplinePulseProblem(
            KetTrajectory(sys_nom, pulse, ψ_init, ψ_goal),
            N;
            Q = 100.0,
            R = 1e-2,
        )
        experiment =
            SimulatedExperiment(KetTrajectory(sys_rabi, pulse, ψ_init, ψ_goal), model)
        ptp_kwargs = (; strategy = DampingStrategy())
        if haskey(kwargs, :selector)
            ptp_kwargs = merge(ptp_kwargs, (; selector = kwargs[:selector]))
        end
        ptp = PulseTuningProblem(qcp, experiment, model; ptp_kwargs...)
        solve_kwargs = haskey(kwargs, :polyak_avg) ? (; polyak_avg = kwargs[:polyak_avg]) : (;)
        solve!(
            ptp;
            max_iter = 3,
            line_search = false,
            verbose = false,
            min_nominal_fidelity = 0.0,
            tol = 0.0,
            solve_kwargs...,
        )
        return sum(qcp.prob.trajectory.u)
    end

    # Legacy value recorded on the pre-seam chassis (record_polyak.jl):
    # average of the last-2 post-step iterates = (u0/4 + u0/8)/2 ⇒ Σu = 0.20625.
    legacy = 0.20625

    # The selector path.
    @test run(selector = PolyakAverage(2)) ≈ legacy rtol = 1e-12

    # The deprecated kwarg forwards to the selector (with a deprecation note).
    @test run(polyak_avg = 2) ≈ legacy rtol = 1e-12
end

@testitem "chassis: adapt! runs before the strategy step each iteration" begin
    using Intonato
    using Intonato: AbstractTuningStrategy, AbstractDeviceModel

    events = Symbol[]

    struct SpyModel <: AbstractDeviceModel
        events::Vector{Symbol}
    end
    Intonato.adapt!(m::SpyModel, pulse, y_exp) = (push!(m.events, :adapt); m)

    struct SpyStrategy <: AbstractTuningStrategy
        events::Vector{Symbol}
    end
    Intonato.step(s::SpyStrategy, ctx) = (push!(s.events, :step); ctx.pulse)

    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    sys = QuantumSystem(0.01 * σz, [σx], [1.0])
    sys_rabi = QuantumSystem(0.01 * σz, [1.15 * σx], [1.0])   # J > 0 ⇒ no early break
    N = 11
    times = collect(range(0.0, 5.0, length = N))
    pulse = LinearSplinePulse(0.1 * ones(1, N), times)
    ψ0 = ComplexF64[1.0, 0.0]
    ψg = ComplexF64[0.0, 1.0]
    qcp = SplinePulseProblem(KetTrajectory(sys, pulse, ψ0, ψg), N; Q = 100.0, R = 1e-2)
    model = MeasurementModel(:ψ̃, [populations], [N])
    experiment = SimulatedExperiment(KetTrajectory(sys_rabi, pulse, ψ0, ψg), model)

    ptp = PulseTuningProblem(
        qcp,
        experiment,
        model;
        strategy = SpyStrategy(events),
        device_model = SpyModel(events),
    )
    solve!(
        ptp;
        max_iter = 2,
        line_search = false,
        verbose = false,
        min_nominal_fidelity = 0.0,
        tol = 0.0,
    )

    # Recalibration must precede Jacobian assembly: adapt! BEFORE step, every
    # iteration (this inverts the pre-Task-7 order).
    @test events == [:adapt, :step, :adapt, :step]
end

@testitem "chassis: extended IterationRecord + solve! logger seam" begin
    using Intonato
    using Intonato: AbstractTuningStrategy, InMemoryExperimentLogger

    mutable struct DampingStrategyF <: AbstractTuningStrategy
        cand::Any
    end
    DampingStrategyF() = DampingStrategyF(nothing)
    function Intonato.step(s::DampingStrategyF, ctx)
        cand = deepcopy(ctx.z_ref)
        cand.u .= 0.5 .* cand.u
        s.cand = cand
        return extract_pulse(ctx.qcp.qtraj, cand)
    end
    Intonato.candidate_trajectory(s::DampingStrategyF) = s.cand
    Intonato.last_f_model(s::DampingStrategyF) = 0.75

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
    u0 = copy(qcp.prob.trajectory.u)
    model = MeasurementModel(:ψ̃, [populations], [N])
    experiment = SimulatedExperiment(KetTrajectory(sys_rabi, pulse, ψ0, ψg), model)

    lg = InMemoryExperimentLogger()
    ptp = PulseTuningProblem(qcp, experiment, model; strategy = DampingStrategyF())
    solve!(
        ptp;
        max_iter = 2,
        line_search = false,
        verbose = false,
        min_nominal_fidelity = 0.0,
        tol = 0.0,
        logger = lg,
    )

    h = ptp.result.history
    @test length(h) == 2
    # The chassis drove the logger: one record! per iteration, same records.
    @test length(lg.iteration_records) == 2
    @test lg.iteration_records[1] === h[1]

    # Extended fields: accepted flag, raw Ĵ, F_model hook value, θ snapshot.
    @test h[1].accepted && h[2].accepted
    @test h[1].J_hat ≈ h[1].J_exp   # deterministic model: Ĵ = J̃
    @test h[1].F_model == 0.75
    # θ is the measurement-time trajectory snapshot: restoring it into the
    # trajectory recovers the measured iterate — iter 1 measured u0, iter 2
    # measured the damped iterate (u halved by iter 1's full-step application).
    @test h[2].θ.data ≠ h[1].θ.data
    qcp.prob.trajectory.data .= h[1].θ.data
    @test qcp.prob.trajectory.u ≈ u0 atol = 1e-12
    qcp.prob.trajectory.data .= h[2].θ.data
    @test qcp.prob.trajectory.u ≈ 0.5 .* u0 atol = 1e-12
end

@testitem "chassis: measurement-dimension change restricts goal + resets acceptance" begin
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

    # Declared model: parity-like element + σ_z-like element; the "lab" stops
    # returning the second element after the first call (the σ_z fallback).
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

    ptp = PulseTuningProblem(
        qcp,
        experiment,
        model;
        acceptance = OneShotAcceptance(β = 0.5),
        y_goal = y_goal,
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
    @test length(h[1].y_exp) == 2
    @test length(h[2].y_exp) == 1
    # The restricted-goal J is finite and computed against y_goal[1:1]:
    # (1.0 − 0.4)² = 0.36.
    @test h[2].J_exp ≈ 0.36
    # Acceptance base was reset — the shrunken-dimension trial re-establishes
    # the base rather than triggering the catastrophe cascade.
    @test h[2].accepted
    @test h[2].tr_scale == h[1].tr_scale
end
