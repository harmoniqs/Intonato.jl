# Tests for the acceptance-policy seam.
#
# The "reproduces the pre-seam chassis traces" item is a behavior-preservation
# regression: the (α, tr_scale, n_experiments, J) traces below were recorded
# on the chassis BEFORE the extraction (commit 3d71273, scratchpad
# record_trace.jl) and hard-coded here. The extraction must not change them.

@testitem "acceptance seam: LineSearchAcceptance reproduces pre-seam chassis traces" begin
    using Intonato
    using Intonato: AbstractTuningStrategy

    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    sys_nom = QuantumSystem(0.01 * σz, [σx], [1.0])
    sys_true = QuantumSystem(0.0105 * σz, [σx], [1.0])
    # 15% Rabi error: large enough that armijo finds a partial-α improvement
    # on a drive-damping candidate (exercises the accept/interpolate path).
    sys_rabi = QuantumSystem(0.01 * σz, [1.15 * σx], [1.0])

    N = 11
    times = collect(range(0.0, 5.0, length = N))
    pulse = LinearSplinePulse(0.1 * ones(1, N), times)
    ψ_init = ComplexF64[1.0, 0.0]
    ψ_goal = ComplexF64[0.0, 1.0]
    model = MeasurementModel(:ψ̃, [populations], [N])

    build_qcp() = SplinePulseProblem(
        KetTrajectory(sys_nom, pulse, ψ_init, ψ_goal),
        N;
        Q = 100.0,
        R = 1e-2,
    )
    build_experiment(sys) =
        SimulatedExperiment(KetTrajectory(sys, pulse, ψ_init, ψ_goal), model)

    # Scripted strategy: candidate trajectory damps the drive by 0.5 each
    # step — deterministic, exercises the accept/interpolate path.
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

    function trace(strategy; max_iter = 4, sys = sys_true, kwargs...)
        qcp = build_qcp()
        ptp = PulseTuningProblem(
            qcp,
            build_experiment(sys),
            model;
            R_tr = (u = 1e-2,),
            strategy = strategy,
        )
        solve!(
            ptp;
            max_iter,
            verbose = false,
            min_nominal_fidelity = 0.0,
            tol = 0.0,
            kwargs...,
        )
        h = ptp.result.history
        return (
            J = [rec.J_exp for rec in h],
            α = [rec.step_size for rec in h],
            tr = [rec.tr_scale for rec in h],
            n_exp = ptp.result.n_experiments,
            u_final = copy(qcp.prob.trajectory.u),
        )
    end

    # Trace A: IdentityStrategy, line_search=true — pure reject path + 1/γ
    # schedule growth (armijo cannot improve on an identical candidate).
    tA = trace(IdentityStrategy())
    @test tA.α == [0.0, 0.0, 0.0, 0.0]
    @test tA.tr == [1.25, 1.5625, 1.953125, 2.44140625]
    @test tA.n_exp == 32
    @test all(isapprox.(tA.J, 7.963203088888532e-10; rtol = 1e-6))

    # Trace B: DampingStrategy vs 15% Rabi-error truth — armijo accepts α=0.5
    # on iter 1 (interpolation applied), then rejects.
    tB = trace(DampingStrategy(); sys = sys_rabi)
    @test tB.α == [0.5, 0.0, 0.0, 0.0]
    @test tB.tr == [0.8, 1.0, 1.25, 1.5625]
    @test tB.n_exp == 27
    @test tB.J[1] ≈ 0.00867249511710011 rtol = 1e-10
    @test tB.J[2] ≈ 0.006065830661113523 rtol = 1e-10
    @test sum(tB.u_final) ≈ 0.825 rtol = 1e-12

    # Trace C: line_search=false — α = 1 always, schedule skipped, tr_scale
    # constant, candidate applied fully each iteration.
    tC = trace(DampingStrategy(); line_search = false, max_iter = 3, sys = sys_rabi)
    @test tC.α == [1.0, 1.0, 1.0]
    @test tC.tr == [1.0, 1.0, 1.0]
    @test tC.n_exp == 3
    @test tC.J[1] ≈ 0.00867249511710011 rtol = 1e-10
    @test tC.J[2] ≈ 0.04459149484124723 rtol = 1e-10
    @test sum(tC.u_final) ≈ 0.13749999999999998 rtol = 1e-12

    # The same traces through an EXPLICIT policy object on the problem (the
    # ptp.acceptance slot), rather than the legacy solve! kwargs.
    function trace_policy(strategy, policy; max_iter = 4, sys = sys_true)
        qcp = build_qcp()
        ptp = PulseTuningProblem(
            qcp,
            build_experiment(sys),
            model;
            R_tr = (u = 1e-2,),
            strategy = strategy,
            acceptance = policy,
        )
        solve!(ptp; max_iter, verbose = false, min_nominal_fidelity = 0.0, tol = 0.0)
        h = ptp.result.history
        return (
            α = [rec.step_size for rec in h],
            tr = [rec.tr_scale for rec in h],
            n_exp = ptp.result.n_experiments,
        )
    end

    tA2 = trace_policy(IdentityStrategy(), LineSearchAcceptance())
    @test tA2.α == tA.α && tA2.tr == tA.tr && tA2.n_exp == tA.n_exp

    tC2 = trace_policy(
        DampingStrategy(),
        LineSearchAcceptance(line_search = false);
        max_iter = 3,
        sys = sys_rabi,
    )
    @test tC2.α == tC.α && tC2.tr == tC.tr && tC2.n_exp == tC.n_exp
end
