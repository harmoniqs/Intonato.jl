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

@testitem "OneShotAcceptance decide semantics (unit)" begin
    using Intonato
    using Intonato: decide, reset_acceptance!, cost_std, diff_std

    # Minimal ctx builder — OneShotAcceptance never touches the experiment.
    mkctx(; J_hat, J_tilde = J_hat, r = Float64[], w = Float64[], σ2 = Float64[], tr = 1.0) = (;
        experiment = nothing,
        pulse = nothing,
        pulse_cand = nothing,
        y_goal = Measurement[],
        J_hat,
        J_tilde,
        r,
        w,
        σ2,
        tr_scale = tr,
        iter = 1,
        verbose = false,
    )

    # ── First measured trial always establishes the base (no rejection). ──
    p = OneShotAcceptance(β = 0.5, ρ_rej = 2.0, k = 3.0)
    reset_acceptance!(p)
    d1 = decide(p, mkctx(J_hat = 1.0e6))    # huge J: still no rejection possible
    @test d1.accepted && !d1.revert && d1.α == 0.5 && d1.tr_scale == 1.0

    # ── All-deterministic (σ_Δ = 0) ⇒ pure ratio test. ──
    reset_acceptance!(p)
    decide(p, mkctx(J_hat = 1.0))                       # base = 1.0
    d_ok = decide(p, mkctx(J_hat = 2.9))                # 1.9 < ρ_rej·J_base = 2.0
    @test d_ok.accepted && d_ok.α == 0.5
    # base is now 2.9; a trial at 9.0 exceeds 2.9 + 2·2.9 = 8.7 → reject.
    d_rej = decide(p, mkctx(J_hat = 9.0, tr = 1.0))
    @test !d_rej.accepted && d_rej.revert && d_rej.α == 0.0
    @test d_rej.tr_scale == 0.5                          # halved, permanently
    # Rejection does NOT update the base: same trial again still rejects.
    d_rej2 = decide(p, mkctx(J_hat = 9.0, tr = 0.5))
    @test !d_rej2.accepted && d_rej2.tr_scale == 0.25

    # ── Noise-tolerated case: the old raw-ratio test would reject, 3σ_Δ
    # correctly tolerates. ──
    q = OneShotAcceptance(β = 0.5, ρ_rej = 2.0, k = 3.0)
    reset_acceptance!(q)
    w4 = ones(4)
    σ2_4 = fill(0.01, 4)
    r_base = fill(0.05, 4)                               # Ĵ_base = 0.01
    J_base = sum(abs2, w4 .* r_base)
    decide(q, mkctx(J_hat = J_base, r = r_base, w = w4, σ2 = σ2_4))
    r_trial = fill(sqrt(0.05 / 4), 4)                     # Ĵ_trial = 0.05
    J_trial = sum(abs2, w4 .* r_trial)
    # Old raw-ratio: J_trial > ρ_rej·J_base (0.05 > 0.02) ⇒ would reject.
    @test J_trial > 2.0 * J_base
    # New: ΔJ = 0.04 is inside 3σ_Δ ⇒ tolerated.
    σΔ = diff_std(
        cost_std(r_trial, w4, σ2_4),
        cost_std(r_base, w4, σ2_4),
    )
    @test J_trial - J_base < 3σΔ
    d_tol = decide(q, mkctx(J_hat = J_trial, r = r_trial, w = w4, σ2 = σ2_4))
    @test d_tol.accepted && !d_tol.revert

    # ── tr_scale is shrink-only and floored at tr_scale_min. ──
    pf = OneShotAcceptance(β = 0.5, ρ_rej = 2.0, k = 3.0, tr_scale_min = 0.25)
    reset_acceptance!(pf)
    decide(pf, mkctx(J_hat = 1.0))
    df = decide(pf, mkctx(J_hat = 9.0, tr = 0.4))
    @test df.tr_scale == 0.25                            # max(0.2, floor)
    # Across any sequence, decide never returns tr_scale > ctx.tr_scale.
    ps = OneShotAcceptance(β = 0.5)
    reset_acceptance!(ps)
    let trs = 1.0
        for J in (1.0, 0.9, 5.0, 0.9, 0.5)
            d = decide(ps, mkctx(J_hat = J, tr = trs))
            @test d.tr_scale ≤ trs
            trs = d.tr_scale
        end
    end

    # ── Stall: after stall_patience flat accepted trials, halve. ──
    st = OneShotAcceptance(β = 0.5, ρ_rej = 2.0, k = 3.0, stall_patience = 2)
    reset_acceptance!(st)
    decide(st, mkctx(J_hat = 1.0))                       # base
    d_s1 = decide(st, mkctx(J_hat = 0.999))              # flat #1 (imp < 0.01·J̃)
    @test d_s1.accepted && d_s1.tr_scale == 1.0
    d_s2 = decide(st, mkctx(J_hat = 0.998))              # flat #2 → halve
    @test d_s2.accepted && d_s2.tr_scale == 0.5
    d_s3 = decide(st, mkctx(J_hat = 0.997, tr = 0.5))    # counter reset → no halve
    @test d_s3.tr_scale == 0.5
    # A real improvement resets the stall counter.
    d_s4 = decide(st, mkctx(J_hat = 0.5, tr = 0.5))
    d_s5 = decide(st, mkctx(J_hat = 0.499, tr = 0.5))    # flat #1 again (not #2)
    @test d_s4.tr_scale == 0.5 && d_s5.tr_scale == 0.5
end

@testitem "OneShotAcceptance chassis integration: β-apply, revert, one experiment per iteration" begin
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

    # Candidate quadruples the drive — a deterministic catastrophe the
    # rejection cascade must revert.
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
    u0 = copy(qcp.prob.trajectory.u)
    experiment =
        SimulatedExperiment(KetTrajectory(sys_rabi, pulse, ψ_init, ψ_goal), model)

    ptp = PulseTuningProblem(
        qcp,
        experiment,
        model;
        strategy = ExplodingStrategy(),
        acceptance = OneShotAcceptance(β = 1.0, ρ_rej = 2.0, k = 3.0),
    )
    solve!(ptp; max_iter = 2, verbose = false, min_nominal_fidelity = 0.0, tol = 0.0)

    h = ptp.result.history
    # Exactly one experiment per iteration — no probing, ever.
    @test ptp.result.n_experiments == 2
    # α is exactly β on the accepted first trial; 0 on the rejected one.
    @test h[1].step_size == 1.0
    @test h[2].step_size == 0.0
    # tr_scale halved permanently on the rejection.
    @test h[1].tr_scale == 1.0
    @test h[2].tr_scale == 0.5
    # The rejected candidate was reverted: the trajectory is back at u0.
    @test qcp.prob.trajectory.u ≈ u0 atol = 1e-12
end
