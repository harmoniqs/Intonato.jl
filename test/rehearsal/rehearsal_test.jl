# M3b — the QILC-through-twin rehearsal gate (#37).
#
# The loop's integration evidence, per the issue's Acceptance Criteria:
#   AC1  baseline — a QILC phase (PulseTuningProblem + StrumentoBackend over TwinSoc
#        + the bosonic family) converges on STATIC truth above a stated gate
#   AC2  drift injected and TRACKED — the adapting loop recovers fidelity while a
#        no-adaptation control (same seed, same drift path, model frozen) sags
#   AC3  belief write-back — the loop's estimates land via calibrate!; believed
#        tracks truth within the demonstrated estimation error; drift moves truth
#        only, calibration moves belief only (asserted live)
#   AC4  seeded replay — the whole rehearsal reproduces bit-exactly under its seed
#        across fresh processes (twin rng + QCP determinism both pinned)
#   AC5  no cross-environment float goldens — every committed gate is a
#        semantic-scale statement; bit-exact `==` only within a run
#
# The harness (test/rehearsal/rehearsal_harness.jl) is test-side code: the loop
# chassis consumes public surface only (PulseTuningProblem, the backend adapter,
# the experiment factory, the Fisher/IRLS weighting) over Strumento 0.3's public
# twin face (TwinSoc, instantiate/advance!/calibrate!/believed, DriftPlan). Zero
# Strumento-side edits; the bosonic family builder is a caller-provided TwinSoc
# family (the seam's design).

@testitem "M3b rehearsal harness: fixture, family physics, Ramsey prep" tags = [:m3b] begin
    using Intonato
    import Strumento
    using LinearAlgebra
    include(joinpath(@__DIR__, "rehearsal_harness.jl"))

    # the committed fixture loads as a bosonic twin record with the documented shape
    record = Strumento.load_record(rehearsal_fixture_path())
    @test record.family == "bosonic"
    @test record.id == "synthetic-bosonic-rehearsal"
    @test record.parameters["chi_kHz"] == -300.0
    @test record.drift_priors["chi_kHz"]["process"] == ["ou", "ramp"]

    # family: the record's truth builds the dispersive cavity⊗transmon system —
    # closed-form check at the fixed diagonal structure (rotating frame)
    truth = rehearsal_truth(record)
    sys = rehearsal_family(truth)
    @test sys.levels == Int(truth[:N_transmon]) * Int(truth[:N_fock])
    @test sys.n_drives == 4                       # transmon I/Q + cavity I/Q
    χ = 2π * truth[:chi_kHz] * 1e-6
    H = Matrix(sys.H_drift)
    @test norm(H - diagm(diag(H))) < 1e-12        # diagonal in the rotating frame
    idx(n, m) = (n - 1) * 2 + m                   # cavity-major, transmon minor
    @test H[idx(2, 2), idx(2, 2)] ≈ χ             # cavity 1-photon, ancilla e: the χ pull

    # the family is a function of CURRENT truth (drift is felt, not cached)
    truth2 = copy(truth); truth2[:chi_kHz] += 1000.0
    @test Matrix(rehearsal_family(truth2).H_drift)[idx(2, 2), idx(2, 2)] ≈ χ + 2π * 1e-3

    # the ancilla marginal: transmon populations, cavity traced out, sums to 1
    # (spread over the joint space: |e,0⟩ ⊕ |g,3⟩ — fock 4 is the cutoff's top)
    ψ = zeros(ComplexF64, 8); ψ[idx(1, 2)] = 0.8; ψ[idx(4, 1)] = 0.6im
    m = rehearsal_ancilla_marginal(ket_to_iso(ψ))
    @test m ≈ [0.6^2, 0.8^2] atol = 1e-12

    # the structured Ramsey prep: converges on static truth at the believed
    # parameters (the QCP polish above the analytic construction)
    pulse, times, ψt = structured_prep(truth)
    qcp = nominal_qcp(truth, pulse, ψt)
    solve!(qcp; max_iter = 12, verbose = false, print_level = 0)
    F = truth_fidelity(truth, qcp, times, ψt)
    @test F ≥ 0.99                                # the baseline gate (machinery: ≥0.998 demonstrated)
end

@testitem "M3b rehearsal: converge, drift, track, write back (adapted vs paired control)" tags = [:m3b] begin
    using Intonato
    import Strumento
    using LinearAlgebra
    include(joinpath(@__DIR__, "rehearsal_harness.jl"))

    summary = run_rehearsal(REHEARSAL_SEED)

    # ── AC1 — the baseline: QILC converges on STATIC truth above the stated gate ──
    @test summary.converged_baseline
    @test summary.F_baseline ≥ 0.99
    # the weighted residual at convergence is consistent with pure shot noise:
    # J is the Fisher-weighted SSR over 12 elements (6 knots × 2 marginal levels) —
    # its shot-noise floor is χ²(12) (mean 12); the gate sits well above it
    @test summary.J_baseline ≤ 40.0

    # ── AC2 — drift injected and TRACKED: adapted recovers, frozen control sags ──
    # the control (same seed, same drift path — the adapted run's recorded path is
    # replayed via JumpSchedule — model frozen) sags below the baseline by a
    # semantic margin (machinery: the drift response at the injected scale)
    @test summary.F_control[end] ≤ summary.F_baseline - 0.03
    # the adapting loop recovers: its final fidelity sits near the baseline…
    @test summary.F_adapted[end] ≥ 0.95
    @test summary.F_adapted[end] ≥ summary.F_baseline - 0.04
    # …and the paired delta is the evidence
    @test summary.F_adapted[end] - summary.F_control[end] ≥ 0.03

    # ── AC3 — belief write-back: the contract's invariants, asserted live ──
    # drift moves truth only: every advance! moved truth while belief held
    # (recorded live by the harness between advance! and calibrate!)
    @test all(summary.truth_moved_on_advance)
    @test all(.!summary.belief_moved_on_advance)
    # calibration moves belief only: every calibrate! moved belief, never truth
    @test all(summary.belief_moved_on_calibrate)
    @test all(.!summary.truth_moved_on_calibrate)
    # believed tracks the drifted truth within the loop's demonstrated estimation error
    @test summary.est_error_max ≤ 12.0    # kHz — ~3× the fit's demonstrated CRB scale
    for (b, t) in zip(summary.chi_believed, summary.chi_truth)
        @test abs(b - t) ≤ 1.5 * summary.est_error_max
    end
    # the write-back actually tracked: the final belief is closer to the final
    # truth than the frozen record value is
    @test abs(summary.chi_believed[end] - summary.chi_truth[end]) <
          abs(summary.chi_truth[1] - summary.chi_truth[end])
end

@testitem "M3b rehearsal: seeded replay is bit-exact (in-process + fresh child process)" tags = [:m3b] begin
    using Intonato
    import Strumento
    include(joinpath(@__DIR__, "rehearsal_harness.jl"))

    # ── AC4 — seeded replay: the whole rehearsal reproduces bit-exactly ──────
    # Three rehearsals per suite run: two in-process (fresh process OBJECTS,
    # same seed) and one FRESH JULIA PROCESS (same seed) — all three digests
    # must match bit-exactly (the twin's StableRNG and the QCP's deterministic
    # solve both pinned; the digest serializes every summary field as Float64
    # bit patterns — == at the bit level, no cross-environment float goldens).
    # Budget: the item's rehearsals run at M3B_REPLAY_EPOCHS (documented in the
    # harness) — every mechanism is live and pinned at that configuration.

    # fresh process OBJECTS, same seed — identical summaries (within-process ==)
    s1 = run_rehearsal(REHEARSAL_SEED; n_epochs = M3B_REPLAY_EPOCHS)
    s2 = run_rehearsal(REHEARSAL_SEED; n_epochs = M3B_REPLAY_EPOCHS)
    @test rehearsal_digest(s1) == rehearsal_digest(s2)

    # a FRESH JULIA PROCESS, same seed — the digest must match bit-exactly
    dir = mktempdir()
    driver = joinpath(dir, "replay_child.jl")
    write(driver, """
        using Pkg
        Pkg.activate($(repr(dirname(Base.active_project()))))
        using Intonato
        import Strumento
        include($(repr(joinpath(@__DIR__, "rehearsal_harness.jl"))))
        s = run_rehearsal($REHEARSAL_SEED; n_epochs = $M3B_REPLAY_EPOCHS)
        println("M3B_DIGEST=", rehearsal_digest(s))
        """)
    proj = dirname(Base.active_project())
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$proj $driver`
    out = read(cmd, String)
    digests = [strip(split(line, "=")[2]) for line in split(out, "\n") if startswith(line, "M3B_DIGEST=")]
    @test length(digests) == 1
    @test digests[1] == rehearsal_digest(s1)
end
