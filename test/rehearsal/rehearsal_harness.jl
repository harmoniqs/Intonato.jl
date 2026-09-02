# ─────────────────────────────────────────────────────────────────────────────
# M3b rehearsal harness (#37) — TEST-SIDE code, included by each rehearsal
# testitem. The loop chassis consumes public surface only:
#
#   Intonato:  PulseTuningProblem + solve!, StrumentoBackend, StrumentoExperiment,
#              MeasurementModel, Measurement, binomial_fisher_weights, Learn-less
#              strategy seam (AbstractTuningStrategy / step / candidate_trajectory)
#   Strumento: DigitalTwin contract (instantiate / advance! / calibrate! /
#              believed), DriftPlan + OrnsteinUhlenbeck, TwinSoc (via the
#              extension-aware reach — the same mechanism Intonato uses for
#              MockSoc), QickChannelMap / QickGenChannel
#
# Zero Strumento-side edits. The bosonic family is a CALLER-PROVIDED TwinSoc
# family (the seam's design — Strumento's own suite passes test-side builders
# the same way): a closed `QuantumSystem` slice of the substrate's bosonic
# family (dispersive + Kerr, transmon-ancilla ⊗ cavity, 4 quadrature drives).
# The OpenQuantumSystem/decay slice rides Strumento main's post-0.3.0 dispatch
# and is out of scope for the released-0.3.0 rehearsal (documented on #37).
#
# REPLAY DISCIPLINE: the QCP's initial pulse is the ANALYTIC structured Ramsey
# construction — no randomness at all — so the QCP solve is deterministic
# (Ipopt, single-threaded BLAS below); the twin's StableRNG(seed) is the ONE
# stochastic source (binomial shot sampling; the drift draws join it in the
# full rehearsal). run_rehearsal(seed) reproduces bit-exactly across fresh
# processes; the digest serialization uses Float64 bit patterns (== at the bit
# level, no cross-environment float goldens — the campaign CI rule).
# ─────────────────────────────────────────────────────────────────────────────

using LinearAlgebra
import Piccolo
BLAS.set_num_threads(1)     # replay determinism: pin the BLAS thread count

# ── the committed fixture (loaded fresh per rehearsal — replay discipline) ────
rehearsal_fixture_path() =
    joinpath(@__DIR__, "..", "fixtures", "twins", "rehearsal-bosonic.md")

const M3B_N_TRANSMON = 2     # the rehearsal family's dims (the fixture's level counts)
const M3B_N_FOCK = 4

# ── the bosonic family (closed-system slice), a TwinSoc family builder ────────
# Cavity ⊗ transmon (cavity-major, transmon minor), rotating frame — the bare
# frequencies are absorbed, so the record's dispersive shift reads directly off
# the diagonal H_drift: the cavity transition shifts by χ per transmon
# excitation. Kerr in the ladder convention E_n = ωn + (α/2)n(n−1). Drive order:
# [transmon I, transmon Q, cavity I, cavity Q]. Units: rad·GHz ⇒ rollout time ns.
function rehearsal_family(truth::Dict{Symbol,Float64})
    χ = 2π * truth[:chi_kHz] * 1e-6          # kHz → rad/ns
    α_c = 2π * truth[:K_c_kHz] * 1e-6
    n_t = Int(truth[:N_transmon])
    n_f = Int(truth[:N_fock])
    a = kron(annihilate(n_f), Matrix{ComplexF64}(I, n_t, n_t))
    q = kron(Matrix{ComplexF64}(I, n_f, n_f), annihilate(n_t))
    na, nq = a'a, q'q
    H_drift = χ * na * nq + (α_c / 2) * (a'^2 * a^2)
    H_drives = [(q + q') / 2, im * (q' - q) / 2, (a + a') / 2, im * (a' - a) / 2]
    return QuantumSystem(H_drift, H_drives, [1.0, 1.0, 1.0, 1.0])
end

# the record's parameters → the family's truth dict (Real parameters only)
rehearsal_truth(record) = Dict{Symbol,Float64}(
    Symbol(k) => Float64(v) for (k, v) in record.parameters if v isa Real
)

# ── the ancilla marginal (the family's measurement path) ──────────────────────
# Transmon populations marginalized over the cavity, from the iso-packed state a
# TwinSoc hands its measurement_fn (ket form: length 2d; density form: 2d²). The
# record's 2×2 readout confusion operates on THIS marginal.
function _ancilla_marginal(iso::AbstractVector{<:Real}, n_t::Int, n_f::Int)
    d = n_t * n_f
    n = length(iso)
    (n == 2 * d || n == 2 * d^2) || error(
        "rehearsal ancilla marginal: expected an iso-packed ket (length $(2d)) or " *
        "vectorized density matrix (length $(2d^2)); got length $n",
    )
    joint = Vector{Float64}(undef, d)
    if n == 2 * d^2
        for i = 1:d
            joint[i] = Float64(iso[(i-1)*d+i])    # ρ_ii (real diagonal)
        end
    else
        for i = 1:d
            re, im_ = iso[i], iso[d+i]
            joint[i] = re^2 + im_^2
        end
    end
    p = zeros(Float64, n_t)
    for i = 1:d
        p[((i-1)%n_t)+1] += joint[i]              # sum the fock axis
    end
    return p
end
rehearsal_ancilla_marginal(iso) = _ancilla_marginal(iso, M3B_N_TRANSMON, M3B_N_FOCK)

m3b_psi0() = begin                                # |g,0⟩ — fock 0, transmon g
    ψ = zeros(ComplexF64, M3B_N_TRANSMON * M3B_N_FOCK)
    ψ[1] = 1.0
    ψ
end

# ── the structured Ramsey prep (the QCP's target class) ───────────────────────
# A dispersive Ramsey construction, grounded in the certification machinery's
# physics: displace the cavity on the g-branch (resonant, χ-blind), then a
# narrowband sin² ancilla pulse that rotates the photon-resolved components
# against their χ·k pulls (creating the |e, n≥1⟩ amplitude — the only
# χ-visible sector), then a hold during which the e-branch components precess
# at χ·k. The TARGET is the construction's own believed-χ endpoint: the QCP
# warm-starts at the solution and polishes it (deterministic). The prep's
# χ-fragility is structural: ~3% fidelity loss per 15 kHz of χ drift
# (demonstrated; the drift-tracking evidence rides it).
const M3B_T_GATE = 10000.0      # ns: displace 1000 + comb pulse 3000 + hold 6000
const M3B_N_KNOTS = 81
const M3B_Q = 1e4
const M3B_R = 1e-3

function structured_prep(truth::Dict{Symbol,Float64})
    T_disp, T_spec, T_hold = 1000.0, 3000.0, 6000.0
    T = T_disp + T_spec + T_hold
    T == M3B_T_GATE || error("structured_prep: gate time drifted from the constant")
    times = collect(range(0.0, T, length = M3B_N_KNOTS))
    sint(t) = sin(π * clamp(t, 0.0, 1.0))^2
    α_goal = 1.2
    ε_c = 2 * α_goal / T_disp                       # resonant displacement ramp
    Ω_q = π / T_spec                                # comb π/2: ∫Ω dt = π/2
    u = zeros(4, M3B_N_KNOTS)
    for (i, t) in enumerate(times)
        if t <= T_disp                              # cavity I: sin²-ramped displacement
            w =
                t < 150 ? sint(t / 300) :
                (t > T_disp - 150 ? sint((T_disp - t) / 300) : 1.0)
            u[3, i] = ε_c * w
        elseif t <= T_disp + T_spec                 # transmon I: narrowband comb pulse
            u[1, i] = Ω_q * sint((t - T_disp) / T_spec)
        end                                         # hold: zero drives
    end
    pulse = LinearSplinePulse(u, times)
    sys = rehearsal_family(truth)
    ψ0 = m3b_psi0()
    qt = rollout(KetTrajectory(sys, pulse, ψ0, ψ0), pulse)
    ψt = qt(times[end])
    return pulse, times, ψt / norm(ψt)              # the target: the believed-χ endpoint
end

"the nominal QCP: unsolved, warm-started at the structured prep (caller solves)"
function nominal_qcp(truth::Dict{Symbol,Float64}, pulse, ψt)
    sys = rehearsal_family(truth)
    return SplinePulseProblem(
        KetTrajectory(sys, pulse, m3b_psi0(), ψt),
        M3B_N_KNOTS;
        Q = M3B_Q,
        R = M3B_R,
    )
end

# ── truth-side evaluation (the harness's evidence — never loop input) ─────────
_truth_fidelity(truth::Dict{Symbol,Float64}, pulse::AbstractPulse, times, ψt) = begin
    sys = rehearsal_family(truth)
    qt = rollout(KetTrajectory(sys, pulse, m3b_psi0(), ψt), pulse)
    return abs(dot(ψt, qt(times[end])))^2
end
truth_fidelity(truth::Dict{Symbol,Float64}, qcp, times, ψt) =
    _truth_fidelity(truth, get_pulse(qcp.qtraj), times, ψt)
truth_fidelity(truth::Dict{Symbol,Float64}, pulse::AbstractPulse, times, ψt) =
    _truth_fidelity(truth, pulse, times, ψt)

# ── measurement geometry ──────────────────────────────────────────────────────
# The TwinSoc measures on the DAC grid (dac_rate samples/ns); the six measurement
# knots sit inside the hold, where the prep's marginal χ-slope is uniform
# (~5e-4/kHz demonstrated) — the χ-fit's signal.
const M3B_DAC_RATE = 0.1
const M3B_MEAS_TIMES = [4500.0, 5500.0, 6500.0, 7500.0, 8500.0, 10000.0]
m3b_dac_times() =
    collect(range(0.0, M3B_T_GATE, length = round(Int, M3B_T_GATE * M3B_DAC_RATE) + 1))
m3b_meas_indices() = [round(Int, t * M3B_DAC_RATE) + 1 for t in M3B_MEAS_TIMES]
m3b_meas_times() = begin
    dac = m3b_dac_times()
    [dac[i] for i in m3b_meas_indices()]
end

m3b_confusion(record) = begin
    rows = record.noise["readout_confusion"]["value"]
    Matrix{Float64}([rows[i][j] for i in eachindex(rows), j in eachindex(rows)])
end

# the GOAL measurements: the believed-parameter response of the solved pulse at
# the measurement knots, through the record's readout confusion — the same
# convention the soc reports (q = Cᵀp)
function goal_measurements(truth::Dict{Symbol,Float64}, pulse, record)
    C = m3b_confusion(record)
    sys = rehearsal_family(truth)
    qt = rollout(KetTrajectory(sys, pulse, m3b_psi0(), m3b_psi0()), pulse)
    idxs = m3b_meas_indices()
    ts = m3b_meas_times()
    Measurement[
        Measurement(
            C' * _ancilla_marginal(ket_to_iso(qt(t)), M3B_N_TRANSMON, M3B_N_FOCK),
            idxs[j],
        ) for (j, t) in enumerate(ts)
    ]
end

# the measurement models: the chassis's rows are deterministic stubs (the twin's
# shot statistics enter through W_task — the Fisher/IRLS increment's designed
# composition: "on a deterministic-stub model the chassis cost is then exactly
# the Fisher-weighted SSR"); the shot-statistics model is the weights' source
function measurement_models(shots::Int)
    idxs = m3b_meas_indices()
    chassis = MeasurementModel(
        :ψ̃,
        DeterministicMeasurement[
            DeterministicMeasurement(rehearsal_ancilla_marginal) for _ in idxs
        ],
        idxs,
    )
    shot = MeasurementModel(
        :ψ̃,
        ShotNoiseMeasurement[
            ShotNoiseMeasurement(rehearsal_ancilla_marginal, shots, population_covariance) for _ in idxs
        ],
        idxs,
    )
    return chassis, shot
end

# ── the χ-tracking device model (the adapting loop's in-loop refit) ───────────
# predict rolls the pulse through the family system at the CURRENT estimate χ̂;
# adapt! refits χ̂ from the loop's own experiment data by a 1-D physics-model
# fit (grid scan + golden refine over the SSR between the model response and
# the measured data) — never reading the twin's truth.
mutable struct ChiTrackingModel <: Intonato.AbstractDeviceModel
    truth_base::Dict{Symbol,Float64}     # believed parameters (record)
    ψ_init::Vector{ComplexF64}
    ψ_target::Vector{ComplexF64}
    meas_times::Vector{Float64}
    meas_indices::Vector{Int}
    confusion::Matrix{Float64}
    χ̂_kHz::Float64
    bracket_kHz::Float64
    scan_step_kHz::Float64
    n_rollouts::Int
end
_truth_with_chi(m::ChiTrackingModel) = merge(copy(m.truth_base), Dict(:chi_kHz => m.χ̂_kHz))

function _model_response(m::ChiTrackingModel, pulse::AbstractPulse)
    sys = rehearsal_family(_truth_with_chi(m))
    qt = rollout(KetTrajectory(sys, pulse, m.ψ_init, m.ψ_target), pulse)
    m.n_rollouts += 1
    out = Float64[]
    for t in m.meas_times
        p = _ancilla_marginal(ket_to_iso(qt(t)), M3B_N_TRANSMON, M3B_N_FOCK)
        append!(out, m.confusion' * p)
    end
    return out
end

function Intonato.predict(
    m::ChiTrackingModel,
    pulse::AbstractPulse,
    ::Intonato.MeasurementModel,
)
    flat = _model_response(m, pulse)
    return Measurement[
        Measurement(flat[(j-1)*2 .+ (1:2)], m.meas_indices[j]) for
        j in eachindex(m.meas_indices)
    ]
end

function Intonato.adapt!(m::ChiTrackingModel, pulse::AbstractPulse, y_exp)
    flat_exp = reduce(vcat, Vector{Float64}[Float64.(e.data) for e in y_exp])
    ssr(χ) = begin
        m.χ̂_kHz = χ
        sum(abs2, _model_response(m, pulse) .- flat_exp)
    end
    # coarse grid over the bracket, then golden-section refine within ±step
    best = Inf
    bestχ = m.χ̂_kHz
    for χ = (m.χ̂_kHz-m.bracket_kHz):m.scan_step_kHz:(m.χ̂_kHz+m.bracket_kHz)
        v = ssr(χ)
        if v < best
            best = v
            bestχ = χ
        end
    end
    φ = (sqrt(5) - 1) / 2
    a, b = bestχ - m.scan_step_kHz, bestχ + m.scan_step_kHz
    c = b - φ * (b - a)
    d = a + φ * (b - a)
    fc, fd = ssr(c), ssr(d)
    for _ = 1:24
        if fc < fd
            b, d, fd = d, c, fc
            c = b - φ * (b - a)
            fc = ssr(c)
        else
            a, c, fc = c, d, fd
            d = a + φ * (b - a)
            fd = ssr(d)
        end
    end
    m.χ̂_kHz = (a + b) / 2
    return m
end

"the system the replan strategy plans against (duck-typed over the two models)"
rehearsal_model_system(m::ChiTrackingModel) = rehearsal_family(_truth_with_chi(m))
rehearsal_model_system(m::NominalModel) = m.system

# ── the model-replan strategy (the adapting loop's inner step) ────────────────
# Each accepted iteration replans the pulse against the device model's CURRENT
# system (a short warm-started QCP solve) — "adapt, then replan". The paired
# control run uses the SAME strategy over a frozen NominalModel: its replans
# optimize against the stale believed-χ model and cannot track the drift.
mutable struct ModelReplanStrategy <: Intonato.AbstractTuningStrategy
    ψ0::Vector{ComplexF64}
    ψt::Vector{ComplexF64}
    max_ipopt_iter::Int
    cand::Any
    n_replans::Int
    last_nlp::Float64
end
ModelReplanStrategy(; ψ0, ψt, max_ipopt_iter) =
    ModelReplanStrategy(ψ0, ψt, max_ipopt_iter, nothing, 0, 0.0)

Intonato.prepare_strategy(s::ModelReplanStrategy, ptp, z_ref; kwargs...) =
    (s.cand = nothing; s.n_replans = 0; s.last_nlp = 0.0; s)
Intonato.candidate_trajectory(s::ModelReplanStrategy) = s.cand
Intonato.last_timings(s::ModelReplanStrategy) = (sysid = 0.0, nlp = s.last_nlp)

function Intonato.step(s::ModelReplanStrategy, ctx)
    sys = rehearsal_model_system(ctx.device_model)
    traj = KetTrajectory(sys, ctx.pulse, s.ψ0, s.ψt)
    qcp = SplinePulseProblem(traj, M3B_N_KNOTS; Q = M3B_Q, R = M3B_R)
    t0 = time()
    Piccolo.solve!(qcp; max_iter = s.max_ipopt_iter, verbose = false, print_level = 0)
    s.last_nlp = time() - t0
    # the candidate: z_ref's control block ← the replanned knots
    z = deepcopy(ctx.z_ref)
    rtraj = qcp.prob.trajectory
    z.data[z.components[:u], :] .= rtraj.data[rtraj.components[:u], :]
    s.cand = z
    s.n_replans += 1
    return Intonato.extract_pulse(ctx.qcp.qtraj, z)
end

# ── the rehearsal ─────────────────────────────────────────────────────────────
# phases = :full (the default) runs the whole gate: the static-truth baseline,
# the drift epochs (harness-owned advance! between iteration batches, TwinSoc's
# per-acquire dt staying 0), the calibrate! write-back at each epoch's
# checkpoint, and the paired no-adaptation control (the adapted run's recorded
# drift path replayed via JumpSchedule, model frozen at NominalModel).
# phases = :baseline runs the skeleton's static-truth phase only.
const M3B_SHOTS = 20_000        # sized so the GLS statistic discriminates the
# drift scale from the χ²(12) shot floor
const M3B_TOL = 40.0            # convergence: the Fisher-weighted SSR consistent
# with pure shot noise (χ²(12), mean 12)
const M3B_QCP_MAX_ITER = 12
const M3B_REPLAN_IPOPT = 15     # the per-epoch replan must fully re-converge
# against the refit model (~one polish budget)
const M3B_ITERS_PER_EPOCH = 2
const M3B_EPOCH_DAYS = 5.0
const M3B_N_EPOCHS = 3
const M3B_FIT_BRACKET_KHZ = 40.0
const M3B_FIT_STEP_KHZ = 2.0

m3b_drift_phases_pending() = false
# AC4's replay is live: three rehearsals per suite run (two in-process + one
# fresh child process) at the replay item's pinned minimal budget
m3b_replay_pending() = false

# the replay item's rehearsal budget: ONE drift epoch — every mechanism of the
# whole rehearsal (QCP, twin soc, loop, χ-fit, replan, acceptance, advance!,
# calibrate!, the paired control) runs and is digest-pinned at this
# configuration; the epoch count only scales the wall-clock (~2.6 min/rehearsal
# vs ~6 at the drift item's 3), keeping the suite's three replay runs reasonable
const M3B_REPLAY_EPOCHS = 1

# the rehearsal's replay seed (pinned once, used by every rehearsal testitem —
# each testitem is its own module, so the seed lives here, in the harness)
const REHEARSAL_SEED = 0x6D3B

struct RehearsalSummary
    seed::Int
    phases::Symbol
    shots::Int
    n_epochs::Int
    F_baseline::Float64
    J_baseline::Float64
    converged_baseline::Bool
    chi_truth::Vector{Float64}
    chi_believed::Vector{Float64}
    chi_estimated::Vector{Float64}
    F_adapted::Vector{Float64}
    F_control::Vector{Float64}
    J_adapted::Vector{Float64}
    J_control::Vector{Float64}
    n_experiments_adapted::Int
    n_experiments_control::Int
    est_error_max::Float64
    truth_moved_on_advance::Vector{Bool}
    belief_moved_on_advance::Vector{Bool}
    belief_moved_on_calibrate::Vector{Bool}
    truth_moved_on_calibrate::Vector{Bool}
end

# bit-exact digest: Float64 bit patterns — replay compares == at the bit level
_bits(x::Float64) = string(reinterpret(UInt64, x); base = 16)
function rehearsal_digest(s::RehearsalSummary)
    parts = String[
        string(s.seed),
        string(s.phases),
        string(s.shots),
        string(s.n_epochs),
        _bits(s.F_baseline),
        _bits(s.J_baseline),
        string(s.converged_baseline),
        _bits(s.est_error_max),
    ]
    for f in (
        :chi_truth,
        :chi_believed,
        :chi_estimated,
        :F_adapted,
        :F_control,
        :J_adapted,
        :J_control,
        :truth_moved_on_advance,
        :belief_moved_on_advance,
        :belief_moved_on_calibrate,
        :truth_moved_on_calibrate,
    )
        push!(parts, join(_bits.(Float64.(collect(getfield(s, f)))), ","))
    end
    push!(parts, string(s.n_experiments_adapted), string(s.n_experiments_control))
    return join(parts, "|")
end

# the twin → soc face → backend adapter → experiment factory stack (one per run)
function _rehearsal_experiment(twin, ψt, chassis_mm, shots)
    ext = Base.get_extension(Strumento, :StrumentoPiccoloExt)
    soc = ext.TwinSoc(
        twin,
        m3b_psi0(),
        ψt;
        families = Dict("bosonic" => rehearsal_family),
        measurement_fn = rehearsal_ancilla_marginal,
        shots = shots,
        exact = false,
        dt = 0.0,
        dac_rate = M3B_DAC_RATE,
    )
    map = QickChannelMap(
        [
            QickGenChannel(0, 5e9; i_drive = 1, q_drive = 2),
            QickGenChannel(1, 6e9; i_drive = 3, q_drive = 4),
        ];
        n_drives = 4,
    )
    backend = StrumentoBackend(soc, map, m3b_meas_indices())
    return StrumentoExperiment(backend; measurement_model = chassis_mm)
end

# the paired control's drift plan: the adapted run's recorded truth path,
# replayed EXACTLY via JumpSchedule — the deltas fire at the advance! call
# times, which are t = 0, epoch, 2·epoch, … (advance! passes the CURRENT twin
# time to the plan BEFORE incrementing it, so the scheduled times start at 0)
function _control_replay_plan(truth_path::Vector{Float64}, epoch_days::Float64)
    times = [(k - 1) * epoch_days for k = 1:(length(truth_path)-1)]
    deltas = [truth_path[k+1] - truth_path[k] for k = 1:(length(truth_path)-1)]
    return Strumento.DriftPlan(
        :chi_kHz => [Strumento.JumpSchedule(times = times, deltas = deltas)],
    )
end

function run_rehearsal(
    seed::Integer;
    phases::Symbol = :full,
    n_epochs::Int = M3B_N_EPOCHS,
    shots::Int = M3B_SHOTS,
    qcp_max_iter::Int = M3B_QCP_MAX_ITER,
    verbose::Bool = false,
)
    phases in (:baseline, :full) ||
        throw(ArgumentError("phases must be :baseline or :full, got :$phases"))

    record = Strumento.load_record(rehearsal_fixture_path())
    believed_truth = rehearsal_truth(record)

    # the drift plan from the record's OWN drift priors: the record family's OU
    # shape (θ = 1/τ, σ = σ_rel·|χ₀|, μ = the record value — calibrations re-lock
    # to nominal) composed with the declared thermal-soak Ramp — the drift-process
    # catalogue's monotone component, at the prior's own scale
    prior = record.drift_priors["chi_kHz"]
    procs = prior["process"] isa AbstractVector ? prior["process"] : [prior["process"]]
    procs == ["ou", "ramp"] ||
        error("rehearsal: the fixture's χ prior must compose [ou, ramp], got $procs")
    σ_stat = Float64(prior["sigma_rel"]) * abs(believed_truth[:chi_kHz])
    plan = Strumento.DriftPlan(
        :chi_kHz => [
            Strumento.OrnsteinUhlenbeck(
                theta = 1.0 / Float64(prior["tau_days"]),
                sigma = σ_stat,
                mu = believed_truth[:chi_kHz],
            ),
            Strumento.Ramp(rate = Float64(prior["ramp_kHz_per_day"])),
        ],
    )

    # 1. the nominal QCP at the BELIEVED parameters: the analytic structured
    #    prep warm start, polished by a deterministic Ipopt solve
    pulse0, times, ψt = structured_prep(believed_truth)
    qcp = nominal_qcp(believed_truth, pulse0, ψt)
    Piccolo.solve!(qcp; max_iter = qcp_max_iter, verbose = false, print_level = 0)
    pulse_solved = get_pulse(qcp.qtraj)
    F_baseline = truth_fidelity(believed_truth, pulse_solved, times, ψt)

    # 2. goal + measurement models + the Fisher/IRLS task weights
    y_goal = goal_measurements(believed_truth, pulse_solved, record)
    chassis_mm, shot_mm = measurement_models(shots)
    W_task = Intonato.binomial_fisher_weights(shot_mm, y_goal)

    # 3. the twin + soc face + backend adapter + experiment factory
    twin = Strumento.instantiate(rehearsal_fixture_path(); drift = plan, seed = seed)
    qexp = _rehearsal_experiment(twin, ψt, chassis_mm, shots)

    # 4. the adapting loop's machinery: the χ-tracking device model + the
    #    model-replan strategy (on static truth the baseline phase converges at
    #    the shot floor before they fire; the drift epochs drive them)
    devmodel = ChiTrackingModel(
        believed_truth,
        m3b_psi0(),
        ψt,
        m3b_meas_times(),
        m3b_meas_indices(),
        m3b_confusion(record),
        Float64(believed_truth[:chi_kHz]),
        M3B_FIT_BRACKET_KHZ,
        M3B_FIT_STEP_KHZ,
        0,
    )
    strategy =
        ModelReplanStrategy(ψ0 = m3b_psi0(), ψt = ψt, max_ipopt_iter = M3B_REPLAN_IPOPT)

    # 5. the static-truth baseline QILC phase: PulseTuningProblem through the
    #    SocBackend over TwinSoc — converges when the weighted residual is
    #    consistent with pure shot noise
    ptp = PulseTuningProblem(
        qcp,
        qexp,
        chassis_mm;
        strategy = strategy,
        device_model = devmodel,
        y_goal = y_goal,
        W_task = W_task,
        verbose = verbose,
    )
    Piccolo.solve!(ptp; max_iter = 3, tol = M3B_TOL, verbose = verbose)
    J_baseline = ptp.result.history[end].J_exp
    converged_baseline = ptp.result.converged
    n_exp_a = ptp.result.n_experiments

    # checkpoint ledgers (checkpoint 0 = the pristine pre-drift state)
    chi_truth = Float64[believed_truth[:chi_kHz]]
    chi_believed = Float64[believed_truth[:chi_kHz]]
    chi_estimated = Float64[devmodel.χ̂_kHz]
    F_adapted = Float64[F_baseline]
    F_control = Float64[F_baseline]
    J_adapted = Float64[J_baseline]
    J_control = Float64[J_baseline]
    truth_path = Float64[believed_truth[:chi_kHz]]        # for the control's replay
    truth_moved_on_advance = Bool[]
    belief_moved_on_advance = Bool[]
    belief_moved_on_calibrate = Bool[]
    truth_moved_on_calibrate = Bool[]

    if phases === :full
        # ── 6. the adapted run's drift epochs ─────────────────────────────────
        # harness-owned choreography: advance! between iteration batches, with
        # TwinSoc's per-acquire dt staying 0 so the drift timing is explicit;
        # calibrate! write-back at each epoch's checkpoint (belief only)
        for k = 1:n_epochs
            # drift injection — truth moves, belief must hold
            χ_pre = twin.truth[:chi_kHz]
            b_pre = Strumento.believed(twin)["chi_kHz"]
            Strumento.advance!(twin, M3B_EPOCH_DAYS)
            push!(truth_moved_on_advance, twin.truth[:chi_kHz] != χ_pre)
            push!(belief_moved_on_advance, Strumento.believed(twin)["chi_kHz"] != b_pre)
            push!(truth_path, twin.truth[:chi_kHz])

            # the QILC phase against the drifted truth: the chassis's adapt!
            # refits χ̂ from the loop's own data, the strategy replans against
            # the refit model, the acceptance measures on the twin
            Piccolo.solve!(
                ptp;
                max_iter = M3B_ITERS_PER_EPOCH,
                tol = M3B_TOL,
                verbose = verbose,
            )
            n_exp_a += ptp.result.n_experiments

            # checkpoint: the loop's estimate vs the drifted truth
            push!(chi_truth, twin.truth[:chi_kHz])
            push!(chi_estimated, devmodel.χ̂_kHz)
            push!(F_adapted, truth_fidelity(twin.truth, get_pulse(qcp.qtraj), times, ψt))
            push!(J_adapted, ptp.result.history[end].J_exp)

            # the belief write-back — calibration moves belief, never truth
            b_pre_cal = Strumento.believed(twin)["chi_kHz"]
            χ_pre_cal = twin.truth[:chi_kHz]
            Strumento.calibrate!(twin, Dict("chi_kHz" => devmodel.χ̂_kHz))
            push!(
                belief_moved_on_calibrate,
                Strumento.believed(twin)["chi_kHz"] != b_pre_cal,
            )
            push!(truth_moved_on_calibrate, twin.truth[:chi_kHz] != χ_pre_cal)
            push!(chi_believed, Strumento.believed(twin)["chi_kHz"])
        end

        # ── 7. the paired no-adaptation control ───────────────────────────────
        # same seed, the adapted run's EXACT drift path replayed via
        # JumpSchedule, the same replan strategy — only the device model is
        # frozen at the believed parameters (NominalModel: adapt! is a no-op),
        # so the control's replans optimize against the stale model and cannot
        # track the drift
        twin_c = Strumento.instantiate(
            rehearsal_fixture_path();
            drift = _control_replay_plan(truth_path, M3B_EPOCH_DAYS),
            seed = seed,
        )
        qexp_c = _rehearsal_experiment(twin_c, ψt, chassis_mm, shots)
        qcp_c = SplinePulseProblem(
            KetTrajectory(rehearsal_family(believed_truth), pulse_solved, m3b_psi0(), ψt),
            M3B_N_KNOTS;
            Q = M3B_Q,
            R = M3B_R,
        )   # warm at the solved pulse; no re-solve
        ptp_c = PulseTuningProblem(
            qcp_c,
            qexp_c,
            chassis_mm;
            strategy = ModelReplanStrategy(
                ψ0 = m3b_psi0(),
                ψt = ψt,
                max_ipopt_iter = M3B_REPLAN_IPOPT,
            ),
            device_model = NominalModel(rehearsal_family(believed_truth), m3b_psi0(), ψt),
            y_goal = y_goal,
            W_task = W_task,
            verbose = verbose,
        )
        n_exp_c = 0
        for k = 1:n_epochs
            Strumento.advance!(twin_c, M3B_EPOCH_DAYS)   # fires the scheduled delta
            Piccolo.solve!(
                ptp_c;
                max_iter = M3B_ITERS_PER_EPOCH,
                tol = M3B_TOL,
                verbose = verbose,
            )
            n_exp_c += ptp_c.result.n_experiments
            push!(
                F_control,
                truth_fidelity(twin_c.truth, get_pulse(qcp_c.qtraj), times, ψt),
            )
            push!(J_control, ptp_c.result.history[end].J_exp)
        end

        # the loop's demonstrated estimation error (post-fit checkpoints)
        est_error_max =
            maximum(abs(chi_estimated[k] - chi_truth[k]) for k = 2:length(chi_truth))
    else
        est_error_max = 0.0
        n_exp_c = 0
    end

    return RehearsalSummary(
        Int(seed),
        phases,
        shots,
        n_epochs,
        F_baseline,
        J_baseline,
        converged_baseline,
        chi_truth,
        chi_believed,
        chi_estimated,
        F_adapted,
        F_control,
        J_adapted,
        J_control,
        n_exp_a,
        n_exp_c,
        est_error_max,
        truth_moved_on_advance,
        belief_moved_on_advance,
        belief_moved_on_calibrate,
        truth_moved_on_calibrate,
    )
end
