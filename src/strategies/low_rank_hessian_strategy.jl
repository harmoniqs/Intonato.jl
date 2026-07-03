# ──── Low-rank Hessian tuning strategy (Liu et al. 2026) ─────────────────────
#
# Closed-loop calibration in the low-rank principal subspace of the model
# cost Hessian, after "High-fidelity neutral atom gates leveraging low-rank
# Hessian optimization" (Liu, Bornet, Kurdak, Xiao, Li, Zhang, Thompson —
# arXiv:2606.05060). The device model supplies the *directions* and the
# *curvatures* (both robust to moderate model mismatch); the cost *gradient*
# is measured on the device by central finite differences along each
# direction. One damped Newton step per outer iteration.

"""
    LowRankHessianStrategy(; kwargs...) <: AbstractTuningStrategy

Measured-gradient Newton in the low-rank principal subspace of the model
cost Hessian (Liu et al. 2026, arXiv:2606.05060).

At the first `step` (and every `refresh_every` iterations if nonzero), the
strategy builds the model Hessian of the strategy cost in flattened
control-knot space through the problem's `device_model` (`predict` —
model-side only, no device cost). With `cost_mode = :squared` (default) the
cost is the chassis's whitened cost `c(u) = ‖W (y_goal − y_model(u))‖²` and
the Hessian is the Gauss–Newton form `H = 2 GᵀW²G` from a central-FD model
Jacobian `G` — correct for vector observables whose residual is first-order
in the control error. With `cost_mode = :linear` the cost is the whitened
residual sum `c(u) = Σ W (y_goal − y_model(u))` and the Hessian is the exact
central-second-difference Hessian — required for defect-scalar measurements
(a `goal_fidelity` row, where `c = w(1 − F)` is the Liu et al. objective):
there the GN form is rank-one and vanishes at the design optimum, while the
exact Hessian carries the full Gram structure `2 Re(J†J)` of the
endpoint-defect map. The Hessian's eigendecomposition gives `r`
well-conditioned principal directions `V` with curvatures `λ`. Each outer
iteration then *measures* the cost gradient on the device along each
direction (central differences, `2r` probe experiments) and takes the damped
Newton step `δu = -V ((g) ./ (λ .+ μ))` as the candidate the chassis
accepts/rejects.

The gradient is measured rather than modeled, so the step direction is
correct even when the model is wrong — the model only supplies the subspace
and the curvature, both of which are robust to moderate mismatch. This is the
complement of a model-gradient strategy (e.g. an ILC step): it needs no
parameter calibration, at the price of `2r` probe experiments per iteration.

# Keyword Arguments
- `control_name::Symbol = :u`: trajectory component holding the tunable
  control knots (perturbations and candidates act on this component only).
- `fd_step::Float64 = 1e-3`: device probe amplitude `h` along each direction.
- `model_fd_step::Float64 = 1e-6`: model-side finite-difference step for `G`
  (`:squared` mode).
- `model_hessian_step::Float64 = 1e-3`: model-side second-difference step for
  the exact Hessian (`:linear` mode; coarser than `model_fd_step` because
  second differences divide roundoff by the step squared).
- `n_modes::Union{Nothing,Int} = nothing`: cap on the number of principal
  directions (`nothing` ⇒ all directions above `rank_tol`).
- `rank_tol::Float64 = 1e-10`: keep eigenpairs with `λ ≥ rank_tol · λ_max`.
- `damping::Float64 = 1e-3`: relative Newton damping; `μ = damping · λ_max`.
- `refresh_every::Int = 0`: rebuild directions every `n` outer iterations on
  the current pulse (`0` ⇒ build once at the first step). Rebuilding on a
  drifted/calibrated operating point is the paper's "more than one round"
  extension.
- `cost_mode::Symbol = :squared`: strategy cost. `:squared` (whitened
  residual norm², GN Hessian) for vector observables; `:linear` (whitened
  residual sum, exact Hessian) for defect-scalar rows such as
  `goal_fidelity` — the Liu et al. configuration.

# Probe accounting
The `2r` gradient probes per iteration are strategy-internal experiment
calls; the chassis's `TuningResult.n_experiments` does **not** include them.
The running total is exposed as `strategy.n_probes` and the per-iteration
probe/build wall-clock in `last_timings` (`:nlp` = probes + Newton,
`:sysid` = direction build).

# Reference
Liu, Bornet, Kurdak, Xiao, Li, Zhang, Thompson, *High-fidelity neutral atom
gates leveraging low-rank Hessian optimization*, arXiv:2606.05060 (2026).
"""
mutable struct LowRankHessianStrategy <: AbstractTuningStrategy
    # ── config ──
    control_name::Symbol
    fd_step::Float64
    model_fd_step::Float64
    model_hessian_step::Float64
    n_modes::Union{Nothing,Int}
    rank_tol::Float64
    damping::Float64
    refresh_every::Int
    cost_mode::Symbol
    # ── per-solve state ──
    experiment::Union{Nothing,AbstractExperiment}
    V::Union{Nothing,Matrix{Float64}}      # M × r principal directions
    λ::Union{Nothing,Vector{Float64}}      # r curvatures (descending)
    w_ref::Union{Nothing,Vector{Float64}}  # whitening frozen at direction build
    cand::Any                              # candidate NamedTrajectory (or nothing)
    n_probes::Int
    timings::NamedTuple{(:sysid, :nlp),NTuple{2,Float64}}
end

function LowRankHessianStrategy(;
    control_name::Symbol = :u,
    fd_step::Float64 = 1e-3,
    model_fd_step::Float64 = 1e-6,
    model_hessian_step::Float64 = 1e-3,
    n_modes::Union{Nothing,Int} = nothing,
    rank_tol::Float64 = 1e-10,
    damping::Float64 = 1e-3,
    refresh_every::Int = 0,
    cost_mode::Symbol = :squared,
)
    cost_mode in (:squared, :linear) ||
        throw(ArgumentError("cost_mode must be :squared or :linear, got :$cost_mode"))
    return LowRankHessianStrategy(
        control_name,
        fd_step,
        model_fd_step,
        model_hessian_step,
        n_modes,
        rank_tol,
        damping,
        refresh_every,
        cost_mode,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        0,
        (sysid = 0.0, nlp = 0.0),
    )
end

# Stash the device experiment for gradient probes and reset per-solve state.
function prepare_strategy(s::LowRankHessianStrategy, ptp, z_ref; kwargs...)
    s.experiment = ptp.experiment
    s.V = nothing
    s.λ = nothing
    s.w_ref = nothing
    s.cand = nothing
    s.n_probes = 0
    s.timings = (sysid = 0.0, nlp = 0.0)
    return s
end

candidate_trajectory(s::LowRankHessianStrategy) = s.cand
last_timings(s::LowRankHessianStrategy) = s.timings

# Flatten a measurement vector to one residual-ordered data vector.
_flat_data(y::Vector{Measurement}) = reduce(vcat, Vector{Float64}[m.data for m in y])

# Strategy cost in the units the model Hessian and the measured gradient
# share. `:squared` is the chassis's whitened cost c = ‖w ⊙ (y_goal − y)‖²;
# `:linear` is the whitened residual sum c = Σ w ⊙ (y_goal − y) — the natural
# cost when the measurement itself is already a defect magnitude (e.g. a
# `goal_fidelity` row, where c = w·(1 − F) is the Liu et al. objective).
function _lrh_cost(
    y_goal::Vector{Measurement},
    y::Vector{Measurement},
    w::Vector{Float64},
    cost_mode::Symbol,
)
    r = _flat_residual(y_goal, y)
    return cost_mode === :linear ? sum(w .* r) : sum(abs2, w .* r)
end

# Trajectory-space probe: copy z, add δ (flattened) to the control block, and
# extract the pulse the same way the chassis does.
function _perturbed_pulse(qcp, z_ref, urows, δ::Vector{Float64})
    z = deepcopy(z_ref)
    block = view(z.data, urows, :)
    block .+= reshape(δ, length(urows), :)
    return extract_pulse(qcp.qtraj, z)
end

# Exact model-cost Hessian by central second differences of the linear
# whitened cost, evaluated through the `predict` seam (model-side only, no
# device cost): H_jj = [c(+δeⱼ) + c(−δeⱼ) − 2c₀]/δ², and
# H_jk = [c(+δeⱼ+δeₖ) − c(+δeⱼ−δeₖ) − c(−δeⱼ+δeₖ) + c(−δeⱼ−δeₖ)]/(4δ²).
# The step `s.model_hessian_step` is deliberately coarser than
# `s.model_fd_step`: second differences divide roundoff by δ², so δ ~ 1e-3
# keeps the noise floor near 1e-10 while the curvature error stays O(δ²).
function _exact_cost_hessian(s::LowRankHessianStrategy, ctx, urows, M::Int, w)
    δ = s.model_hessian_step
    c =
        v -> _lrh_cost(
            ctx.y_goal,
            predict(
                ctx.device_model,
                _perturbed_pulse(ctx.qcp, ctx.z_ref, urows, v),
                ctx.measurement_model,
            ),
            w,
            :linear,
        )
    e = zeros(M)
    c0 = c(e)
    H = Matrix{Float64}(undef, M, M)
    cp = Vector{Float64}(undef, M)
    cm = Vector{Float64}(undef, M)
    for j = 1:M
        e[j] = δ
        cp[j] = c(e)
        e[j] = -δ
        cm[j] = c(e)
        e[j] = 0.0
        H[j, j] = (cp[j] + cm[j] - 2c0) / δ^2
    end
    for j = 1:M, k = (j+1):M
        e[j] = δ
        e[k] = δ
        cpp = c(e)
        e[k] = -δ
        cpm = c(e)
        e[j] = -δ
        cmm = c(e)
        e[k] = δ
        cmp = c(e)
        e[j] = 0.0
        e[k] = 0.0
        H[j, k] = (cpp - cpm - cmp + cmm) / (4δ^2)
        H[k, j] = H[j, k]
    end
    return Symmetric(H)
end

# Build the principal directions from the model. `:squared` mode: central-FD
# model Jacobian G through `predict(device_model, ·, measurement_model)`, GN
# Hessian H = 2 GᵀW²G. `:linear` mode: exact central-second-difference Hessian
# of the linear model cost c(u) = Σ w ⊙ (y_goal − y_model(u)) — required when
# the measurement is a defect scalar (a `goal_fidelity` row), where the GN
# form is rank-one and vanishes at the design optimum while the exact Hessian
# carries the full Gram structure 2 Re(J†J) of the endpoint-defect map.
# Eigendecomposition, keep the well-conditioned top modes.
function _build_directions!(s::LowRankHessianStrategy, ctx)
    urows = ctx.z_ref.components[s.control_name]
    M = length(urows) * size(ctx.z_ref.data, 2)

    w = collect(Float64, ctx.w)
    s.w_ref = w

    H = if s.cost_mode === :linear
        _exact_cost_hessian(s, ctx, urows, M, w)
    else
        δm = s.model_fd_step
        G = Matrix{Float64}(undef, length(w), M)
        e = zeros(M)
        for j = 1:M
            e[j] = δm
            y_plus = predict(
                ctx.device_model,
                _perturbed_pulse(ctx.qcp, ctx.z_ref, urows, e),
                ctx.measurement_model,
            )
            e[j] = -δm
            y_minus = predict(
                ctx.device_model,
                _perturbed_pulse(ctx.qcp, ctx.z_ref, urows, e),
                ctx.measurement_model,
            )
            e[j] = 0.0
            G[:, j] = (_flat_data(y_plus) .- _flat_data(y_minus)) ./ (2δm)
        end
        WG = w .* G
        Symmetric(2 .* (WG' * WG))
    end
    F = eigen(H)
    order = sortperm(F.values; rev = true)
    λs = F.values[order]
    λmax = max(λs[1], eps())
    r = count(≥(s.rank_tol * λmax), λs)
    if !isnothing(s.n_modes)
        r = min(r, s.n_modes)
    end
    r = max(r, 1)

    s.V = F.vectors[:, order[1:r]]
    s.λ = λs[1:r]
    return nothing
end

"""
    step(s::LowRankHessianStrategy, ctx) -> candidate_pulse

One low-rank Hessian iteration: (re)build the principal directions from the
model if due, measure the cost gradient along each direction on the device
(`2r` probe experiments), and return the damped-Newton candidate pulse. The
candidate trajectory (current controls plus the Newton step in the principal
subspace) is stashed for the chassis's acceptance interpolation.
"""
function step(s::LowRankHessianStrategy, ctx)
    t_build = 0.0
    rebuild =
        isnothing(s.V) || (s.refresh_every > 0 && (ctx.iter - 1) % s.refresh_every == 0)
    if rebuild
        t_build = @elapsed _build_directions!(s, ctx)
    end

    urows = ctx.z_ref.components[s.control_name]
    V, λ, w = s.V, s.λ, s.w_ref
    r = length(λ)
    h = s.fd_step

    t_probe = @elapsed begin
        g = Vector{Float64}(undef, r)
        for i = 1:r
            y_plus = run_experiment(
                s.experiment,
                _perturbed_pulse(ctx.qcp, ctx.z_ref, urows, h .* V[:, i]),
            )
            y_minus = run_experiment(
                s.experiment,
                _perturbed_pulse(ctx.qcp, ctx.z_ref, urows, -h .* V[:, i]),
            )
            g[i] =
                (
                    _lrh_cost(ctx.y_goal, y_plus, w, s.cost_mode) -
                    _lrh_cost(ctx.y_goal, y_minus, w, s.cost_mode)
                ) / (2h)
            s.n_probes += 2
        end

        μ = s.damping * λ[1]
        δu = V * (-(g) ./ (λ .+ μ))

        z_cand = deepcopy(ctx.z_ref)
        block = view(z_cand.data, urows, :)
        block .+= reshape(δu, length(urows), :)
        s.cand = z_cand
    end

    s.timings = (sysid = t_build, nlp = t_probe)
    ctx.verbose && @info "LowRankHessian step" iter = ctx.iter rank = r probes = 2r

    return extract_pulse(ctx.qcp.qtraj, s.cand)
end

@testitem "LowRankHessianStrategy: directions, probes, and convergence on Rabi mismatch" begin
    using Intonato
    using Intonato: step, prepare_strategy, candidate_trajectory, last_timings
    using LinearAlgebra

    # Single-qubit fixture with a 15% Rabi mismatch (mirrors the chassis test
    # items): the model is exact in structure, wrong in drive scale — the
    # low-rank method needs no sysid to correct the pulse.
    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    sys_nom = QuantumSystem(0.01 * σz, [σx], [1.0])
    sys_true = QuantumSystem(0.01 * σz, [1.15 * σx], [1.0])
    N = 11
    times = collect(range(0.0, 5.0, length = N))
    pulse = LinearSplinePulse(0.1 * ones(1, N), times)
    ψ0 = ComplexF64[1.0, 0.0]
    ψg = ComplexF64[0.0, 1.0]
    qcp = SplinePulseProblem(KetTrajectory(sys_nom, pulse, ψ0, ψg), N; Q = 100.0, R = 1e-2)
    model = MeasurementModel(:ψ̃, [populations], [N])
    experiment = SimulatedExperiment(KetTrajectory(sys_true, pulse, ψ0, ψg), model)

    strat = LowRankHessianStrategy(; fd_step = 1e-3, damping = 1e-3)
    ptp = PulseTuningProblem(qcp, experiment, model; strategy = strat)
    solve!(ptp; max_iter = 6, tol = 1e-8, verbose = false, min_nominal_fidelity = 0.0)

    hist = ptp.result.history
    @test length(hist) ≥ 2
    # The measured-gradient Newton step reduces the whitened cost by a large
    # factor from the uncalibrated start.
    @test hist[end].J_exp < 0.05 * hist[1].J_exp

    # Directions were built from the model, probes charged to the strategy.
    @test !isnothing(strat.V)
    @test length(strat.λ) ≥ 1
    @test strat.λ[1] > 0
    @test strat.n_probes ≥ 2 * length(strat.λ)
    # Populations of a 2-level ket at one knot ⇒ flat dim 2 ⇒ rank ≤ 2.
    @test length(strat.λ) ≤ 2

    # Contract hooks.
    @test candidate_trajectory(strat) !== nothing
    @test last_timings(strat).nlp ≥ 0.0
end

@testitem "LowRankHessianStrategy: rank cap and refresh rebuild" begin
    using Intonato
    using Intonato: step, prepare_strategy
    using LinearAlgebra

    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    sys_nom = QuantumSystem(0.01 * σz, [σx], [1.0])
    sys_true = QuantumSystem(0.01 * σz, [1.1 * σx], [1.0])
    N = 9
    times = collect(range(0.0, 4.0, length = N))
    pulse = LinearSplinePulse(0.12 * ones(1, N), times)
    ψ0 = ComplexF64[1.0, 0.0]
    ψg = ComplexF64[0.0, 1.0]
    qcp = SplinePulseProblem(KetTrajectory(sys_nom, pulse, ψ0, ψg), N; Q = 100.0, R = 1e-2)
    model = MeasurementModel(:ψ̃, [populations], [N])
    experiment = SimulatedExperiment(KetTrajectory(sys_true, pulse, ψ0, ψg), model)

    # n_modes caps the subspace at 1 even though the flat dim admits 2.
    strat = LowRankHessianStrategy(; n_modes = 1, refresh_every = 1)
    ptp = PulseTuningProblem(qcp, experiment, model; strategy = strat)
    solve!(ptp; max_iter = 2, tol = 0.0, verbose = false, min_nominal_fidelity = 0.0)

    @test length(strat.λ) == 1
    @test size(strat.V, 2) == 1
    # refresh_every = 1 ⇒ directions rebuilt each iteration ⇒ sysid time
    # recorded on the final iteration too.
    @test last_timings(strat).sysid > 0.0
    # Even rank-1 measured-gradient Newton makes progress on a pure Rabi error.
    hist = ptp.result.history
    @test hist[end].J_exp < hist[1].J_exp
end

@testitem "LowRankHessianStrategy: :linear cost mode on a goal_fidelity row (Liu configuration)" begin
    using Intonato
    using Intonato: step, prepare_strategy
    using LinearAlgebra

    # Drift-free qubit with two quadrature drives and an analytically exact
    # designed optimum: constant u_x with area π/2 transfers |0⟩ → −i|1⟩, so
    # the nominal pulse sits at an exact maximum of the (phase-invariant)
    # goal fidelity — the regime where the GN Hessian of a scalar fidelity
    # row is rank-one and vanishing while the exact Hessian carries the
    # defect Gram structure.
    σx = ComplexF64[0 1; 1 0]
    σy = ComplexF64[0 -im; im 0]
    sys_nom = QuantumSystem(zeros(ComplexF64, 2, 2), [σx, σy], [1.0, 1.0])
    sys_true = QuantumSystem(zeros(ComplexF64, 2, 2), [1.1 * σx, 1.1 * σy], [1.0, 1.0])
    N = 11
    T = 5.0
    times = collect(range(0.0, T, length = N))
    u0 = vcat(fill(π / (2T), 1, N), zeros(1, N))
    pulse = LinearSplinePulse(copy(u0), times)
    ψ0 = ComplexF64[1.0, 0.0]
    ψg = ComplexF64[0.0, 1.0]
    qcp = SplinePulseProblem(KetTrajectory(sys_nom, pulse, ψ0, ψg), N; Q = 100.0, R = 1e-2)
    model = MeasurementModel(:ψ̃, [goal_fidelity_at(ket_to_iso(ψg))], [N])
    experiment = SimulatedExperiment(KetTrajectory(sys_true, pulse, ψ0, ψg), model)

    # The Liu configuration: linear cost w(1 − F), exact model Hessian.
    strat = LowRankHessianStrategy(; cost_mode = :linear, rank_tol = 1e-6)
    ptp = PulseTuningProblem(qcp, experiment, model; strategy = strat)
    solve!(ptp; max_iter = 8, tol = 1e-14, verbose = false, min_nominal_fidelity = 0.0)

    # The exact Hessian at the design optimum is O(1) and low-rank: the ket
    # defect visible to |⟨g|ψ⟩|² has two channels (the goal-phase channel is
    # gauge and invisible).
    @test strat.λ[1] > 1e-2
    @test 2 ≤ length(strat.λ) ≤ 3
    @test strat.n_probes ≥ 2 * length(strat.λ)

    # Measured-gradient Newton against a 10% Rabi mismatch converges deep:
    # J_exp = (1 − F)² < 1e-10 ⇒ 1 − F < 1e-5.
    hist = ptp.result.history
    @test hist[end].J_exp < 1e-10
    @test hist[end].J_exp < 1e-6 * hist[1].J_exp

    # Contrast: the GN (:squared) build on the same fidelity row is
    # degenerate at the design optimum — its top curvature is roundoff-scale,
    # orders of magnitude below the exact Hessian's.
    strat_sq = LowRankHessianStrategy(; cost_mode = :squared)
    qcp_sq =
        SplinePulseProblem(KetTrajectory(sys_nom, pulse, ψ0, ψg), N; Q = 100.0, R = 1e-2)
    ptp_sq = PulseTuningProblem(qcp_sq, experiment, model; strategy = strat_sq)
    solve!(ptp_sq; max_iter = 1, tol = 0.0, verbose = false, min_nominal_fidelity = 0.0)
    @test strat_sq.λ[1] < 1e-8 * strat.λ[1]

    # Constructor guard.
    @test_throws ArgumentError LowRankHessianStrategy(; cost_mode = :not_a_mode)
end
