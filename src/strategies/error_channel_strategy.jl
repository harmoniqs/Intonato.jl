# ──── Error-channel-informed tuning strategy (Schuster 2026-07-21 direction ①a) ─
#
# Closed-loop calibration restricted to a *physically labelled*, low-dimensional
# subspace built by conjugating an ENUMERATED list of physical error operators
# {G_k} through the current pulse's trajectory. This is the error-channel-informed
# complement of `LowRankHessianStrategy`: instead of eigendecomposing the model
# cost Hessian (a SECOND derivative, O(M²) to assemble) to *discover* an
# unlabelled principal subspace, it takes the user's known error channels and
# *constructs* the calibration subspace from a FIRST derivative — the
# error-generator Jacobian
#
#     J[k, j] = ⟨ i·G_k , (∂U/∂u_j)·U† ⟩_HS / ‖G_k‖²                       (†)
#
# whose k-th row is "how strongly control knot-coordinate j excites physical
# error channel k, measured through the whole trajectory." The row space of J is
# the sensitive control subspace; each retained direction carries the label of
# the channel it predominantly moves. Because the user enumerates only the
# channels the device actually has, the resulting rank r can be far below the
# full Hessian rank ⇒ fewer device probes per iteration.
#
# The connection to `LowRankHessianStrategy`: {i·G_k} for a COMPLETE Hermitian
# basis spans every linear functional of the anti-Hermitian generator B_j =
# (∂U/∂u_j)·U†; the model cost gradient/Hessian is itself such a functional, so
# the row space of J contains the dominant Hessian eigenvectors exactly
# (captured fraction → 1). This is the prototype result: 7 dominant directions =
# 3 in-subspace Paulis + 4 leakage channels, recovered from the FIRST derivative
# with no Hessian assembly, each direction physically labelled — the QILC
# calibration payload.

"""
    ErrorChannelStrategy(error_ops; kwargs...) <: AbstractTuningStrategy

Measured-gradient damped-Newton confined to the *labelled* calibration subspace
built by conjugating an enumerated list of physical error operators through the
current pulse's trajectory. The error-channel-informed complement of
[`LowRankHessianStrategy`](@ref).

At the first `step` (and every `refresh_every` iterations if nonzero) the
strategy builds, from the nominal system + current pulse, the error-generator
Jacobian `J` of equation (†) by a central finite-difference sweep of the total
propagator (a FIRST derivative, `M` model rollouts — cheaper than the exact
Hessian's `O(M²)` second differences). Its rows are the control-space
directions `{d_k}` — the projection into control space of each error operator
conjugated through the trajectory. Orthonormalizing them (`svd`, keep singular
values `≥ rank_tol · σ_max`) gives the subspace `V` (`M × r`), each direction
labelled by the channel it predominantly excites (`strategy.dir_labels`).

Per outer iteration the strategy takes a **measured-gradient** damped-Newton
step confined to `span(V)`, exactly the `LowRankHessianStrategy` inner loop: it
*measures* the cost gradient on the device along each direction (central
differences, `2r` probe experiments), takes the model curvature `λ` along each
direction (a cheap central second difference of the model cost, model-side, no
device cost), and returns the damped Newton candidate
`δu = V ((−g) ./ (λ .+ μ))`, `μ = damping · max(λ)`.

**Measured gradient vs model gradient (design decision).** The gradient is
*measured*, not modelled — so the step direction is correct even when the model
is wrong; the model supplies only the *subspace* (from the error channels) and
the *curvature*, both robust to moderate mismatch. This mirrors
`LowRankHessianStrategy` (the whole point of closed-loop calibration: correct
the device without trusting the model's gradient) and lets the two strategies be
compared apples-to-apples — they differ only in how the subspace is sourced
(error-channel conjugation here, Hessian eigendecomposition there). A
model-gradient-projected step would have been simpler but would forfeit that
robustness, so it was not chosen.

# Arguments
- `error_ops::Vector{<:AbstractMatrix}`: the enumerated physical error
  operators `G_k` (Hermitian generators / jump operators), each a `d × d`
  matrix in the system's Hilbert space. The coherent error they model is
  `exp(-iε G_k)`.

# Keyword Arguments
- `labels::Vector{Symbol}`: a name per error operator (default `:G1, :G2, …`).
- `control_name::Symbol = :u`: trajectory component holding the tunable knots.
- `fd_step::Float64 = 1e-3`: device probe amplitude `h` along each direction.
- `model_fd_step::Float64 = 1e-6`: model-side finite-difference step for the
  propagator Jacobian (†).
- `model_curv_step::Float64 = 1e-3`: model-side second-difference step for the
  per-direction curvature (coarser: second differences divide roundoff by δ²).
- `n_substeps::Int = 100`: piecewise-constant substeps used to roll the pulse
  into the total propagator when conjugating the error operators.
- `n_modes::Union{Nothing,Int} = nothing`: cap on the number of directions
  (`nothing` ⇒ all above `rank_tol`).
- `rank_tol::Float64 = 1e-8`: keep directions with `σ ≥ rank_tol · σ_max`.
- `damping::Float64 = 1e-3`: relative Newton damping; `μ = damping · max(λ)`.
- `refresh_every::Int = 0`: rebuild directions every `n` outer iterations on the
  current pulse (`0` ⇒ build once).
- `cost_mode::Symbol = :squared`: strategy cost for the measured gradient —
  `:squared` (whitened residual norm²) for vector observables, `:linear`
  (whitened residual sum) for defect-scalar rows such as `goal_fidelity`. Same
  semantics as `LowRankHessianStrategy`.

# Probe accounting
The `2r` gradient probes per iteration are strategy-internal experiment calls;
`TuningResult.n_experiments` does not include them. The running total is
`strategy.n_probes`; per-iteration build/probe wall-clock is in `last_timings`
(`:sysid` = direction build, `:nlp` = probes + Newton).

# See also
[`error_channel_directions`](@ref) — the standalone direction builder;
[`LowRankHessianStrategy`](@ref) — the Hessian-eigenvector complement.
"""
mutable struct ErrorChannelStrategy <: AbstractTuningStrategy
    # ── config ──
    error_ops::Vector{Matrix{ComplexF64}}
    labels::Vector{Symbol}
    control_name::Symbol
    fd_step::Float64
    model_fd_step::Float64
    model_curv_step::Float64
    n_substeps::Int
    n_modes::Union{Nothing,Int}
    rank_tol::Float64
    damping::Float64
    refresh_every::Int
    cost_mode::Symbol
    # ── per-solve state ──
    experiment::Union{Nothing,AbstractExperiment}
    V::Union{Nothing,Matrix{Float64}}       # M × r control-space directions
    λ::Union{Nothing,Vector{Float64}}       # r model curvatures
    dir_labels::Union{Nothing,Vector{Symbol}}  # dominant channel label per direction
    w_ref::Union{Nothing,Vector{Float64}}   # whitening frozen at direction build
    cand::Any                               # candidate NamedTrajectory (or nothing)
    n_probes::Int
    timings::NamedTuple{(:sysid, :nlp),NTuple{2,Float64}}
end

function ErrorChannelStrategy(
    error_ops::AbstractVector;
    labels::AbstractVector{Symbol} = [Symbol("G", k) for k = 1:length(error_ops)],
    control_name::Symbol = :u,
    fd_step::Float64 = 1e-3,
    model_fd_step::Float64 = 1e-6,
    model_curv_step::Float64 = 1e-3,
    n_substeps::Int = 100,
    n_modes::Union{Nothing,Int} = nothing,
    rank_tol::Float64 = 1e-8,
    damping::Float64 = 1e-3,
    refresh_every::Int = 0,
    cost_mode::Symbol = :squared,
)
    isempty(error_ops) && throw(ArgumentError("error_ops must be non-empty"))
    length(labels) == length(error_ops) ||
        throw(ArgumentError("labels ($(length(labels))) must match error_ops ($(length(error_ops)))"))
    cost_mode in (:squared, :linear) ||
        throw(ArgumentError("cost_mode must be :squared or :linear, got :$cost_mode"))
    ops = Matrix{ComplexF64}[Matrix{ComplexF64}(G) for G in error_ops]
    for (k, G) in enumerate(ops)
        size(G, 1) == size(G, 2) ||
            throw(ArgumentError("error_ops[$k] must be square, got $(size(G))"))
    end
    return ErrorChannelStrategy(
        ops,
        collect(Symbol, labels),
        control_name,
        fd_step,
        model_fd_step,
        model_curv_step,
        n_substeps,
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
        nothing,
        0,
        (sysid = 0.0, nlp = 0.0),
    )
end

function prepare_strategy(s::ErrorChannelStrategy, ptp, z_ref; kwargs...)
    s.experiment = ptp.experiment
    s.V = nothing
    s.λ = nothing
    s.dir_labels = nothing
    s.w_ref = nothing
    s.cand = nothing
    s.n_probes = 0
    s.timings = (sysid = 0.0, nlp = 0.0)
    return s
end

candidate_trajectory(s::ErrorChannelStrategy) = s.cand
last_timings(s::ErrorChannelStrategy) = s.timings

# ──── Trajectory conjugation core ─────────────────────────────────────────────

# Total propagator of `pulse` through the (drift, drives) Hamiltonian by a
# fine piecewise-constant rollout: U = ∏ₘ exp(-i Δt H(u(tₘ))), midpoint sampled.
# Plain `LinearAlgebra.exp` (no AD needed — every derivative here is a finite
# difference).
function _propagator(
    Hd::Matrix{ComplexF64},
    Hc::Vector{Matrix{ComplexF64}},
    pulse::AbstractPulse,
    n_substeps::Int,
)
    times = get_knot_times(pulse)
    t0 = first(times)
    Tspan = last(times) - t0
    d = size(Hd, 1)
    U = Matrix{ComplexF64}(I, d, d)
    dt = Tspan / n_substeps
    @inbounds for m = 1:n_substeps
        t = t0 + (m - 0.5) * dt
        u = pulse(t)
        H = copy(Hd)
        for i in eachindex(Hc)
            H .+= u[i] .* Hc[i]
        end
        U = exp(-im * dt * H) * U
    end
    return U
end

# Error-generator Jacobian (†): J[k, j] = ⟨ i·Gₖ , (∂U/∂u_j)·U† ⟩_HS / ‖Gₖ‖².
# `pulse_from_delta(e::Vector)` returns the pulse with the flattened control
# perturbation `e` applied; `M` is the control-space dimension. Central FD in
# each coordinate — a FIRST derivative, `2M` propagator rollouts.
function _error_generator_jacobian(
    Hd::Matrix{ComplexF64},
    Hc::Vector{Matrix{ComplexF64}},
    pulse_ref::AbstractPulse,
    pulse_from_delta,
    M::Int,
    error_ops::Vector{Matrix{ComplexF64}},
    gnorm2::Vector{Float64},
    n_substeps::Int,
    δ::Float64,
)
    U_ref = _propagator(Hd, Hc, pulse_ref, n_substeps)
    r_ops = length(error_ops)
    # Pre-form the dual channel operators i·Gₖ (so ⟨i·Gₖ, B⟩ = Re tr((i·Gₖ)' B)).
    dual = Matrix{ComplexF64}[im .* G for G in error_ops]
    J = zeros(Float64, r_ops, M)
    e = zeros(M)
    for j = 1:M
        e[j] = δ
        Up = _propagator(Hd, Hc, pulse_from_delta(e), n_substeps)
        e[j] = -δ
        Um = _propagator(Hd, Hc, pulse_from_delta(e), n_substeps)
        e[j] = 0.0
        B = ((Up .- Um) ./ (2δ)) * U_ref'      # anti-Hermitian generator
        for k = 1:r_ops
            J[k, j] = real(tr(dual[k]' * B)) / gnorm2[k]
        end
    end
    return J, U_ref
end

# Orthonormalize the rows of J → control-space directions V (M × r), and label
# each retained direction by the channel it predominantly excites.
function _directions_from_jacobian(
    J::Matrix{Float64},
    labels::Vector{Symbol},
    rank_tol::Float64,
    n_modes::Union{Nothing,Int},
)
    Jt = Matrix(J')                    # M × r_ops
    F = svd(Jt)
    σmax = isempty(F.S) ? 0.0 : F.S[1]
    σmax = σmax > 0 ? σmax : one(σmax)
    r = count(≥(rank_tol * σmax), F.S)
    !isnothing(n_modes) && (r = min(r, n_modes))
    r = max(r, 1)
    V = F.U[:, 1:r]
    C = V' * Jt                        # r × r_ops correlation of directions ↔ channels
    dir_labels = Symbol[labels[argmax(abs.(view(C, i, :)))] for i = 1:r]
    return V, dir_labels
end

# Rebuild a spline pulse from perturbed knot values, preserving type + times.
_rebuild_pulse(p::LinearSplinePulse, u::AbstractMatrix) =
    LinearSplinePulse(collect(u), get_knot_times(p))
_rebuild_pulse(p::CubicSplinePulse, u::AbstractMatrix) =
    CubicSplinePulse(collect(u), get_knot_derivatives(p), get_knot_times(p))

"""
    error_channel_directions(system, pulse, error_ops; kwargs...)
        -> (; V, labels, J, U)

Standalone builder for the labelled calibration subspace: conjugate each
`error_ops[k]` through the `pulse` trajectory over `system`, forming the
error-generator Jacobian `J` (†) by central finite differences of the total
propagator w.r.t. each control knot value, and return its orthonormalized row
space `V` (`M × r`, `M = n_drives · n_knots`) with a channel `labels[i]` per
direction. `U` is the reference propagator.

This is the same construction `ErrorChannelStrategy` performs internally; it is
exposed for analysis (e.g. comparing the labelled subspace against the top
`LowRankHessianStrategy` eigenvectors).

# Keyword Arguments
`labels`, `n_substeps`, `fd_step` (Jacobian FD step), `rank_tol`, `n_modes` —
see [`ErrorChannelStrategy`](@ref).
"""
function error_channel_directions(
    system,
    pulse::AbstractPulse,
    error_ops::AbstractVector;
    labels::AbstractVector{Symbol} = [Symbol("G", k) for k = 1:length(error_ops)],
    n_substeps::Int = 100,
    fd_step::Float64 = 1e-6,
    rank_tol::Float64 = 1e-8,
    n_modes::Union{Nothing,Int} = nothing,
)
    Hd = Matrix{ComplexF64}(get_drift(system))
    Hc = Matrix{ComplexF64}[Matrix{ComplexF64}(H) for H in get_drives(system)]
    ops = Matrix{ComplexF64}[Matrix{ComplexF64}(G) for G in error_ops]
    gnorm2 = Float64[real(tr(G' * G)) for G in ops]
    u0 = get_knot_values(pulse)
    nd, N = size(u0)
    M = nd * N
    pulse_from_delta = e -> _rebuild_pulse(pulse, u0 .+ reshape(e, nd, N))
    J, U_ref = _error_generator_jacobian(
        Hd,
        Hc,
        pulse,
        pulse_from_delta,
        M,
        ops,
        gnorm2,
        n_substeps,
        fd_step,
    )
    V, dir_labels = _directions_from_jacobian(J, collect(Symbol, labels), rank_tol, n_modes)
    return (; V = V, labels = dir_labels, J = J, U = U_ref)
end

# Model-side strategy cost at a flattened control perturbation (no device cost).
function _ecs_model_cost(s::ErrorChannelStrategy, ctx, urows, δ::Vector{Float64}, w)
    y = predict(
        ctx.device_model,
        _perturbed_pulse(ctx.qcp, ctx.z_ref, urows, δ),
        ctx.measurement_model,
    )
    return _lrh_cost(ctx.y_goal, y, w, s.cost_mode)
end

# Build the labelled directions from the nominal system + current pulse, then
# the per-direction model curvature λ.
function _build_directions!(s::ErrorChannelStrategy, ctx)
    system = get_system(ctx.qcp.qtraj)
    d = size(get_drift(system), 1)
    for (k, G) in enumerate(s.error_ops)
        size(G, 1) == d || throw(
            DimensionMismatch(
                "error_ops[$k] is $(size(G,1))×$(size(G,1)) but the system Hilbert " *
                "dimension is $d",
            ),
        )
    end

    urows = ctx.z_ref.components[s.control_name]
    M = length(urows) * size(ctx.z_ref.data, 2)

    w = collect(Float64, ctx.w)
    s.w_ref = w

    Hd = Matrix{ComplexF64}(get_drift(system))
    Hc = Matrix{ComplexF64}[Matrix{ComplexF64}(H) for H in get_drives(system)]
    ops = s.error_ops
    gnorm2 = Float64[real(tr(G' * G)) for G in ops]
    pulse_from_delta = e -> _perturbed_pulse(ctx.qcp, ctx.z_ref, urows, e)

    J, _ = _error_generator_jacobian(
        Hd,
        Hc,
        ctx.pulse,
        pulse_from_delta,
        M,
        ops,
        gnorm2,
        s.n_substeps,
        s.model_fd_step,
    )
    V, dir_labels = _directions_from_jacobian(J, s.labels, s.rank_tol, s.n_modes)
    s.V = V
    s.dir_labels = dir_labels

    # Per-direction model curvature (central second difference of the model
    # cost). Floored positive so the damped Newton stays well-behaved on flat /
    # weakly non-convex directions; the chassis line search guards the rest.
    r = size(V, 2)
    δc = s.model_curv_step
    c0 = _ecs_model_cost(s, ctx, urows, zeros(M), w)
    λraw = Vector{Float64}(undef, r)
    for i = 1:r
        cp = _ecs_model_cost(s, ctx, urows, δc .* V[:, i], w)
        cm = _ecs_model_cost(s, ctx, urows, -δc .* V[:, i], w)
        λraw[i] = (cp + cm - 2c0) / δc^2
    end
    λmax = maximum(abs, λraw)
    floor = max(s.rank_tol * λmax, eps())
    s.λ = Float64[max(abs(λraw[i]), floor) for i = 1:r]
    return nothing
end

"""
    step(s::ErrorChannelStrategy, ctx) -> candidate_pulse

One error-channel iteration: (re)build the labelled directions from the nominal
model if due, measure the cost gradient along each direction on the device
(`2r` probe experiments), and return the damped-Newton candidate pulse confined
to `span(V)`.
"""
function step(s::ErrorChannelStrategy, ctx)
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

        μ = s.damping * maximum(λ)
        δu = V * (-(g) ./ (λ .+ μ))

        z_cand = deepcopy(ctx.z_ref)
        block = view(z_cand.data, urows, :)
        block .+= reshape(δu, length(urows), :)
        s.cand = z_cand
    end

    s.timings = (sysid = t_build, nlp = t_probe)
    ctx.verbose && @info "ErrorChannel step" iter = ctx.iter rank = r probes = 2r labels =
        s.dir_labels

    return extract_pulse(ctx.qcp.qtraj, s.cand)
end

@testitem "ErrorChannelStrategy: labelled directions span the top LowRankHessian eigenvectors" begin
    using Intonato
    using Intonato: error_channel_directions
    using LinearAlgebra

    # Single-qubit fixture (mirrors the LowRankHessian test item): the labelled
    # error-generator directions built from a COMPLETE Hermitian channel basis
    # must span the model cost Hessian's dominant subspace — the prototype's
    # "captured fraction ≈ 1" connection, here on a Rabi-mismatch fixture.
    σx = ComplexF64[0 1; 1 0]
    σy = ComplexF64[0 -im; im 0]
    σz = ComplexF64[1 0; 0 -1]
    I2 = ComplexF64[1 0; 0 1]
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

    # (1) Labelled error-channel subspace from a complete su(2)+I basis.
    ecd = error_channel_directions(
        sys_nom,
        pulse,
        [I2, σx, σy, σz];
        labels = [:I, :X, :Y, :Z],
        n_substeps = 400,
    )
    @test size(ecd.V, 1) == N                      # M = n_drives · N_knots = 1·11
    @test size(ecd.V, 2) ≥ 1
    @test length(ecd.labels) == size(ecd.V, 2)     # every direction is labelled

    # (2) The top LowRankHessian eigenvectors on the same fixture / pulse.
    lrh = LowRankHessianStrategy(; fd_step = 1e-3)
    ptp = PulseTuningProblem(qcp, experiment, model; strategy = lrh)
    solve!(ptp; max_iter = 1, tol = 0.0, verbose = false, min_nominal_fidelity = 0.0)
    Vh = lrh.V
    @test size(Vh, 2) ≥ 1

    # (3) Captured fraction: each dominant Hessian eigenvector lies (almost
    # entirely) in the labelled error-channel subspace. First derivative,
    # no Hessian assembly — yet it reproduces the Hessian subspace.
    P = ecd.V * ecd.V'                              # orthogonal projector onto span(V)
    captured = [norm(P * Vh[:, i])^2 for i = 1:size(Vh, 2)]
    @test minimum(captured) ≥ 0.99
end

@testitem "ErrorChannelStrategy: converges with fewer directions than LowRankHessian rank" begin
    using Intonato
    using Intonato: step, prepare_strategy, candidate_trajectory, last_timings
    using LinearAlgebra

    # 15% Rabi mismatch — the true error excites a SINGLE physical channel (the
    # σx drive over-rotation). Enumerating just that channel gives a rank-1
    # calibration subspace: fewer probe directions than the LowRankHessian
    # method's full rank, yet a measured-gradient Newton step still converges.
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

    strat = ErrorChannelStrategy(
        [σx];
        labels = [:X],
        fd_step = 1e-3,
        damping = 1e-3,
        n_substeps = 200,
    )
    ptp = PulseTuningProblem(qcp, experiment, model; strategy = strat)
    solve!(ptp; max_iter = 6, tol = 1e-8, verbose = false, min_nominal_fidelity = 0.0)

    hist = ptp.result.history
    @test length(hist) ≥ 2
    # Large cost reduction from the uncalibrated start.
    @test hist[end].J_exp < 0.05 * hist[1].J_exp

    # Directions built, labelled, and probed.
    @test !isnothing(strat.V)
    @test size(strat.V, 2) == 1                     # single enumerated channel ⇒ rank 1
    @test strat.dir_labels == [:X]
    @test strat.n_probes ≥ 2 * size(strat.V, 2)

    # Fewer directions than the LowRankHessian method on the same fixture
    # (populations at one knot ⇒ GN rank up to 2).
    lrh = LowRankHessianStrategy(; fd_step = 1e-3)
    qcp2 =
        SplinePulseProblem(KetTrajectory(sys_nom, pulse, ψ0, ψg), N; Q = 100.0, R = 1e-2)
    ptp2 = PulseTuningProblem(qcp2, experiment, model; strategy = lrh)
    solve!(ptp2; max_iter = 1, tol = 0.0, verbose = false, min_nominal_fidelity = 0.0)
    @test size(strat.V, 2) ≤ size(lrh.V, 2)

    # Contract hooks.
    @test candidate_trajectory(strat) !== nothing
    @test last_timings(strat).nlp ≥ 0.0
end

@testitem "ErrorChannelStrategy: 3-level leaky transmon — 7 labelled channels + closed-loop calibration" begin
    using Intonato
    using Intonato: error_channel_directions, _propagator, step
    using LinearAlgebra

    # ── Realistic leaky-transmon fixture (prototype parameters, rotating frame,
    #    GHz·ns ×2π). The gate-error space of a 2-level gate embedded in 3 levels
    #    is 7-dimensional: 3 in-subspace Paulis + 4 leakage channels. ───────────
    NL = 3
    δ = 0.20
    a = diagm(1 => sqrt.(1.0:(NL-1))) |> Matrix{ComplexF64}
    ad = collect(a')
    H0 = 2π * (-δ / 2 * ad * ad * a * a)
    HI = 2π * Matrix{ComplexF64}(a + ad)
    HQ = 2π * Matrix{ComplexF64}(im * (a - ad))
    sys = QuantumSystem(H0, [HI, HQ], [(-1.0, 1.0), (-1.0, 1.0)])

    # The enumerated 7 physical error operators (Hermitian), exactly the
    # prototype's basis: 3 in-subspace Paulis + 4 leakage generators.
    eij(i, j) = (M = zeros(ComplexF64, NL, NL); M[i, j] = 1; M)
    error_ops = [
        eij(1, 2) + eij(2, 1),          # X01
        im * (eij(2, 1) - eij(1, 2)),   # Y01
        eij(1, 1) - eij(2, 2),          # Z01
        eij(1, 3) + eij(3, 1),          # leakX0
        im * (eij(3, 1) - eij(1, 3)),   # leakY0
        eij(2, 3) + eij(3, 2),          # leakX1
        im * (eij(3, 2) - eij(2, 3)),   # leakY1
    ]
    labels = [:X01, :Y01, :Z01, :leakX0, :leakY0, :leakX1, :leakY1]

    # ── Optimize an X gate on the {|0>, |1>} subspace (real Piccolo solve). ─────
    N = 8
    T = 20.0
    times = collect(range(0.0, T, length = N))
    u0 = vcat(fill(1 / (2T), 1, N), zeros(1, N))
    pulse0 = LinearSplinePulse(copy(u0), times)
    goal = EmbeddedOperator(:X, [1, 2], 3)
    qcp_u = SplinePulseProblem(UnitaryTrajectory(sys, pulse0, goal), N; Q = 1e5, R = 1e-4)
    solve!(qcp_u; max_iter = 300, verbose = false, print_level = 0)
    pulse = get_pulse(qcp_u.qtraj)

    # ── (A) Subspace reproduction: the 7 labelled error-channel directions built
    #    from the FIRST-derivative conjugation span the SAME control subspace as
    #    the low-rank (Gauss–Newton) Hessian of the phase-invariant gate error —
    #    the prototype's 7-dimensional result (3 Paulis + 4 leakage). ───────────
    nsub = 60
    Hd = Matrix{ComplexF64}(H0)
    Hc = Matrix{ComplexF64}[HI, HQ]
    u0k = collect(get_knot_values(pulse))
    nd, Nk = size(u0k)
    M = nd * Nk
    Pc = ComplexF64[1 0 0; 0 1 0]

    # (A.1) The labelled 7-channel subspace from the error-op conjugation.
    ecd = error_channel_directions(sys, pulse, error_ops; labels = labels, n_substeps = nsub)
    r = size(ecd.V, 2)

    # (A.2) The "ground-truth" low-rank Hessian subspace: right-singular vectors
    #   of the phase-invariant comp-subspace unitary Jacobian J[j] = ∂(P U P')/∂u_j
    #   with the (gauge) global-phase direction i·U_c projected out. The
    #   Gauss–Newton Hessian is JᵀJ, so these ARE the low-rank-Hessian
    #   eigenvectors — computed at THIS pulse, no optimum required. Its rank is
    #   the gate-error dimension: 7 (8 comp-subspace real DOFs minus global phase).
    U_ref = _propagator(Hd, Hc, pulse, nsub)
    Uc_ref = Pc * U_ref * Pc'
    gph = im .* Uc_ref                                  # global-phase generator on U_c
    gph_n2 = real(sum(abs2, gph))
    δ = 1e-6
    JFr = zeros(Float64, 8, M)                          # 2×2 comp block = 4 complex = 8 real
    e = zeros(M)
    for j = 1:M
        e[j] = δ
        Up = _propagator(Hd, Hc, LinearSplinePulse(u0k .+ reshape(e, nd, Nk), times), nsub)
        e[j] = -δ
        Um = _propagator(Hd, Hc, LinearSplinePulse(u0k .+ reshape(e, nd, Nk), times), nsub)
        e[j] = 0.0
        dUc = Pc * ((Up .- Um) ./ (2δ)) * Pc'
        dUc = dUc .- (real(sum(conj(gph) .* dUc)) / gph_n2) .* gph   # project out global phase
        v = vec(dUc)
        JFr[1:4, j] = real.(v)
        JFr[5:8, j] = imag.(v)
    end
    Fsvd = svd(Matrix(JFr'))                            # M × 8; U columns = u-space sing vecs
    σ = Fsvd.S
    ndom = count(≥(1e-6 * σ[1]), σ)                     # gate-error dimension ≈ 7
    Vh = Fsvd.U[:, 1:ndom]                              # low-rank Hessian subspace (M × ndom)

    @info "3-level transmon subspace" rank_V = r labels = ecd.labels gate_error_dim = ndom

    # The gate-error space is 7-dimensional; a generic 2-quadrature pulse excites
    # (up to) all 7 channels. Both the labelled subspace and the low-rank Hessian
    # subspace land on that space.
    @test 6 ≤ r ≤ 7
    @test 6 ≤ ndom ≤ 7
    @test any(l -> occursin("leak", String(l)), ecd.labels)   # leakage channels present

    # Captured fraction vs the 2×2 comp-block Jacobian ∂(P U P')/∂u.
    # NOTE (informational, NOT asserted): the block Jacobian is not a valid
    # first-order reference for the LEAKAGE channels. A leakage generator moves
    # population OUT of the computational block — it changes U outside the block
    # and is only *second*-order-visible in the endpoint 2×2 block — so this
    # reference under-captures the 4 leakage directions (min ≈ 0.48) even though
    # the labelled subspace is correct and full-rank (7). The prototype's
    # "captured ≈ 1" is against the FULL error-generator Jacobian / infidelity
    # Hessian; the meaningful, independent chassis check here is the
    # LowRankHessianStrategy capture at the end of this testitem (asserted ≥ 0.95).
    P = ecd.V * ecd.V'
    ncmp = min(ndom, r)
    captured = [norm(P * Vh[:, i])^2 for i = 1:ncmp]
    @info "captured fraction vs 2×2-block Jacobian (leakage under-captured by construction)" ncmp min =
        minimum(captured) mean = sum(captured) / length(captured)
    @test ncmp ≥ 6

    # ── (B) Closed-loop calibration of a KNOWN injected device error. The true
    #    device has an over-rotation (5% drive scale) AND excess anharmonicity,
    #    which drives leakage into |2>. Enumerating the 7 channels, the strategy
    #    calibrates it through the leakage-sensitive populations observable. ─────
    ψ0 = ComplexF64[1.0, 0.0, 0.0]
    ψg = ComplexF64[0.0, 1.0, 0.0]
    δ_true = 0.16
    H0_true = 2π * (-δ_true / 2 * ad * ad * a * a)
    sys_true = QuantumSystem(H0_true, [1.05 * HI, 1.05 * HQ], [(-1.0, 1.0), (-1.0, 1.0)])

    qcp_k = SplinePulseProblem(KetTrajectory(sys, pulse, ψ0, ψg), N; Q = 100.0, R = 1e-2)
    model = MeasurementModel(:ψ̃, [populations], [N])   # 3 populations at final knot; P2 = leakage
    experiment = SimulatedExperiment(KetTrajectory(sys_true, pulse, ψ0, ψg), model)

    strat = ErrorChannelStrategy(
        error_ops;
        labels = labels,
        fd_step = 1e-3,
        damping = 1e-2,
        n_substeps = nsub,
    )
    ptp = PulseTuningProblem(qcp_k, experiment, model; strategy = strat)
    solve!(ptp; max_iter = 8, tol = 1e-10, verbose = false, min_nominal_fidelity = 0.0)

    hist = ptp.result.history
    @info "closed-loop calibration" J_start = hist[1].J_exp J_end = hist[end].J_exp rank =
        size(strat.V, 2) labels = strat.dir_labels
    @test length(hist) ≥ 2
    @test hist[end].J_exp < 0.2 * hist[1].J_exp     # calibrates the injected error
    @test !isnothing(strat.V)
    @test length(strat.dir_labels) == size(strat.V, 2)
    @test strat.n_probes ≥ 2 * size(strat.V, 2)

    # Same-fixture comparison against LowRankHessianStrategy: the labelled
    # error-channel subspace contains the low-rank Hessian's principal subspace.
    lrh = LowRankHessianStrategy(; fd_step = 1e-3)
    qcp_k2 = SplinePulseProblem(KetTrajectory(sys, pulse, ψ0, ψg), N; Q = 100.0, R = 1e-2)
    ptp2 = PulseTuningProblem(qcp_k2, experiment, model; strategy = lrh)
    solve!(ptp2; max_iter = 1, tol = 0.0, verbose = false, min_nominal_fidelity = 0.0)
    Pecs = strat.V * strat.V'
    cap_lrh = [norm(Pecs * lrh.V[:, i])^2 for i = 1:size(lrh.V, 2)]
    @info "captured LowRankHessian subspace" cap = cap_lrh
    @test minimum(cap_lrh) ≥ 0.95
end

@testitem "ErrorChannelStrategy: constructor guards + refresh rebuild" begin
    using Intonato
    using Intonato: step, prepare_strategy
    using LinearAlgebra

    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]

    # Constructor guards.
    @test_throws ArgumentError ErrorChannelStrategy(Matrix{ComplexF64}[])
    @test_throws ArgumentError ErrorChannelStrategy([σx]; cost_mode = :not_a_mode)
    @test_throws ArgumentError ErrorChannelStrategy([σx]; labels = [:X, :Y])

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

    strat = ErrorChannelStrategy([σx]; labels = [:X], refresh_every = 1, n_substeps = 200)
    ptp = PulseTuningProblem(qcp, experiment, model; strategy = strat)
    solve!(ptp; max_iter = 2, tol = 0.0, verbose = false, min_nominal_fidelity = 0.0)

    @test size(strat.V, 2) == 1
    # refresh_every = 1 ⇒ directions rebuilt each iteration ⇒ sysid recorded.
    @test last_timings(strat).sysid > 0.0
    hist = ptp.result.history
    @test hist[end].J_exp < hist[1].J_exp
end
