# Tests for the binomial Fisher / IRLS measurement weighting (fisher_irls.jl).
#
# Statistical tests against ANALYTIC Fisher information for canonical binomial
# cases: the closed form n/(p(1-p)) for known (p, n), the IRLS convergence to
# the analytic MLE, and the chassis composition through `W_task`.

@testitem "binomial_fisher_information: closed form for canonical binomial cases" begin
    using Intonato

    # textbook closed form: I(p) = n/(p(1-p))
    @test binomial_fisher_information(0.25, 100) ≈ 100 / (0.25 * 0.75)
    @test binomial_fisher_information(0.5, 1000) == 4000.0      # minimum at p = 0.5
    @test binomial_fisher_information(0.3, 400) ≈ 400 / 0.21

    # the asymptotic MLE variance is the inverse: p̂ = k/n has Var = p(1-p)/n
    p, n = 0.3, 400
    @test 1 / binomial_fisher_information(p, n) ≈ p * (1 - p) / n

    # count data (multinomial outcomes): same closed form per outcome through
    # the marginal binomial statistic
    ps = [0.7, 0.3]
    @test [binomial_fisher_information(pⱼ, 500) for pⱼ in ps] ≈ [500 / (0.7 * 0.3), 500 / (0.3 * 0.7)]

    # saturated limits are the exact Inf (floor at the weights level)
    @test binomial_fisher_information(0.0, 100) == Inf
    @test binomial_fisher_information(1.0, 100) == Inf

    # guards: nonsense probability / shot count
    @test_throws ArgumentError binomial_fisher_information(-0.1, 100)
    @test_throws ArgumentError binomial_fisher_information(1.5, 100)
    @test_throws ArgumentError binomial_fisher_information(0.5, 0)
end

@testitem "binomial_fisher_weights from a measurement model's shot statistics alone" begin
    using Intonato
    using LinearAlgebra

    # ── Rademacher assignment (Pauli expectations): y = 2p - 1 ──────────────
    # w² = I_y = n/(1-y²) — the p-space closed form n/(p(1-p)) with the p↔y
    # Jacobian absorbed by the covariance function's own convention.
    m_pauli = ShotNoiseMeasurement(x -> x, 1000, pauli_covariance)
    y = [Measurement([0.6], 1)]
    w = binomial_fisher_weights(MeasurementModel(:ψ̃, [m_pauli], [1]), y)
    @test w[1] ≈ sqrt(1000 / (1 - 0.6^2))
    # cross-check against the p-space closed form transformed to y-space:
    # sqrt(I_p) · |dp/dy| = sqrt(n/(p(1-p))) / 2 with p = (1+y)/2 = 0.8
    @test w[1] ≈ sqrt(binomial_fisher_information(0.8, 1000)) / 2

    # ── proportions (populations / counts): y = p ───────────────────────────
    m_pop = ShotNoiseMeasurement(x -> x, 400, population_covariance)
    p = [0.3, 0.7]
    w_pop = binomial_fisher_weights(MeasurementModel(:ψ̃, [m_pop], [1]), [Measurement(p, 1)])
    @test w_pop ≈ sqrt.(binomial_fisher_information.(p, 400))
    @test w_pop ≈ [sqrt(400 / (0.3 * 0.7)), sqrt(400 / (0.7 * 0.3))]

    # ── Wigner / displaced parity: same Rademacher form, different physics ──
    m_wig = ShotNoiseMeasurement(x -> x, 500, wigner_covariance)
    w_wig = binomial_fisher_weights(
        MeasurementModel(:ψ̃, [m_wig], [1]),
        [Measurement([-0.4], 1)],
    )
    @test w_wig[1] ≈ sqrt(500 / (1 - 0.16))

    # ── a lab's OWN binomial-form covariance_fn (no preset coupling) ────────
    # shot statistics alone: any binomial-form covariance works, not just the
    # bundled presets — the A1 extensibility story.
    lab_cov = (y, n) -> Diagonal(y .* (1 .- y) ./ n)   # count statistics, y = p
    m_lab = ShotNoiseMeasurement(x -> x, 250, lab_cov)
    w_lab =
        binomial_fisher_weights(MeasurementModel(:ψ̃, [m_lab], [1]), [Measurement([0.2], 1)])
    @test w_lab[1] ≈ sqrt(binomial_fisher_information(0.2, 250))

    # ── mixed model: flattened element order (measurement-major), as whiten ─
    m_det = DeterministicMeasurement(x -> x)
    model = MeasurementModel(:ψ̃, AbstractMeasurement[m_pauli, m_pop, m_det], [1, 1, 1])
    y_fit = [Measurement([0.6], 1), Measurement([0.3, 0.7], 1), Measurement([0.9], 1)]
    w_mixed = binomial_fisher_weights(model, y_fit)
    @test w_mixed ≈ [
        sqrt(1000 / (1 - 0.36)),
        sqrt(400 / 0.21),
        sqrt(400 / 0.21),
        1.0,                        # deterministic: no shot statistics, w = 1
    ]

    # ── KnownCovarianceMeasurement: plug-in diagonal (Fisher iff Σ binomial) ─
    m_known = KnownCovarianceMeasurement(x -> x, [0.04 0.01; 0.01 0.09])
    w_known = binomial_fisher_weights(
        MeasurementModel(:ψ̃, [m_known], [1]),
        [Measurement([0.1, 0.2], 1)],
    )
    @test w_known ≈ [1 / 0.2, 1 / 0.3]

    # ── var_floor: saturated fit stays finite; 0 restores the raw form ──────
    m_sat = ShotNoiseMeasurement(x -> x, 400, population_covariance)
    mmodel_sat = MeasurementModel(:ψ̃, [m_sat], [1])
    w_floor = binomial_fisher_weights(mmodel_sat, [Measurement([1.0], 1)])
    @test w_floor[1] ≈ 1 / sqrt(0.05 / 400)
    @test binomial_fisher_weights(mmodel_sat, [Measurement([1.0], 1)]; var_floor = 0.0)[1] ==
          Inf

    # ── Fisher consistency: w² · Var = 1 at interior fits ───────────────────
    σ2 = diag(pauli_covariance([0.6], 1000))[1]
    @test w[1]^2 * σ2 ≈ 1.0

    # ── dimension mismatch is a hard error ──────────────────────────────────
    @test_throws ErrorException binomial_fisher_weights(
        MeasurementModel(:ψ̃, [m_pauli, m_pop], [1, 1]),
        [Measurement([0.6], 1)],
    )
end

@testitem "IRLS Fisher weighting reproduces the analytic binomial MLE (identity link)" begin
    using Intonato

    # Canonical binomial case: k ~ Bin(n, p), identity link (proportion
    # statistic). The Fisher-scoring step expressed through our weights,
    #   p_new = p + score(p) / w(p)²,   score(p) = k/p - (n-k)/(1-p),
    # lands EXACTLY on the analytic MLE p̂ = k/n in one step from any
    # interior fit — the analytic Fisher-information solution.
    n = 400
    k = 121
    p̂ = k / n                      # 0.3025

    g = x -> collect(Float64, x)   # identity readout of the fitted probability
    model = MeasurementModel(:ψ̃, [ShotNoiseMeasurement(g, n, population_covariance)], [1])

    p_fit = 0.5                    # any interior fit
    w = binomial_fisher_weights(model, [Measurement([p_fit], 1)])
    score = k / p_fit - (n - k) / (1 - p_fit)
    p_new = p_fit + score / abs2(w[1])
    @test p_new ≈ p̂ rtol = 1e-12

    # the IRLS fixed point: weights at the converged fit equal the closed-form
    # Fisher weights at p̂
    w_conv = binomial_fisher_weights(model, [Measurement([p̂], 1)])
    @test w_conv[1] ≈ sqrt(binomial_fisher_information(p̂, n)) rtol = 1e-12

    # Fisher information × asymptotic MLE variance = 1
    @test abs2(w_conv[1]) * (p̂ * (1 - p̂) / n) ≈ 1.0 rtol = 1e-12
end

@testitem "IRLS on a canonical binomial GLM (logistic link) converges to the MLE" begin
    using Intonato
    using LinearAlgebra

    # Binomial GLM: kᵢ ~ Bin(n, μᵢ), logistic link μ = logistic(Xβ). The IRLS
    # update with the Fisher weighting must converge to the MLE (score ≈ 0),
    # and the converged μ-space weights must reproduce the analytic Fisher
    # information in η-space: I_η = Xᵀ diag(n μ(1-μ)) X.
    n = 200
    X = [1.0 0.0; 1.0 1.0; 1.0 2.0; 1.0 3.0]
    β_true = [-0.5, 0.8]
    μ_true = 1.0 ./ (1.0 .+ exp.(-(X * β_true)))
    k = round.(Int, n .* μ_true)          # "observed" counts (deterministic)
    p_obs = k ./ n

    logistic(η) = 1 / (1 + exp(-η))
    # The measurement function reads the linear predictor; the weights touch
    # only the shot statistics (n_shots + covariance_fn) — never the physics.
    model = MeasurementModel(
        :ψ̃,
        [ShotNoiseMeasurement(x -> logistic.(x), n, population_covariance)],
        [1],
    )

    β = let β = zeros(2)                  # cold start (hard local scope: the
        for _ = 1:50                     # IRLS loop rebinds β each iteration)
            η_fit = X * β
            μ_fit = logistic.(η_fit)
            w = binomial_fisher_weights(model, [Measurement(μ_fit, 1)])  # at the FIT
            # η-space working weights: W_η = W_μ·(dμ/dη)² = n·μ(1-μ)
            wη2 = abs2.(w) .* abs2.(μ_fit .* (1 .- μ_fit))
            # working response of the canonical-link IRLS update
            z = η_fit .+ (p_obs .- μ_fit) ./ (μ_fit .* (1 .- μ_fit))
            β = (X' * (Diagonal(wη2) * X)) \ (X' * (Diagonal(wη2) * z))
        end
        β
    end

    # MLE criterion: the likelihood score vanishes at the IRLS fixed point
    μ_final = logistic.(X * β)
    score = X' * (k .- n .* μ_final)
    @test maximum(abs, score) < 1e-8

    # the converged weights carry the analytic Fisher information
    w_final = binomial_fisher_weights(model, [Measurement(μ_final, 1)])
    @test abs2.(w_final) .* (μ_final .* (1 .- μ_final) ./ n) ≈ ones(4) rtol = 1e-10
    I_η = X' * Diagonal(abs2.(w_final) .* abs2.(μ_final .* (1 .- μ_final))) * X
    I_analytic = X' * Diagonal(n .* μ_final .* (1 .- μ_final)) * X
    @test I_η ≈ I_analytic rtol = 1e-10
end

@testitem "Fisher weighting composes into PulseTuningProblem as W_task (stub model)" begin
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

    # The lab's shot-statistics model (binomial/count populations) provides
    # the Fisher weighting, evaluated at the model-side fit.
    shot_model = MeasurementModel(:ψ̃, [pop(; n_shots = 400)], [N])
    y_fit = model_predict(qcp.prob.trajectory, shot_model)
    w_fisher = binomial_fisher_weights(shot_model, y_fit)
    @test all(isfinite, w_fisher)   # saturated fits (|p|→0/1) floored, not Inf

    # Stub measurement model: a deterministic readout — the Fisher weighting
    # is the ONLY weighting in the chassis cost (no plug-in whitening stacks),
    # and the strategy seam is untouched (default IdentityStrategy).
    stub_model = MeasurementModel(:ψ̃, [populations], [N])
    experiment = SimulatedExperiment(KetTrajectory(sys_rabi, pulse, ψ0, ψg), stub_model)
    ptp = PulseTuningProblem(qcp, experiment, stub_model; W_task = w_fisher)
    solve!(
        ptp;
        max_iter = 1,
        line_search = false,
        verbose = false,
        min_nominal_fidelity = 0.0,
        tol = 0.0,
    )
    @test length(ptp.result.history) == 1

    # one chassis step ran with the Fisher-weighted cost Σ w²r²
    y_goal = model_predict(qcp.prob.trajectory, stub_model)
    y_exp = run_experiment(experiment, extract_pulse(qcp.qtraj, qcp.prob.trajectory))
    r = y_goal[1].data .- y_exp[1].data
    @test ptp.result.history[1].J_exp ≈ sum(abs2, w_fisher .* r)

    # On a model that already carries shot statistics the chassis multiplies
    # W_task with its own plug-in whitening (W = W_task·Σ^{-1/2}) — the
    # documented composition contract, exercised here end to end.
    qcp2 = SplinePulseProblem(KetTrajectory(sys_nom, pulse, ψ0, ψg), N; Q = 100.0, R = 1e-2)
    experiment_shot =
        SimulatedExperiment(KetTrajectory(sys_rabi, pulse, ψ0, ψg), shot_model)
    ptp2 = PulseTuningProblem(qcp2, experiment_shot, shot_model; W_task = w_fisher)
    solve!(
        ptp2;
        max_iter = 1,
        line_search = false,
        verbose = false,
        min_nominal_fidelity = 0.0,
        tol = 0.0,
    )
    y_goal2 = model_predict(qcp2.prob.trajectory, shot_model)
    y_exp2 =
        run_experiment(experiment_shot, extract_pulse(qcp2.qtraj, qcp2.prob.trajectory))
    w_plugin, σ2 = whiten(shot_model, y_exp2)
    r2 = y_goal2[1].data .- y_exp2[1].data
    @test ptp2.result.history[1].J_hat ≈ sum(abs2, w_fisher .* r2 ./ sqrt.(σ2))
end

@testitem "Fisher/IRLS docstrings carry the A1 boundary note" begin
    using Intonato

    d_w = string(@doc Intonato.binomial_fisher_weights)
    @test occursin("A1", d_w)
    @test occursin(r"public"i, d_w)
    @test occursin(r"internal"i, d_w)

    d_i = string(@doc Intonato.binomial_fisher_information)
    @test occursin("A1", d_i)
end

@testitem "public purity: no private-package imports (standing pin)" begin
    using Intonato

    # The A1 thin split's purity pin: the public measurement surface never
    # imports the private packages. Checked against the dependency manifest
    # and every source file, so the pin holds structurally, not by review.
    private_packages = ("Piccolissimo", "Intonatissimo", "Altissimo")
    root = pkgdir(Intonato)

    proj = read(joinpath(root, "Project.toml"), String)
    for name in private_packages
        @test !occursin(name, proj)
    end

    for dir in ("src", "ext"),
        (branch, _, files) in walkdir(joinpath(root, dir)),
        f in files

        endswith(f, ".jl") || continue
        src = read(joinpath(branch, f), String)
        for name in private_packages
            @test !occursin(Regex("(using|import)\\s+$name"), src)
        end
    end
end
