# Tests for whiten (whitening.jl) — GLS weights + variances from the
# measurement model's noise types.

@testitem "whiten assembles W and Σ from the model" begin
    using Intonato

    g = x -> x[1:2]
    m_det = DeterministicMeasurement(g)
    m_shot = ShotNoiseMeasurement(g, 400, wigner_covariance)
    model = MeasurementModel(:ψ̃, AbstractMeasurement[m_shot, m_det], [5, 5])
    y_exp = [Measurement([0.9, 0.0], 5), Measurement([0.5, 0.5], 5)]
    w, σ2 = whiten(model, y_exp)

    # shot elements: σ² = (1-y²)/n, w = 1/σ ; deterministic: w = 1, σ² = 0
    @test σ2[1] ≈ (1 - 0.81) / 400
    @test w[1] ≈ 1 / sqrt(σ2[1])
    @test σ2[2] ≈ 1 / 400
    @test w[2] ≈ sqrt(400)
    @test σ2[3] == σ2[4] == 0.0 && w[3] == w[4] == 1.0

    # variance floor caps the weight near |y|→1
    wf, σ2f = whiten(
        model,
        [Measurement([1.0, 0.0], 5), Measurement([0.5, 0.5], 5)];
        var_floor = 0.25,
    )
    @test σ2f[1] ≈ 0.25 / 400
    @test wf[1] ≈ 1 / sqrt(0.25 / 400)

    # task weights compose multiplicatively
    wt, _ = whiten(model, y_exp; W_task = [1.0, 1.0, 1.0, sqrt(40)])
    @test wt[4] ≈ sqrt(40)
    @test wt[1] ≈ 1 / sqrt((1 - 0.81) / 400)
end

@testitem "whiten handles KnownCovarianceMeasurement (diagonal fast path)" begin
    using Intonato

    g = x -> x[1:2]
    Σ = [0.04 0.01; 0.01 0.09]     # off-diagonals dropped by the diagonal path
    m_known = KnownCovarianceMeasurement(g, Σ)
    model = MeasurementModel(:ψ̃, AbstractMeasurement[m_known], [3])
    y_exp = [Measurement([0.1, 0.2], 3)]

    w, σ2 = whiten(model, y_exp)
    @test σ2 ≈ [0.04, 0.09]
    @test w ≈ [1 / 0.2, 1 / 0.3]
end
