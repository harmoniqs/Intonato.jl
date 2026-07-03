# Tests for the whitened-cost noise statistics helpers (noise_stats.jl).

@testitem "noise floor and J statistics" begin
    using Intonato: noise_floor, debiased_cost, cost_std

    w = ones(3);
    σ2 = [0.1, 0.2, 0.0]            # third element deterministic
    @test noise_floor(w, σ2) ≈ 0.3               # tr(WΣWᵀ) = Σ wᵢ² σᵢ²

    r = [1.0, -1.0, 2.0]
    Ĵ = sum((w .* r) .^ 2)
    @test debiased_cost(Ĵ, w, σ2) ≈ Ĵ - 0.3

    # Var[Ĵ] = 2Σ(wᵢ²σᵢ²)² + 4Σ rᵢ² wᵢ⁴ σᵢ²  (plug-in r)
    @test cost_std(r, w, σ2)^2 ≈ 2 * (0.01 + 0.04) + 4 * (1 * 1 * 0.1 + 1 * 1 * 0.2 + 0.0)

    # all-deterministic ⇒ zero
    @test cost_std(r, w, zeros(3)) == 0.0

    # difference scale
    @test Intonato.diff_std(0.3, 0.4) ≈ hypot(0.3, 0.4)
end

@testitem "noise stats respect nonuniform weights" begin
    using Intonato: noise_floor, cost_std

    # Task-weighted element (w² = 40) dominates the floor the way the spec's
    # flagship σ_z×40 configuration predicts: floor = Σ wᵢ²σᵢ².
    w = [1.0, sqrt(40)]
    σ2 = [0.01, 0.01]
    @test noise_floor(w, σ2) ≈ 0.01 + 40 * 0.01

    # cost_std grows with the weighted residual magnitude.
    r_small = [0.1, 0.1]
    r_big = [1.0, 1.0]
    @test cost_std(r_big, w, σ2) > cost_std(r_small, w, σ2)
end
