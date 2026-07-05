# Tests for propagated_covariance / whiten_block / mahalanobis.

@testitem "propagated_covariance assembles S Σθ Sᵀ + Σmeas" begin
    using Intonato, LinearAlgebra
    S = [1.0 0.0; 0.5 2.0; 0.0 1.0]          # 3 rows, 2 params
    Σθ = [0.04 0.0; 0.0 0.09]
    Σmeas = Diagonal([1e-3, 2e-3, 3e-3])
    Σr = propagated_covariance(S, Σθ, Σmeas)
    @test size(Σr) == (3, 3)
    @test issymmetric(Σr)
    @test Σr[1, 1] ≈ 0.04 + 1e-3
    @test Σr[2, 2] ≈ 0.25*0.04 + 4*0.09 + 2e-3
    @test Σr[1, 2] ≈ 0.5*0.04                # cross-correlation from shared param 1
    # dimension checks
    @test_throws ErrorException propagated_covariance(S, [0.1 0.0; 0.0 0.1; 0.0 0.0], Σmeas)
end

@testitem "whiten_block whitens to identity covariance" begin
    using Intonato, LinearAlgebra, Statistics, Random
    Σ = [0.04 0.01 0.0; 0.01 0.09 0.02; 0.0 0.02 0.16]
    M = whiten_block(Σ)
    # Cov(Mν) = M Σ Mᵀ = I when Cov(ν)=Σ
    @test M * Σ * M' ≈ I(3) atol = 1e-10
    # empirical check
    rng = MersenneTwister(0); L = cholesky(Symmetric(Σ)).L
    samples = [M * (L * randn(rng, 3)) for _ = 1:20000]
    C = cov(reduce(hcat, samples)'; dims = 1)
    @test isapprox(C, I(3); atol = 0.05)
    # task weights scale rows
    Mt = whiten_block(Σ; W_task = [1.0, 2.0, 3.0])
    @test Mt[2, :] ≈ 2 .* M[2, :]
end

@testitem "mahalanobis is χ²-calibrated under the true covariance" begin
    using Intonato, LinearAlgebra, Statistics, Random
    S = [0.9 0.1; 0.2 0.8]; Σθ = Diagonal([0.02, 0.03]); Σmeas = Diagonal([1e-3, 1e-3])
    Σr = propagated_covariance(S, Matrix(Σθ), Σmeas)
    rng = MersenneTwister(1); L = cholesky(Symmetric(Matrix(Σr))).L
    # innovations drawn with the TRUE covariance → mean χ² ≈ dof = 2
    χ2 = [mahalanobis(L * randn(rng, 2), Σr) for _ = 1:40000]
    @test isapprox(mean(χ2), 2.0; atol = 0.1)
    # whitening by the SHOT-ONLY covariance (ignoring S Σθ Sᵀ) over-fires (χ² ≫ dof)
    χ2_shot = [mahalanobis(L * randn(rng, 2), Matrix(Σmeas)) for _ = 1:40000]
    @test mean(χ2_shot) > 5 * 2          # decisively over-dispersed
end
