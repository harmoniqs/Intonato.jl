# This file contains tests for the surrounding source directory.

# ============================================================================ #
# Tests for measurement_models.jl
# ============================================================================ #

@testitem "MeasurementModel with AbstractMeasurement" begin
    using Intonato
    using LinearAlgebra

    σx_iso = operator_to_iso_operator(ComplexF64[0 1; 1 0])
    σy_iso = operator_to_iso_operator(ComplexF64[0 -im; im 0])
    σz_iso = operator_to_iso_operator(ComplexF64[1 0; 0 -1])

    N = 11

    # New API: pass AbstractMeasurement directly
    m = pauli([σx_iso, σy_iso, σz_iso]; n_shots=1000)
    model = MeasurementModel(:ψ̃, [m], [N])
    @test model.measurements[1] isa ShotNoiseMeasurement
    @test length(model.measurements) == 1
    @test model.indices == [N]

    # Backward compat: bare Function auto-wraps
    model_compat = MeasurementModel(:ψ̃, [populations], [N])
    @test model_compat.measurements[1] isa DeterministicMeasurement

    # model_predict still works
    ψ_iso = Intonato.ket_to_iso(ComplexF64[1.0, 0.0])
    traj = NamedTrajectory(
        (ψ̃ = repeat(ψ_iso, 1, N), u = zeros(1, N), Δt = fill(0.1, 1, N));
        timestep=:Δt, controls=(:u,)
    )
    y = model_predict(traj, model_compat)
    @test length(y) == 1
    @test y[1].data ≈ [1.0, 0.0]  # populations of |0⟩
end

# ============================================================================ #
# Tests for noise_models.jl
# ============================================================================ #

@testitem "pauli_covariance formula" begin
    using Intonato
    using LinearAlgebra

    # Bloch vector [0.5, -0.3, 0.8], 1000 shots
    y = [0.5, -0.3, 0.8]
    n = 1000
    Σ = pauli_covariance(y, n)

    @test Σ isa Diagonal
    @test size(Σ) == (3, 3)
    @test Σ[1,1] ≈ (1 - 0.25) / 1000   # (1 - 0.5²) / 1000
    @test Σ[2,2] ≈ (1 - 0.09) / 1000   # (1 - 0.3²) / 1000
    @test Σ[3,3] ≈ (1 - 0.64) / 1000   # (1 - 0.8²) / 1000

    # Edge: y = ±1 → variance → 0
    Σ_edge = pauli_covariance([1.0, -1.0, 0.0], 100)
    @test Σ_edge[1,1] ≈ 0.0 atol=1e-15
    @test Σ_edge[2,2] ≈ 0.0 atol=1e-15
    @test Σ_edge[3,3] ≈ 1/100  # maximum variance at equator
end

@testitem "population_covariance formula" begin
    using Intonato
    using LinearAlgebra

    # Qubit populations [0.7, 0.3], 500 shots
    p = [0.7, 0.3]
    n = 500
    Σ = population_covariance(p, n)

    @test size(Σ) == (2, 2)
    @test Σ[1,1] ≈ 0.7 * 0.3 / 500     # p(1-p)/N
    @test Σ[2,2] ≈ 0.3 * 0.7 / 500
    @test Σ[1,2] ≈ -0.7 * 0.3 / 500     # -p_j*p_k/N
    @test issymmetric(Σ)
end


@testitem "AbstractMeasurement callable" begin
    using Intonato

    # All subtypes should be callable
    g = x -> x .^ 2
    x = [1.0, 2.0, 3.0]

    det = DeterministicMeasurement(g)
    @test det(x) == [1.0, 4.0, 9.0]

    sn = ShotNoiseMeasurement(g, 100, pauli_covariance)
    @test sn(x) == [1.0, 4.0, 9.0]

    kc = KnownCovarianceMeasurement(g, zeros(3,3))
    @test kc(x) == [1.0, 4.0, 9.0]
end

@testitem "measurement presets" begin
    using Intonato
    using LinearAlgebra

    σx_iso = operator_to_iso_operator(ComplexF64[0 1; 1 0])
    σy_iso = operator_to_iso_operator(ComplexF64[0 -im; im 0])
    σz_iso = operator_to_iso_operator(ComplexF64[1 0; 0 -1])
    ops = [σx_iso, σy_iso, σz_iso]

    # pauli without n_shots → Deterministic
    m_det = pauli(ops)
    @test m_det isa DeterministicMeasurement

    # pauli with n_shots → ShotNoise
    m_sn = pauli(ops; n_shots=1000)
    @test m_sn isa ShotNoiseMeasurement
    @test m_sn.n_shots == 1000

    # Both should be callable and return same values
    x = Intonato.ket_to_iso(ComplexF64[1.0, 0.0])  # |0⟩
    @test m_det(x) ≈ m_sn(x)

    # pop presets
    p_det = pop()
    @test p_det isa DeterministicMeasurement
    p_sn = pop(; n_shots=500)
    @test p_sn isa ShotNoiseMeasurement
    @test p_sn.n_shots == 500
end

# ============================================================================ #
# Tests for the free-phase fidelity helper (experiments.jl)
# ============================================================================ #

@testitem "phase_max_fidelity recovers free-phase overlap" begin
    using Intonato
    using LinearAlgebra

    # The free phase e^{iφ n̂} (n̂ = diag(0,1,…)) only acts non-trivially across
    # number states with DIFFERENT n. Use a superposition so a *relative* phase
    # on |1⟩ degrades the raw overlap but is exactly removable by the free phase.
    ψ_goal = ComplexF64[1.0, 1.0] / sqrt(2)             # (|0⟩+|1⟩)/√2
    ψ_T    = ComplexF64[1.0, cis(0.7)] / sqrt(2)        # (|0⟩+e^{i·0.7}|1⟩)/√2

    raw = abs2(dot(ψ_goal, ψ_T))
    F = phase_max_fidelity(ψ_T, ψ_goal)

    @test raw < 1.0 - 1e-3                              # relative phase degrades raw overlap
    @test F ≈ 1.0 atol = 1e-3                           # free-phase e^{iφ n̂} recovers it (φ ≈ -0.7)

    # Identical states ⇒ F = 1 regardless of grid.
    @test phase_max_fidelity(ψ_goal, ψ_goal) ≈ 1.0 atol = 1e-6

    # A genuine population mismatch can't be fixed by a phase: F < 1.
    @test phase_max_fidelity(ComplexF64[0.0, 1.0], ψ_goal) < 0.75
end

