# This file contains tests for the surrounding source directory.

# ============================================================================ #
# Tests for displaced_parity.jl
# ============================================================================ #

@testitem "displaced parity vacuum" begin
    using LinearAlgebra

    n_max = 5
    ψ0 = zeros(ComplexF64, n_max)
    ψ0[1] = 1.0
    x0 = ket_to_iso(ψ0)

    # Parity at origin for vacuum: ⟨0|Π|0⟩ = 1
    P0 = displaced_parity(x0, 0.0 + 0.0im; n_max=n_max)
    @test P0 ≈ 1.0 atol = 1e-10

    # Displaced parity is related to Wigner: P(α) = (π/2) W(α)
    # For vacuum at α=0: P(0) = 1, W(0) = 2/π → P(0) = (π/2)(2/π) = 1 ✓
end

@testitem "displaced parity ForwardDiff" begin
    using ForwardDiff, LinearAlgebra

    n_max = 3
    ψ = randn(ComplexF64, n_max)
    ψ ./= norm(ψ)
    x = ket_to_iso(ψ)

    α = 0.3 + 0.2im
    g = x -> displaced_parity(x, α; n_max=n_max)
    grad = ForwardDiff.gradient(g, x)
    @test length(grad) == length(x)
    @test all(isfinite, grad)
end

# ============================================================================ #
# Tests for partial_trace.jl
# ============================================================================ #

@testitem "partial trace product state" begin
    using LinearAlgebra

    # Product state |0⟩_A ⊗ |1⟩_B in a 2×3 system
    d_A, d_B = 2, 3
    ψ_A = [1.0 + 0im, 0.0]
    ψ_B = [0.0 + 0im, 1.0, 0.0]
    ψ = kron(ψ_A, ψ_B)
    ρ = ψ * ψ'
    ρ_iso = density_to_iso_vec(ρ)

    ρ_A_iso = partial_trace_B(ρ_iso, (d_A, d_B))
    ρ_A = iso_vec_to_density(ρ_A_iso)

    expected = ψ_A * ψ_A'
    @test ρ_A ≈ expected atol = 1e-10
end

@testitem "partial trace preserves trace" begin
    using LinearAlgebra

    d_A, d_B = 2, 2
    # Random density matrix
    A = randn(ComplexF64, d_A * d_B, d_A * d_B)
    ρ = A * A' / tr(A * A')
    ρ_iso = density_to_iso_vec(ρ)

    ρ_A_iso = partial_trace_B(ρ_iso, (d_A, d_B))
    ρ_A = iso_vec_to_density(ρ_A_iso)

    @test tr(ρ_A) ≈ 1.0 atol = 1e-10
end

# ============================================================================ #
# Tests for state_measurements.jl
# ============================================================================ #

@testitem "populations" begin
    using LinearAlgebra
    # |0⟩ in 2-level iso-vec: [1, 0, 0, 0]
    x = [1.0, 0.0, 0.0, 0.0]
    p = populations(x)
    @test p ≈ [1.0, 0.0]
    @test sum(p) ≈ 1.0

    # Equal superposition: (|0⟩ + |1⟩)/√2
    x = [1.0, 1.0, 0.0, 0.0] / sqrt(2)
    p = populations(x)
    @test p ≈ [0.5, 0.5]
    @test sum(p) ≈ 1.0
end

@testitem "populations ForwardDiff" begin
    using ForwardDiff, LinearAlgebra
    x = randn(4)
    x ./= norm(x)
    J = ForwardDiff.jacobian(populations, x)
    @test size(J) == (2, 4)
    # Finite-difference check
    ε = 1e-7
    J_fd = zeros(2, 4)
    for i in 1:4
        x_p = copy(x); x_p[i] += ε
        x_m = copy(x); x_m[i] -= ε
        J_fd[:, i] = (populations(x_p) - populations(x_m)) / (2ε)
    end
    @test J ≈ J_fd atol = 1e-5
end

@testitem "full_state" begin
    x = [1.0, 2.0, 3.0, 4.0]
    @test full_state(x) == x
end

@testitem "density_matrix_measurement ForwardDiff" begin
    using ForwardDiff, LinearAlgebra
    x = randn(4)
    x ./= norm(x)
    J = ForwardDiff.jacobian(density_matrix_measurement, x)
    @test size(J) == (8, 4)  # density iso-vec of 2×2 matrix
end

@testitem "populations_density" begin
    using LinearAlgebra
    using Intonato

    # |0⟩⟨0| in compact iso: density_to_compact_iso([1 0; 0 0])
    ρ0 = ComplexF64[1 0; 0 0]
    ρ̃ = density_to_compact_iso(ρ0)
    p = populations_density(ρ̃)
    @test p ≈ [1.0, 0.0]
    @test sum(p) ≈ 1.0

    # Maximally mixed: ρ = I/2
    ρ_mixed = ComplexF64[0.5 0; 0 0.5]
    ρ̃_mixed = density_to_compact_iso(ρ_mixed)
    p = populations_density(ρ̃_mixed)
    @test p ≈ [0.5, 0.5]
    @test sum(p) ≈ 1.0
end

@testitem "populations_density ForwardDiff" begin
    using ForwardDiff, LinearAlgebra
    using Intonato

    # n=2 qubit: compact iso has n²=4 elements
    ρ = ComplexF64[0.7 0.1+0.2im; 0.1-0.2im 0.3]
    ρ̃ = density_to_compact_iso(ρ)

    J = ForwardDiff.jacobian(populations_density, ρ̃)
    @test size(J) == (2, 4)

    # Finite-difference check
    ε = 1e-7
    J_fd = zeros(2, 4)
    for i in 1:4
        x_p = copy(ρ̃); x_p[i] += ε
        x_m = copy(ρ̃); x_m[i] -= ε
        J_fd[:, i] = (populations_density(x_p) - populations_density(x_m)) / (2ε)
    end
    @test J ≈ J_fd atol = 1e-5
end

# ============================================================================ #
# Tests for wigner.jl
# ============================================================================ #

@testitem "wigner vacuum state" begin
    using LinearAlgebra

    # Vacuum state |0⟩⟨0| in 5-level Fock space
    n_max = 5
    ρ = zeros(ComplexF64, n_max, n_max)
    ρ[1, 1] = 1.0
    ρ_iso = density_to_iso_vec(ρ)

    # Wigner at origin should be 2/π for vacuum
    W0 = wigner(ρ_iso, 0.0 + 0.0im; n_max=n_max)
    @test W0 ≈ 2 / π atol = 1e-10

    # Wigner should be positive everywhere for vacuum (Gaussian)
    for r in 0.0:0.5:2.0
        W = wigner(ρ_iso, r + 0.0im; n_max=n_max)
        @test W > -1e-10  # non-negative (up to numerics)
    end
end

@testitem "wigner ForwardDiff" begin
    using ForwardDiff, LinearAlgebra

    n_max = 3
    # Random density matrix
    A = randn(ComplexF64, n_max, n_max)
    ρ = A * A' / tr(A * A')
    ρ_iso = density_to_iso_vec(ρ)

    α = 0.5 + 0.3im
    g = ρ_iso -> wigner(ρ_iso, α; n_max=n_max)
    grad = ForwardDiff.gradient(g, ρ_iso)
    @test length(grad) == length(ρ_iso)
    @test all(isfinite, grad)
end
