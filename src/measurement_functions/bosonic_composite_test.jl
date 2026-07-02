# This file contains tests for the composite-space (transmon ⊗ cavity) bosonic
# observables: composite displaced parity, qubit σ_z, the reduced-cavity ρ
# measurement-vector encoding, and the MLE-style parity reconstruction.
#
# Composite basis convention: transmon-outer / cavity-inner, i.e. operators are
# kron(op_qubit, op_cavity) and composite index i = (q − 1)·Nc + c.

# ============================================================================ #
# Composite displaced parity (displaced_parity.jl)
# ============================================================================ #

@testitem "bosonic composite parity Fock states" begin
    using LinearAlgebra

    Nq, Nc = 2, 8
    g = ComplexF64[1.0, 0.0]
    for n = 0:3
        ψc = zeros(ComplexF64, Nc)
        ψc[n+1] = 1.0
        x = ket_to_iso(kron(g, ψc))
        P = displaced_parity(x, 0.0 + 0.0im; Nq = Nq, Nc = Nc)
        @test P ≈ (-1.0)^n atol = 1e-10
    end
end

@testitem "bosonic composite parity coherent state" begin
    using LinearAlgebra

    # |g⟩ ⊗ |β⟩ with Fock coefficients c_n = e^{−|β|²/2} β^n/√n! built
    # recursively (Nc = 24 makes the truncation error negligible).
    Nq, Nc = 2, 24
    β = 0.7 + 0.2im
    c = zeros(ComplexF64, Nc)
    c[1] = exp(-abs2(β) / 2)
    for n = 1:(Nc-1)
        c[n+1] = c[n] * β / sqrt(n)
    end
    x = ket_to_iso(kron(ComplexF64[1, 0], c))

    # Package convention op = D†(α) Π D(α) displaces the STATE by +α, so the
    # coherent-state anchor is P(α) = exp(−2|β + α|²); the Wigner-point form
    # exp(−2|β − α|²) is the same statement evaluated at −α.
    for α in (0.0 + 0.0im, 0.3 - 0.4im, -0.5 + 0.1im)
        P = displaced_parity(x, α; Nq = Nq, Nc = Nc)
        @test P ≈ exp(-2 * abs2(β + α)) rtol = 1e-6
        @test displaced_parity(x, -α; Nq = Nq, Nc = Nc) ≈
              exp(-2 * abs2(β - α)) rtol = 1e-6
    end
end

@testitem "bosonic composite parity Nq=1 matches cavity-only" begin
    using LinearAlgebra
    using Random
    Random.seed!(7)

    Nc = 10
    ψ = randn(ComplexF64, Nc)
    ψ ./= norm(ψ)
    x = ket_to_iso(ψ)
    for α in (0.2 + 0.5im, -0.4 - 0.1im)
        @test displaced_parity(x, α; Nq = 1, Nc = Nc) ≈
              displaced_parity(x, α; n_max = Nc) atol = 1e-12
    end
end

@testitem "bosonic composite displaced_parity kwarg routing" begin
    x = ket_to_iso(ComplexF64[1, 0, 0, 0])
    α = 0.1 + 0.0im
    # exactly one of {n_max} or {Nq, Nc} must be given
    @test_throws ArgumentError displaced_parity(x, α)
    @test_throws ArgumentError displaced_parity(x, α; n_max = 2, Nq = 2, Nc = 2)
    @test_throws ArgumentError displaced_parity(x, α; Nq = 2)  # missing Nc
    @test_throws ArgumentError displaced_parity_at(α)(x)
end

# ============================================================================ #
# qubit σ_z (displaced_parity.jl)
# ============================================================================ #

@testitem "bosonic composite qubit sigma_z" begin
    Nc = 4
    vac = zeros(ComplexF64, Nc)
    vac[1] = 1.0

    xg = ket_to_iso(kron(ComplexF64[1, 0], vac))
    xe = ket_to_iso(kron(ComplexF64[0, 1], vac))
    @test qubit_sigma_z(xg; Nq = 2, Nc = Nc) ≈ 1.0 atol = 1e-12
    @test qubit_sigma_z(xe; Nq = 2, Nc = Nc) ≈ -1.0 atol = 1e-12

    # leakage-defense convention: ALL excited transmon levels count −1
    xf = ket_to_iso(kron(ComplexF64[0, 0, 1], vac))
    @test qubit_sigma_z(xf; Nq = 3, Nc = Nc) ≈ -1.0 atol = 1e-12

    # closure factory returns [value]
    @test qubit_sigma_z_at(; Nq = 2, Nc = Nc)(xg) ≈ [1.0] atol = 1e-12
end

# ============================================================================ #
# ρ measurement-vector encoding (density_encoding.jl)
# ============================================================================ #

@testitem "bosonic composite measvec roundtrip" begin
    using LinearAlgebra
    using Random
    Random.seed!(11)

    rfd = 5
    A = randn(ComplexF64, rfd, rfd)
    H = Matrix((A + A') / 2)
    v = rho_to_measvec(H, rfd)
    @test length(v) == rfd^2
    @test measvec_to_rho(v, rfd) ≈ H atol = 1e-12
end

@testitem "bosonic composite reduced cavity rho" begin
    using LinearAlgebra

    Nq, Nc, rfd = 2, 8, 5
    ψc = zeros(ComplexF64, Nc)
    ψc[2] = 1.0   # cavity |1⟩
    x = ket_to_iso(kron(ComplexF64[1, 0], ψc))

    ρ = reduced_cavity_rho(x; Nq = Nq, Nc = Nc, rfd = rfd)
    expected = zeros(ComplexF64, rfd, rfd)
    expected[2, 2] = 1.0
    @test ρ ≈ expected atol = 1e-12

    # the per-element closures agree with the canonical flattening
    fns = rho_measurement_functions(; Nq = Nq, Nc = Nc, rfd = rfd)
    @test length(fns) == rfd^2
    y = reduce(vcat, [f(x) for f in fns])
    @test y ≈ rho_to_measvec(expected, rfd) atol = 1e-12
end

# ============================================================================ #
# Parity reconstruction (parity_reconstruction.jl) — diagnostics-only
# ============================================================================ #

@testitem "bosonic composite parity reconstruction Fock-1" begin
    using LinearAlgebra

    # Noiseless displaced parities of cavity |1⟩ on a 9×9 grid.
    n_max = 24
    ψ1 = zeros(ComplexF64, n_max)
    ψ1[2] = 1.0
    x_iso = ket_to_iso(ψ1)
    alphas = vec([re + im * imm for re in range(-1.2, 1.2, length = 9),
                  imm in range(-1.2, 1.2, length = 9)])
    parities = [displaced_parity(x_iso, α; n_max = n_max) for α in alphas]

    rfd = 5
    ρ = reconstruct_rho_from_parity(parities, alphas, rfd)
    @test size(ρ) == (rfd, rfd)
    @test real(ρ[2, 2]) > 0.999                      # ⟨1|ρ|1⟩ fidelity
    @test minimum(eigvals(Hermitian(ρ))) ≥ -1e-10    # PSD
    @test tr(ρ) ≈ 1.0 atol = 1e-10                   # unit trace
end

# ============================================================================ #
# ForwardDiff compatibility
# ============================================================================ #

@testitem "bosonic composite ForwardDiff" begin
    using ForwardDiff, LinearAlgebra
    using Random
    Random.seed!(3)

    Nq, Nc = 2, 6
    ψ = randn(ComplexF64, Nq * Nc)
    ψ ./= norm(ψ)
    x = ket_to_iso(ψ)

    f_parity = displaced_parity_at(0.3 + 0.1im; Nq = Nq, Nc = Nc)
    Jp = ForwardDiff.jacobian(f_parity, x)
    @test size(Jp) == (1, 2 * Nq * Nc)
    @test all(isfinite, Jp)

    fns = rho_measurement_functions(; Nq = Nq, Nc = Nc, rfd = 3)
    f_rho = fns[2]   # an off-diagonal real-part closure
    Jr = ForwardDiff.jacobian(f_rho, x)
    @test size(Jr) == (1, 2 * Nq * Nc)
    @test all(isfinite, Jr)

    f_sz = qubit_sigma_z_at(; Nq = Nq, Nc = Nc)
    Js = ForwardDiff.jacobian(f_sz, x)
    @test all(isfinite, Js)
end
