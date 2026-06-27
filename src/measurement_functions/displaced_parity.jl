"""
    displaced_parity(x_iso::AbstractVector, α::Complex; n_max::Int)

Compute displaced parity measurement Re⟨ψ| D†(α) Π D(α) |ψ⟩ from iso-vec ket.

D(α) is the displacement operator, Π = (-1)^(a†a) is the parity operator.
Returns a scalar Float64.
"""
function displaced_parity(x_iso::AbstractVector, α::Complex; n_max::Int)
    D = _displacement_operator(α, n_max)
    Π = _parity_operator(n_max)
    DΠD = D' * Π * D

    ψ = iso_to_ket(x_iso)
    return real(dot(ψ, DΠD * ψ))
end

"""
    _displacement_operator(α, n_max)

Compute displacement operator D(α) = exp(α a† - α* a) in the Fock basis.
"""
function _displacement_operator(α::Complex, n_max::Int)
    a = _annihilation_operator(n_max)
    M = α * a' - conj(α) * a
    return exp(Matrix(M))
end

"""
    _parity_operator(n_max)

Parity operator Π = (-1)^(a†a) = diag(1, -1, 1, -1, …).
"""
function _parity_operator(n_max::Int)
    return Diagonal([iseven(n) ? 1.0 : -1.0 for n = 0:(n_max-1)])
end

"""
    _annihilation_operator(n_max)

Bosonic annihilation operator a in the Fock basis.
"""
function _annihilation_operator(n_max::Int)
    a = zeros(ComplexF64, n_max, n_max)
    for n = 1:(n_max-1)
        a[n, n+1] = sqrt(n)
    end
    return a
end

"""
    displaced_parity_at(α; n_max)

Create a closure g(x_iso) → [displaced_parity(x_iso, α)] for MeasurementModel.
"""
displaced_parity_at(α::Complex; n_max::Int) =
    x_iso -> [displaced_parity(x_iso, α; n_max = n_max)]

# ──── Tests ──────────────────────────────────────────────────────────────────
