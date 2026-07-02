"""
    displaced_parity(x_iso::AbstractVector, α::Complex; n_max::Int)
    displaced_parity(x_iso::AbstractVector, α::Complex; Nq::Int, Nc::Int)

Compute the displaced parity measurement from an iso-vec ket.

Exactly one of `n_max` or the pair `(Nq, Nc)` must be given (Julia does not
dispatch on keyword arguments, so a single entry point routes between the two
bodies):

- `n_max` — cavity-only: Re⟨ψ| D†(α) Π D(α) |ψ⟩ on a length-`2 n_max` iso-ket.
- `Nq, Nc` — composite transmon ⊗ cavity space (transmon-outer / cavity-inner,
  composite index i = (q − 1)·Nc + c): Re⟨ψ| kron(I_Nq, D†(α) Π D(α)) |ψ⟩ on a
  length-`2 Nq Nc` iso-ket. The `kron(I_q, ·)` structure makes this observable
  blind to transmon leakage by construction (pair with [`qubit_sigma_z`](@ref)).

D(α) is the displacement operator, Π = (-1)^(a†a) is the parity operator.
Returns a scalar Float64.
"""
function displaced_parity(
    x_iso::AbstractVector,
    α::Complex;
    n_max::Union{Int,Nothing} = nothing,
    Nq::Union{Int,Nothing} = nothing,
    Nc::Union{Int,Nothing} = nothing,
)
    cavity_only = n_max !== nothing
    composite = Nq !== nothing || Nc !== nothing
    if cavity_only && composite
        throw(ArgumentError(
            "displaced_parity: give exactly one of `n_max` (cavity-only) or `(Nq, Nc)` (composite), not both",
        ))
    elseif !cavity_only && !composite
        throw(ArgumentError(
            "displaced_parity: give either `n_max` (cavity-only) or `(Nq, Nc)` (composite)",
        ))
    elseif composite && (Nq === nothing || Nc === nothing)
        throw(ArgumentError(
            "displaced_parity: composite space needs BOTH `Nq` and `Nc`",
        ))
    end

    if cavity_only
        D = _displacement_operator(α, n_max)
        Π = _parity_operator(n_max)
        DΠD = D' * Π * D

        ψ = iso_to_ket(x_iso)
        return real(dot(ψ, DΠD * ψ))
    else
        D = _displacement_operator(α, Nc)
        Π = _parity_operator(Nc)
        op = kron(Matrix{ComplexF64}(I, Nq, Nq), D' * Π * D)

        ψ = iso_to_ket(x_iso)
        return real(dot(ψ, op * ψ))
    end
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
    displaced_parity_at(α; Nq, Nc)

Create a closure g(x_iso) → [displaced_parity(x_iso, α)] for MeasurementModel.

Routes exactly like [`displaced_parity`](@ref): give `n_max` for the
cavity-only observable or `(Nq, Nc)` for the composite transmon ⊗ cavity one.
"""
displaced_parity_at(
    α::Complex;
    n_max::Union{Int,Nothing} = nothing,
    Nq::Union{Int,Nothing} = nothing,
    Nc::Union{Int,Nothing} = nothing,
) = x_iso -> [displaced_parity(x_iso, α; n_max = n_max, Nq = Nq, Nc = Nc)]

"""
    qubit_sigma_z(x_iso::AbstractVector; Nq::Int, Nc::Int)

Compute ⟨ kron(σz_eff, I_Nc) ⟩ on a composite transmon ⊗ cavity iso-vec ket
(transmon-outer / cavity-inner, length `2 Nq Nc`), where
σz_eff = diag(+1, −1, −1, …) over the `Nq` transmon levels: +1 for |g⟩ and −1
for ALL excited levels (the leakage-defense convention — this is the one
observable that opposes the transmon leakage the `kron(I_q, ·)` cavity
observables are blind to).

Returns a scalar Float64.
"""
function qubit_sigma_z(x_iso::AbstractVector; Nq::Int, Nc::Int)
    σz = Diagonal([q == 0 ? 1.0 : -1.0 for q = 0:(Nq-1)])
    op = kron(Matrix{ComplexF64}(σz), Matrix{ComplexF64}(I, Nc, Nc))

    ψ = iso_to_ket(x_iso)
    return real(dot(ψ, op * ψ))
end

"""
    qubit_sigma_z_at(; Nq, Nc)

Create a closure g(x_iso) → [qubit_sigma_z(x_iso)] for MeasurementModel.
"""
qubit_sigma_z_at(; Nq::Int, Nc::Int) =
    x_iso -> [qubit_sigma_z(x_iso; Nq = Nq, Nc = Nc)]

# ──── Tests ──────────────────────────────────────────────────────────────────
