"""
    wigner(ρ_iso::AbstractVector, α::Complex; n_max::Int)

Compute Wigner function W(α) for density matrix in iso-vec form.
Uses Laguerre-Clenshaw recurrence for numerical stability.

Returns a scalar Float64.
"""
function wigner(ρ_iso::AbstractVector, α::Complex; n_max::Int)
    ρ = iso_vec_to_density(ρ_iso)
    return _wigner_from_density(ρ, α, n_max)
end

"""
    _wigner_from_density(ρ, α, n_max)

Compute W(α) = (2/π) Tr[D†(α) Π D(α) ρ] via displaced parity.

Uses the relation W(α) = (2/π) Σ_{m,n} ρ_{mn} W_{mn}(α) where
W_{mn}(α) = (-1)^n (2/π) exp(-2|α|²) √(n!/m!) (2α)^{m-n} L_n^{m-n}(4|α|²)
for m ≥ n (and conjugate for m < n).
"""
function _wigner_from_density(ρ::AbstractMatrix, α::Complex, n_max::Int)
    x = 4 * abs2(α)
    W = 0.0

    for m in 0:n_max-1
        for n in 0:n_max-1
            if m >= n
                # L_n^{m-n}(x) via recurrence
                k = m - n  # order of associated Laguerre
                L = _laguerre(n, k, x)

                # Prefactor: (-1)^n * sqrt(n!/m!) * (2α)^{m-n}
                phase = iseven(n) ? 1.0 : -1.0
                sqrt_ratio = exp(0.5 * (loggamma(n + 1) - loggamma(m + 1)))
                power_term = (2α)^k

                W_mn = phase * sqrt_ratio * power_term * L
                W += real(ρ[m+1, n+1] * W_mn)

                # Add conjugate for off-diagonal
                if m != n
                    W += real(ρ[n+1, m+1] * conj(W_mn))
                end
            end
        end
    end

    return (2 / π) * exp(-2 * abs2(α)) * W
end

"""
    _laguerre(n, k, x)

Evaluate associated Laguerre polynomial L_n^k(x) via forward recurrence.
No in-place mutation for ForwardDiff compatibility.
"""
function _laguerre(n::Int, k::Int, x)
    if n == 0
        return one(x)
    elseif n == 1
        return 1 + k - x
    end

    L_prev2 = one(x)
    L_prev1 = 1 + k - x

    for i in 2:n
        L_curr = ((2i - 1 + k - x) * L_prev1 - (i - 1 + k) * L_prev2) / i
        L_prev2 = L_prev1
        L_prev1 = L_curr
    end

    return L_prev1
end

using SpecialFunctions: loggamma

"""
    wigner_at(α; n_max)

Create a closure g(ρ_iso) → [W(α)] for use in MeasurementModel.
"""
wigner_at(α::Complex; n_max::Int) = ρ_iso -> [wigner(ρ_iso, α; n_max=n_max)]

# ──── Tests ──────────────────────────────────────────────────────────────────
