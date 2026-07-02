# MLE-style cavity-state reconstruction from displaced-parity measurements.

"""
    reconstruct_rho_from_parity(parities, alphas, rfd; work_dim=4rfd)
        -> Matrix{ComplexF64}

Emulate the lab's MLE cavity-state reconstruction: least-squares fit of an
`rfd × rfd` Hermitian ρ to measured displaced-parity values, then projection
onto the physical cone (PSD eigenvalue clamp + UNIT trace — hardware
results.json returns tr(ρ) = 1.0 exactly).

The forward map `parity_k = Tr[ρ · M_k]` uses displaced-parity operators built
in a `work_dim`-dimensional Fock space (so D(α) has room to displace) and
truncated to the leading `rfd × rfd` block — i.e. the fit assumes the state's
support lies in the first `rfd` levels, like the lab's reconstruction.

!!! warning "Diagnostics-only"
    This is a DIAGNOSTICS-ONLY tool — never use the reconstructed ρ as a
    feedback observable for ILC. The PSD/trace projection is a nonlinear,
    state-dependent bias that the linear Jacobian cannot track; feed the loop
    raw parity values (or model-side ρ closures) instead.
"""
function reconstruct_rho_from_parity(
    parities::AbstractVector{<:Real},
    alphas::AbstractVector{<:Number},
    rfd::Int;
    work_dim::Int = 4rfd,
)
    length(parities) == length(alphas) || throw(
        ArgumentError(
            "parities ($(length(parities))) and alphas ($(length(alphas))) must match",
        ),
    )
    Π = _parity_operator(work_dim)
    tri = rho_triangle(rfd)
    nv = rfd^2                                 # measvec length: rfd + rfd(rfd−1)
    A = Matrix{Float64}(undef, length(alphas), nv)
    for (k, α) in enumerate(alphas)
        D = _displacement_operator(Complex(α), work_dim)
        M = (D' * Π * D)[1:rfd, 1:rfd]         # Hermitian rfd×rfd block
        i = 1
        for (m, n) in tri                       # Tr[ρM] in rho_to_measvec coords
            if m == n
                A[k, i] = real(M[m, m])
                i += 1
            else
                A[k, i] = 2 * real(M[n, m])                    # ∂/∂Re(ρ_mn)
                A[k, i+1] = -2 * imag(M[n, m])                 # ∂/∂Im(ρ_mn)
                i += 2
            end
        end
    end
    v = A \ collect(Float64, parities)         # linear LS (e.g. 81 eqs × 25 DOF)
    ρ = measvec_to_rho(v, rfd)
    λ, U = eigen(Hermitian(ρ))                  # physical projection: PSD + tr=1
    λ = max.(λ, 0.0)
    s = sum(λ)
    s > 0 || return Matrix{ComplexF64}(I, rfd, rfd) ./ rfd
    return U * Diagonal(λ ./ s) * U'
end
