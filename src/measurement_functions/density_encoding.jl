# Reduced-cavity ρ measurement-vector encoding.
#
# The cavity ρ is Hermitian, so its information lives in the diagonal + upper
# triangle. The canonical flattening order — real(ρ_mm) on the diagonal, then
# real & imag of ρ_mn (m < n), row-major over (m, n) — is shared by the
# model-side closures and the hardware parse so residuals line up
# element-for-element.
#
# Composite basis convention: transmon-outer / cavity-inner, i.e. the composite
# ket is kron(ψ_transmon, ψ_cavity) and reshape(ψ, Nc, Nq) puts the cavity
# along rows.

"""
    rho_triangle(rfd::Int) -> Vector{Tuple{Int,Int}}

Canonical (m, n) visit order — diagonal + upper triangle, row-major — for an
`rfd × rfd` Hermitian ρ.
"""
rho_triangle(rfd::Int) = [(m, n) for m = 1:rfd for n = m:rfd]

"""
    rho_to_measvec(ρ::AbstractMatrix, rfd::Int) -> Vector

Flatten the leading `rfd × rfd` block of a Hermitian ρ in the canonical
[`rho_triangle`](@ref) order: `real(ρ_mm)` on the diagonal, `real` & `imag` of
`ρ_mn` (m < n) off it. Length `rfd^2`.

Type-generic (the element type follows `ρ`), so ForwardDiff duals flow through.
"""
function rho_to_measvec(ρ::AbstractMatrix, rfd::Int)
    return reduce(
        vcat,
        [
            m == n ? [real(ρ[m, n])] : [real(ρ[m, n]), imag(ρ[m, n])] for
            (m, n) in rho_triangle(rfd)
        ],
    )
end

"""
    measvec_to_rho(v::AbstractVector, rfd::Int) -> Matrix{ComplexF64}

Inverse of [`rho_to_measvec`](@ref): rebuild the Hermitian `rfd × rfd` ρ from
its canonical measurement vector.
"""
function measvec_to_rho(v::AbstractVector, rfd::Int)
    ρ = zeros(ComplexF64, rfd, rfd)
    i = 1
    for (m, n) in rho_triangle(rfd)
        if m == n
            ρ[m, m] = v[i]
            i += 1
        else
            ρ[m, n] = complex(v[i], v[i+1])
            ρ[n, m] = conj(ρ[m, n])
            i += 2
        end
    end
    return ρ
end

"""
    reduced_cavity_rho(x_iso::AbstractVector; Nq::Int, Nc::Int, rfd::Int)

Reduced cavity density matrix Tr_transmon(|ψ⟩⟨ψ|) — leading `rfd × rfd` block —
from a composite transmon ⊗ cavity iso-vec ket (length `2 Nq Nc`). The
composite index is transmon-outer / cavity-inner, so `reshape(ψ, Nc, Nq)` puts
the cavity along rows and ρ_cav = M·M′.
"""
function reduced_cavity_rho(x_iso::AbstractVector; Nq::Int, Nc::Int, rfd::Int)
    M = reshape(iso_to_ket(x_iso), Nc, Nq)
    return (M * M')[1:rfd, 1:rfd]
end

"""
    rho_measurement_functions(; Nq, Nc, rfd) -> Vector{Function}

Model-side observables for reconstruction-mode campaigns: one closure per
canonical [`rho_triangle`](@ref) element, each mapping a composite iso-vec ket
to `[real(...)]` or `[imag(...)]` of the reduced cavity ρ (in the exact
[`rho_to_measvec`](@ref) order, `rfd^2` closures total), for MeasurementModel
use against a lab-reported ρ flattened the same way.

Note the feedback contract: MLE ρ-reconstruction (see
[`reconstruct_rho_from_parity`](@ref)) is diagnostics-only per spec — these
closures are the model-side counterpart for campaigns where the lab reports a
reconstructed ρ, not an endorsement of reconstruction as a feedback observable.
"""
function rho_measurement_functions(; Nq::Int, Nc::Int, rfd::Int)
    fns = Function[]
    for (m, n) in rho_triangle(rfd)
        if m == n
            push!(fns, let m = m
                x -> [real(reduced_cavity_rho(x; Nq = Nq, Nc = Nc, rfd = rfd)[m, m])]
            end)
        else
            push!(fns, let m = m, n = n
                x -> [real(reduced_cavity_rho(x; Nq = Nq, Nc = Nc, rfd = rfd)[m, n])]
            end)
            push!(fns, let m = m, n = n
                x -> [imag(reduced_cavity_rho(x; Nq = Nq, Nc = Nc, rfd = rfd)[m, n])]
            end)
        end
    end
    return fns
end
