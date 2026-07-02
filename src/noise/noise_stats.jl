# ============================================================================ #
#            Statistics of the whitened measurement cost under noise
# ============================================================================ #
#
# For measured y_exp = y_true + ε with ε ~ N(0, Σ) and any weight W, the
# observed cost Ĵ = ‖W r‖², r = y_goal − y_exp, satisfies
#
#     E[Ĵ]   = J_true + tr(WΣWᵀ)
#     Var[Ĵ] = 2 tr((WΣWᵀ)²) + 4 r_trueᵀ Wᵀ W Σ Wᵀ W r_true .
#
# These helpers implement the diagonal fast path: W = Diag(w), Σ = Diag(σ2),
# with the plug-in residual r̂ standing in for r_true. Deterministic elements
# carry σ² = 0 and drop out of every formula — an all-deterministic model gives
# σ_J = 0 and downstream acceptance tests reduce to the legacy raw-ratio
# behavior. See the spec section "Noise-aware acceptance and selection".

"""
    noise_floor(w, σ2) -> Float64

The additive noise bias of the whitened cost, `tr(WΣWᵀ) = Σᵢ wᵢ² σᵢ²`
(diagonal case). This is what `E[Ĵ]` exceeds `J_true` by; subtract it via
[`debiased_cost`](@ref) before comparing or selecting on measured costs.
"""
noise_floor(w::AbstractVector, σ2::AbstractVector) = sum(abs2.(w) .* σ2)

"""
    debiased_cost(Ĵ, w, σ2) -> Float64

The debiased cost `J̃ = Ĵ − tr(WΣWᵀ)` — an unbiased estimate of the true
whitened cost. Logs and best-iterate selection use `J̃`, removing the
documented best-J selection bias under shot noise at the source.
"""
debiased_cost(Ĵ::Real, w, σ2) = Ĵ - noise_floor(w, σ2)

"""
    cost_std(r, w, σ2) -> Float64

Standard deviation of the observed whitened cost `Ĵ = ‖W r‖²` under
measurement noise, using the plug-in residual `r`:

    Var[Ĵ] = 2 Σᵢ (wᵢ²σᵢ²)² + 4 Σᵢ rᵢ² wᵢ⁴ σᵢ²  (diagonal case).

Deterministic elements (σᵢ² = 0) contribute nothing; an all-deterministic
model returns exactly 0.
"""
function cost_std(r::AbstractVector, w::AbstractVector, σ2::AbstractVector)
    a = abs2.(w) .* σ2                                  # diag(WΣWᵀ)
    return sqrt(2sum(abs2, a) + 4sum(abs2.(r) .* abs2.(w) .* a))
end

"""
    diff_std(σA, σB) -> Float64

Standard deviation of the DIFFERENCE of two independent noisy costs,
`σ_Δ = √(σ_A² + σ_B²)`. Acceptance and stall tests compare two *measured*
costs (the base is itself a measured quantity), so thresholds must be scaled
by σ_Δ ≈ √2·σ_J — a bare `k·σ_J` threshold would silently be a `k/√2` test.
"""
diff_std(σA::Real, σB::Real) = hypot(σA, σB)
