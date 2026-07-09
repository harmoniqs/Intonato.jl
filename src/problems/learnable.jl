# ──── Learnable-parameter declaration + strategy seam ─────────────────────────
#
# The public, strategy-generic surface for declaring that a closed-loop tuning
# strategy learns a device parameter. A parameter is declared ONCE, as a `Learn`
# value in a strategy's learn set; the `PulseTuningProblem` constructor reads the
# set back through the `learnables` generic and validates the QCP against it
# before any device time is spent. Names are deliberately method-agnostic
# (`Learn`, `learnables`) — no calibration/ILC vocabulary leaks into the public
# chassis (public/private split, 2026-06-12).

"""
    Learn(init; bounds, weight=1.0)
    Learn(; bounds, weight=1.0)

Declaration of a single learnable global parameter for a closed-loop tuning
strategy. A learnable parameter is declared **once** — as a `Learn` value in a
strategy's `learn` set — and everything else (the regularizer weight, the box
bounds, and the check that the QCP's integrator actually carries the parameter)
is derived and validated from it at `PulseTuningProblem` construction.

# Arguments / keywords
- `init::Real` (positional, optional): initial value for the parameter. Omit it
  (the keyword-only form) to take the nominal value from `sys.global_params` at
  wiring time.
- `bounds::Tuple{Real,Real}`: **required** `(lo, hi)` box on the parameter. There
  is no bounds-free form by design: an unbounded learnable global is exactly the
  phantom-drift failure this declaration exists to prevent.
- `weight::Real = 1.0`: regularizer weight for the parameter (the strategy's
  per-parameter `R_θ` entry — larger keeps the estimate closer to `init`).

# Examples
```julia
Learn(1.0; bounds = (0.5, 2.0), weight = 5.0)   # start at 1.0, box [0.5, 2.0]
Learn(; bounds = (0.5, 2.0))                     # start at the nominal value
```
"""
struct Learn
    init::Union{Nothing,Float64}
    bounds::Tuple{Float64,Float64}
    weight::Float64
end

function _check_learn_bounds(init, bounds)
    lo, hi = bounds
    lo < hi || throw(
        ArgumentError("Learn bounds must satisfy lo < hi, got (lo, hi) = ($lo, $hi)."),
    )
    if !isnothing(init) && !(lo ≤ init ≤ hi)
        throw(
            ArgumentError(
                "Learn init ($init) is outside its bounds ($lo, $hi). Pass an " *
                "init within the box or widen the bounds.",
            ),
        )
    end
    return nothing
end

function Learn(init::Real; bounds::Tuple{<:Real,<:Real}, weight::Real = 1.0)
    _check_learn_bounds(init, bounds)
    return Learn(Float64(init), (Float64(bounds[1]), Float64(bounds[2])), Float64(weight))
end

function Learn(; bounds::Tuple{<:Real,<:Real}, weight::Real = 1.0)
    _check_learn_bounds(nothing, bounds)
    return Learn(nothing, (Float64(bounds[1]), Float64(bounds[2])), Float64(weight))
end

function Base.show(io::IO, l::Learn)
    initstr = isnothing(l.init) ? "init=nominal" : string(l.init)
    print(io, "Learn(", initstr, "; bounds=", l.bounds, ", weight=", l.weight, ")")
end

"""
    learnables(strategy::AbstractTuningStrategy) -> NamedTuple

The set of learnable global parameters a strategy declares, as a `NamedTuple` of
[`Learn`](@ref) values keyed by global-parameter name. The `PulseTuningProblem`
constructor reads this to validate the QCP (parameter present in the system;
carried by a globals-aware integrator; bounds consistent) before the outer loop
runs. Default: `(;)` — a strategy that learns nothing (e.g. `IdentityStrategy`,
`LowRankHessianStrategy`).

A concrete calibrating strategy overrides this to return its `learn` set; that
one method is the entire coupling needed to receive full chassis validation.
"""
learnables(::AbstractTuningStrategy) = (;)

# ──── Validation helpers (used by the PulseTuningProblem constructor) ──────────
#
# All three inspect their inputs structurally (duck-typed on `global_params` /
# `global_names` / the bounds-constraint fields) so the public chassis needs no
# new dependency on DirectTrajOpt's concrete integrator/constraint types.

"""
    _check_learn_keys(learn_keys, available)

Every declared learnable name must be an actual global parameter of the system.
Throws `ArgumentError` listing the available globals otherwise.
"""
function _check_learn_keys(learn_keys, available)
    unknown = [k for k in learn_keys if !(k in available)]
    isempty(unknown) && return nothing
    throw(
        ArgumentError(
            "learn=(…) declares parameter(s) $(unknown) that are not global " *
            "parameters of the QCP's system. Available globals: " *
            "$(collect(available)). Declare only names present in " *
            "`sys.global_params`, or re-parameterize the system to expose them.",
        ),
    )
end

"""
    _require_globals_aware_integrators(integrators, names)

Every learnable name must be carried by at least one globals-aware integrator
(one exposing a `global_names` field that includes it). A learnable global that
no integrator reads cannot affect the dynamics — the silent no-op this guards
against. Throws `ErrorException` naming the fix otherwise. Integrators with no
`global_names` concept (e.g. `DerivativeIntegrator`) are simply not counted.
"""
function _require_globals_aware_integrators(integrators, names)
    covered = Set{Symbol}()
    for integ in integrators
        hasproperty(integ, :global_names) || continue
        gnames = integ.global_names
        isnothing(gnames) && continue
        for n in gnames
            push!(covered, Symbol(n))
        end
    end
    uncovered = [n for n in names if !(n in covered)]
    isempty(uncovered) && return nothing
    error(
        "Learnable parameter(s) $(uncovered) are not carried by any globals-aware " *
        "integrator in the QCP, so they cannot affect the dynamics. Build the " *
        "dynamics integrator with `global_names = $(collect(names))` (or a " *
        "superset) — e.g. `HermitianExponentialIntegrator(qtraj, N; " *
        "global_names = $(collect(names)))` — then rebuild the QCP.",
    )
end

# Normalize a BoundsConstraint's `bounds_values` (scalar | vector | (lo,hi)
# vectors) to a scalar `(lo, hi)` pair for a 1-dim global; returns `nothing`
# for shapes we don't compare (so we never raise a false mismatch alarm).
function _global_bounds_pair(bounds_values)
    if bounds_values isa Tuple && length(bounds_values) == 2
        lo, hi = bounds_values
        (
            lo isa AbstractVector &&
            hi isa AbstractVector &&
            length(lo) == 1 &&
            length(hi) == 1
        ) || return nothing
        return (Float64(lo[1]), Float64(hi[1]))
    elseif bounds_values isa Real
        return (-Float64(bounds_values), Float64(bounds_values))
    elseif bounds_values isa AbstractVector && length(bounds_values) == 1
        return (-Float64(bounds_values[1]), Float64(bounds_values[1]))
    end
    return nothing
end

"""
    _check_learn_bounds_consistency(constraints, learn; verbose)

Warn (never mutate) when a learnable parameter's declared `Learn.bounds` are
absent from or disagree with the QCP's installed global bounds. `Learn.bounds`
is the intended single source of truth; the check surfaces the mismatch at
construction rather than letting a stale/absent bound (e.g. Piccolo's permissive
default — the free-phase 2π-wall trap) silently cap or misdirect the learner.
Non-mutating by design: the authoritative-bounds behavior belongs to
`CalibrationProblem`, which builds the QCP.
"""
function _check_learn_bounds_consistency(constraints, learn; verbose::Bool)
    verbose || return nothing
    for (name, l) in pairs(learn)
        installed = nothing
        for c in constraints
            if hasproperty(c, :is_global) &&
               c.is_global &&
               hasproperty(c, :var_names) &&
               c.var_names == name &&
               hasproperty(c, :bounds_values)
                installed = _global_bounds_pair(c.bounds_values)
                break
            end
        end
        if isnothing(installed)
            @warn(
                "learn: no global bounds constraint found on the QCP for :$name; " *
                "it may inherit permissive default bounds (drift / 2π-wall risk). " *
                "Pass `global_bounds = Dict(:$name => $(l.bounds))` when building " *
                "the QCP, or use `CalibrationProblem` to build it for you."
            )
        elseif !(
            isapprox(installed[1], l.bounds[1]; atol = 1e-12) &&
            isapprox(installed[2], l.bounds[2]; atol = 1e-12)
        )
            @warn(
                "learn: Learn bounds $(l.bounds) for :$name disagree with the QCP's " *
                "installed global bounds $(installed). Learn is the intended source " *
                "of truth — reconcile the QCP's `global_bounds`, or use " *
                "`CalibrationProblem`."
            )
        end
    end
    return nothing
end

"""
    _wire_learnables!(qcp, strategy; verbose)

Validate a QCP against a strategy's declared learnable parameters
([`learnables`](@ref)) at construction time — the first point the QCP and
strategy meet. No-op (and allocation-free on the fast path) when the strategy
learns nothing. Checks, in order: every learnable name is a system global
([`_check_learn_keys`](@ref)); every name is carried by a globals-aware
integrator ([`_require_globals_aware_integrators`](@ref)); declared bounds are
consistent with the QCP's installed bounds ([`_check_learn_bounds_consistency`](@ref),
warn-only). Errors are raised before device time; the constructor does not
mutate the QCP.
"""
function _wire_learnables!(qcp, strategy::AbstractTuningStrategy; verbose::Bool = true)
    learn = learnables(strategy)
    isempty(learn) && return nothing
    sys = get_system(qcp.qtraj)
    _check_learn_keys(keys(learn), keys(sys.global_params))
    _require_globals_aware_integrators(qcp.prob.integrators, collect(Symbol, keys(learn)))
    _check_learn_bounds_consistency(qcp.prob.constraints, learn; verbose)
    return nothing
end

# ============================================================================ #
#                                   Tests
# ============================================================================ #

@testitem "Learn construction + validation" begin
    using Intonato

    # Positional init + required bounds.
    l = Learn(1.0; bounds = (0.5, 2.0), weight = 5.0)
    @test l.init == 1.0
    @test l.bounds == (0.5, 2.0)
    @test l.weight == 5.0

    # Keyword-only form: init taken from nominal at wiring time (nothing here).
    l2 = Learn(; bounds = (0.5, 2.0))
    @test isnothing(l2.init)
    @test l2.weight == 1.0

    # bounds is required (no positional-only, no bounds-free form): omitting the
    # required `bounds` keyword throws UndefKeywordError.
    @test_throws UndefKeywordError Learn(1.0)
    @test_throws UndefKeywordError Learn()

    # lo < hi enforced.
    @test_throws ArgumentError Learn(1.0; bounds = (2.0, 0.5))
    @test_throws ArgumentError Learn(1.0; bounds = (1.0, 1.0))

    # init must be within bounds.
    @test_throws ArgumentError Learn(3.0; bounds = (0.5, 2.0))

    # show is compact and mentions the bounds/weight.
    s = sprint(show, l)
    @test occursin("Learn(", s)
    @test occursin("bounds=(0.5, 2.0)", s)
    @test occursin("init=nominal", sprint(show, l2))
end

@testitem "learnables default is empty for non-learning strategies" begin
    using Intonato
    @test learnables(IdentityStrategy()) == (;)
end

@testitem "learnables validation helpers" begin
    using Intonato
    using Intonato:
        _check_learn_keys, _require_globals_aware_integrators, _global_bounds_pair

    # _check_learn_keys: unknown parameter → ArgumentError listing available.
    @test _check_learn_keys([:ω], [:ω, :Δ]) === nothing
    err = try
        _check_learn_keys([:γ], [:ω, :Δ])
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("γ", sprint(showerror, err))
    @test occursin("ω", sprint(showerror, err))

    # _require_globals_aware_integrators: at least one integrator must carry the
    # learnable global. Mock integrators duck-typed on `global_names`.
    struct _MockGlobalInteg
        global_names::Vector{Symbol}
    end
    struct _MockNoGlobals end
    struct _MockNothingGlobals
        global_names::Nothing
    end

    # Covered by a globals-aware integrator → ok.
    @test _require_globals_aware_integrators(
        [_MockNoGlobals(), _MockGlobalInteg([:ω])],
        [:ω],
    ) === nothing

    # No integrator carries :ω → error naming the fix.
    err2 = try
        _require_globals_aware_integrators([_MockNoGlobals()], [:ω])
        nothing
    catch e
        e
    end
    @test err2 isa ErrorException
    @test occursin("ω", err2.msg)
    @test occursin("global_names", err2.msg)

    # global_names === nothing counts as "carries nothing".
    err3 = try
        _require_globals_aware_integrators([_MockNothingGlobals(nothing)], [:ω])
        nothing
    catch e
        e
    end
    @test err3 isa ErrorException

    # _global_bounds_pair: normalize the three BoundsConstraint shapes.
    @test _global_bounds_pair(([0.5], [2.0])) == (0.5, 2.0)
    @test _global_bounds_pair(1.5) == (-1.5, 1.5)
    @test _global_bounds_pair([1.5]) == (-1.5, 1.5)
    @test _global_bounds_pair(([0.5, 0.1], [2.0, 0.9])) === nothing  # multi-dim: skip
end

@testitem "learn bounds-consistency check warns, never mutates" begin
    using Intonato
    using Intonato: _check_learn_bounds_consistency

    struct _MockBoundsCon
        is_global::Bool
        var_names::Symbol
        bounds_values::Any
    end

    learn = (ω = Learn(1.0; bounds = (0.5, 2.0)),)

    # Matching installed bounds → no warning.
    matching = [_MockBoundsCon(true, :ω, ([0.5], [2.0]))]
    @test_logs _check_learn_bounds_consistency(matching, learn; verbose = true)

    # Disagreeing installed bounds → warn.
    mismatched = [_MockBoundsCon(true, :ω, ([-6.28], [6.28]))]
    @test_logs (:warn, r"disagree") match_mode = :any _check_learn_bounds_consistency(
        mismatched,
        learn;
        verbose = true,
    )

    # No global bounds constraint at all → warn about defaults.
    @test_logs (:warn, r"no global bounds") match_mode = :any _check_learn_bounds_consistency(
        _MockBoundsCon[],
        learn;
        verbose = true,
    )

    # verbose=false silences everything.
    @test_logs _check_learn_bounds_consistency(mismatched, learn; verbose = false)
end
