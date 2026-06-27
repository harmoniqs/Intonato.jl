# ============================================================================ #
#                          AbstractExperimentLogger
# ============================================================================ #

"""
    AbstractExperimentLogger

Sink for [`ExperimentRecord`](@ref)s produced during experiment runs.

This is a **new** abstract type for the experiment data-collection seam — it is
deliberately *not* a subtype of `Base.AbstractLogger` (which is for textual
log-message dispatch via `@info`/`@warn`). It mirrors the *shape* of a logging
seam: concrete loggers implement

    record!(logger, record::ExperimentRecord) -> nothing

`run_experiment(exp, pulse; logger = …)` builds an `ExperimentRecord` after
each run and calls `record!(logger, record)`. The default logger is a no-op
([`NullExperimentLogger`](@ref)), so logging is strictly opt-in and never
changes the positional `run_experiment` contract.
"""
abstract type AbstractExperimentLogger end

"""
    record!(logger::AbstractExperimentLogger, record::ExperimentRecord)

Record one experiment outcome into `logger`. Returns `nothing`.
"""
function record! end

"""
    NullExperimentLogger <: AbstractExperimentLogger

No-op logger: `record!` discards the record. This is the default `logger` for
`run_experiment`, so the data-collection seam is zero-overhead unless a real
logger is passed.

Distinct from `Base.NullLogger` (textual log sink) — this drops
`ExperimentRecord`s, not log messages.
"""
struct NullExperimentLogger <: AbstractExperimentLogger end

record!(::NullExperimentLogger, ::ExperimentRecord) = nothing

"""
    InMemoryExperimentLogger <: AbstractExperimentLogger

Logger that accumulates every recorded [`ExperimentRecord`](@ref) in its
`records` vector. Useful for tests and short-lived in-process data collection;
persistent vault/catalog-backed loggers land in a follow-on plan.

# Fields
- `records::Vector{ExperimentRecord}`: recorded experiments, in call order.
"""
struct InMemoryExperimentLogger <: AbstractExperimentLogger
    records::Vector{ExperimentRecord}
end

InMemoryExperimentLogger() = InMemoryExperimentLogger(ExperimentRecord[])

record!(lg::InMemoryExperimentLogger, r::ExperimentRecord) = push!(lg.records, r)

@testitem "experiment logger sink + additive run_experiment" begin
    using Intonato
    using Intonato: record!, NullExperimentLogger, InMemoryExperimentLogger
    using LinearAlgebra

    # Single-qubit Rabi SimulatedExperiment fixture.
    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    sys = QuantumSystem(1.0 * σz, [σx], [1.0])

    N = 11;
    T = 5.0
    times = range(0.0, T, length = N) |> collect
    ψ0 = ComplexF64[1.0, 0.0]
    ψg = ComplexF64[0.0, 1.0]
    pulse = LinearSplinePulse(0.1 * ones(1, N), times)
    model = MeasurementModel(:ψ̃, [populations], [N])
    exp = SimulatedExperiment(KetTrajectory(sys, pulse, ψ0, ψg), model)

    lg = InMemoryExperimentLogger()

    # The original positional contract is unchanged: returns Vector{Measurement}.
    y = run_experiment(exp, pulse)
    @test y isa Vector{Measurement}

    # The additive path returns the same measurements AND logs an ExperimentRecord.
    # (`Measurement` has no value `==`, so compare the deterministic .data / index
    # the codebase uses, not struct identity — same gotcha as QuantumSystem.)
    y2 = run_experiment(exp, pulse; logger = lg)
    @test length(y2) == length(y)
    @test all(m2.data == m1.data && m2.index == m1.index for (m2, m1) in zip(y2, y))
    @test measurement_error(y2, y) == 0.0
    @test length(lg.records) == 1
    @test lg.records[1] isa ExperimentRecord
    @test lg.records[1].pulse === pulse
    @test lg.records[1].measurements === y2   # the SAME vector the call returned

    # A second logged run accumulates.
    run_experiment(exp, pulse; logger = lg)
    @test length(lg.records) == 2

    # NullExperimentLogger is a no-op and does NOT shadow Base.NullLogger.
    @test NullExperimentLogger !== Base.NullLogger
    @test record!(NullExperimentLogger(), lg.records[1]) === nothing  # no error, no-op

    # The default logger (NullExperimentLogger) records nothing.
    lg_default = InMemoryExperimentLogger()
    run_experiment(exp, pulse)            # default logger = NullExperimentLogger()
    @test length(lg_default.records) == 0
end
