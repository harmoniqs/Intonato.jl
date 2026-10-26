"""
    Intonato.Twins

The digital-twin core — drift processes (Ornstein–Uhlenbeck with the exact
Gaussian transition, ramp, random telegraph, scheduled jumps; `DriftPlan`
per-parameter composition), the twin-record loader (`TwinRecord` /
`load_record` / `RecordError`), and the `DigitalTwin` truth/belief/record
contract (`instantiate`, `believed`, `advance!`, `calibrate!`).

Drift moves truth only; calibration moves belief only. Twin records are
documents: code loads records, it never owns parameters.

A submodule (not top-level) by design: it keeps generic verbs such as
`instantiate`, `advance!` and `calibrate!` out of Intonato's namespace. Reach it
with `using Intonato.Twins`.

Moved from Strumento.jl (Intonato #54) — the behaviour is unchanged.
"""
module Twins

using TestItems

include("drift.jl")
include("records.jl")
include("twin.jl")

end

@testitem "Twins — the submodule exports the twin core" begin
    using Intonato.Twins
    @test isdefined(Intonato, :Twins)
    @test Intonato.Twins isa Module
    exported = names(Intonato.Twins)
    for n in (
        :DriftProcess, :OrnsteinUhlenbeck, :Ramp, :RandomTelegraph, :JumpSchedule,
        :DriftPlan, :step!, :apply,
        :TwinRecord, :load_record, :RecordError,
        :DigitalTwin, :instantiate, :believed, :advance!, :calibrate!,
    )
        @test n in exported
    end
    # the twin core is defined BY Intonato, not re-bound from Strumento
    @test parentmodule(DigitalTwin) === Intonato.Twins
    @test parentmodule(OrnsteinUhlenbeck) === Intonato.Twins
    @test parentmodule(load_record) === Intonato.Twins
end

@testitem "Twins — Intonato's top-level export surface carries no twin names" begin
    top = names(Intonato)
    for n in (
        :Twins, :DriftProcess, :OrnsteinUhlenbeck, :Ramp, :RandomTelegraph,
        :JumpSchedule, :DriftPlan, :step!, :apply, :TwinRecord, :load_record,
        :RecordError, :DigitalTwin, :instantiate, :believed, :advance!, :calibrate!,
    )
        @test !(n in top)
    end
end
