"""
    Intonato.Twins

The digital-twin core — drift processes (Ornstein–Uhlenbeck with the exact
Gaussian transition, ramp, random telegraph, scheduled jumps; `DriftPlan`
per-parameter composition), the twin-record loader (`TwinRecord` /
`load_record` / `RecordError`), and the `DigitalTwin` truth/belief/record
contract (`instantiate`, `believed`, `advance!`, `calibrate!`) — plus the twin's
soc face (`TwinSoc`: family-built systems, readout confusion, binomial shots,
drift-after-acquire, the DC control path and charge-sensor readout), the
device-channel → twin-drive wiring (`TwinWiringMap`), and the physics families
(the bosonic Lindblad family, the spin charge landscape).

Drift moves truth only; calibration moves belief only. Twin records are
documents: code loads records, it never owns parameters.

A submodule (not top-level) by design: it keeps generic verbs such as
`instantiate`, `advance!` and `calibrate!` out of Intonato's namespace. Reach it
with `using Intonato.Twins`.

Moved from Strumento.jl (Intonato #54, #55) — the behaviour is unchanged.
"""
module Twins

using TestItems
using LinearAlgebra: I, kron
using Piccolo:
    AbstractPulse,
    QuantumSystem,
    OpenQuantumSystem,
    KetTrajectory,
    DensityTrajectory,
    ket_to_iso,
    density_to_iso_vec,
    annihilate
using Piccolo.Quantum.Pulses: get_knot_times

# The soc contract the twin face implements (defined by Intonato's hardware
# substrate, src/hardware/): extended here, never redeclared.
import ..Intonato:
    AbstractSoc,
    execute!,
    load_envelope!,
    play_program!,
    acquire,
    dac_rate,
    adc_rate,
    set_gate!,
    get_gate,
    gate_snapshot,
    gate_names
using ..Intonato:
    QickChannelMap, QickProgram, pulse_to_envelopes, populations, _played_pulse

include("drift.jl")
include("records.jl")
include("twin.jl")
include("wiring.jl")
include("twin_soc.jl")
include("bosonic_family.jl")
include("spin_family.jl")

end

@testitem "Twins — the submodule exports the twin core" begin
    using Intonato.Twins
    @test isdefined(Intonato, :Twins)
    @test Intonato.Twins isa Module
    exported = names(Intonato.Twins)
    for n in (
        :DriftProcess,
        :OrnsteinUhlenbeck,
        :Ramp,
        :RandomTelegraph,
        :JumpSchedule,
        :DriftPlan,
        :step!,
        :apply,
        :TwinRecord,
        :load_record,
        :RecordError,
        :DigitalTwin,
        :instantiate,
        :believed,
        :advance!,
        :calibrate!,
        :TwinSoc,
        :read_charge_sensor,
        :charge_sensor_sweep,
        :TwinWiringMap,
        :TwinGenWiring,
        :wiring_for,
        :bosonic_system_builder,
        :bosonic_ancilla_populations,
        :spin_landscape_builder,
        :DoubleDotLandscape,
    )
        @test n in exported
    end
    # the twin core is defined BY Intonato.Twins itself (no re-bound substrate)
    @test parentmodule(DigitalTwin) === Intonato.Twins
    @test parentmodule(OrnsteinUhlenbeck) === Intonato.Twins
    @test parentmodule(load_record) === Intonato.Twins
    for T in (
        TwinSoc,
        TwinWiringMap,
        TwinGenWiring,
        DoubleDotLandscape,
        Intonato.Twins.AbstractChargeLandscape,
    )
        @test parentmodule(T) === Intonato.Twins
    end
    for f in (
        read_charge_sensor,
        charge_sensor_sweep,
        wiring_for,
        bosonic_system_builder,
        bosonic_ancilla_populations,
        spin_landscape_builder,
        Intonato.Twins.sensor_probabilities,
    )
        @test parentmodule(f) === Intonato.Twins
    end
    # the twin face implements the soc contract — the SAME function objects
    # Intonato's hardware substrate declares (extended, never shadowed)
    @test TwinSoc <: Intonato.AbstractSoc
    for f in (
        :execute!,
        :acquire,
        :load_envelope!,
        :play_program!,
        :dac_rate,
        :adc_rate,
        :set_gate!,
        :get_gate,
        :gate_snapshot,
        :gate_names,
    )
        @test getfield(Intonato.Twins, f) === getfield(Intonato, f)
    end
end

@testitem "Twins — Intonato's top-level export surface carries no twin names" begin
    top = names(Intonato)
    for n in (
        :Twins,
        :DriftProcess,
        :OrnsteinUhlenbeck,
        :Ramp,
        :RandomTelegraph,
        :JumpSchedule,
        :DriftPlan,
        :step!,
        :apply,
        :TwinRecord,
        :load_record,
        :RecordError,
        :DigitalTwin,
        :instantiate,
        :believed,
        :advance!,
        :calibrate!,
        :TwinSoc,
        :read_charge_sensor,
        :charge_sensor_sweep,
        :TwinWiringMap,
        :TwinGenWiring,
        :wiring_for,
        :bosonic_system_builder,
        :bosonic_ancilla_populations,
        :spin_landscape_builder,
        :DoubleDotLandscape,
        :AbstractChargeLandscape,
        :sensor_probabilities,
    )
        @test !(n in top)
    end
    # the soc face, wiring and families are not even bound at top level
    for n in (
        :TwinSoc,
        :read_charge_sensor,
        :charge_sensor_sweep,
        :TwinWiringMap,
        :TwinGenWiring,
        :wiring_for,
        :bosonic_system_builder,
        :bosonic_ancilla_populations,
        :spin_landscape_builder,
        :DoubleDotLandscape,
        :AbstractChargeLandscape,
        :sensor_probabilities,
    )
        @test !isdefined(Intonato, n)
    end
end
