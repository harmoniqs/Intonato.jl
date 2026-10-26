# The ownership guard: this package DEFINES its soc substrate. Every name it
# once re-exported from the retired Strumento.jl substrate — the soc contract and
# its verbs, the channel map, the translation record + verb, readout conversion —
# plus MockSoc and the slow-DC control class resolves to a binding owned by an
# Intonato module, and no retired substrate package rides the dependency set or
# the resolved manifest.

@testitem "ownership guard: no Strumento in the dependency set or resolved manifest" begin
    using Intonato
    using Pkg
    STRUMENTO_UUID = Base.UUID("9a96c2ab-728f-48d1-a688-c746ce4d362f")
    # not loadable from this package's environment at all …
    @test Base.identify_package(Intonato, "Strumento") === nothing
    # … not a direct dependency …
    project = Pkg.TOML.parsefile(joinpath(pkgdir(Intonato), "Project.toml"))
    @test !haskey(project["deps"], "Strumento")
    @test !haskey(project["compat"], "Strumento")
    # … and not anywhere in the resolved manifest (direct or transitive).
    deps = Pkg.dependencies()
    @test !haskey(deps, STRUMENTO_UUID)
    @test all(d -> d.name != "Strumento", values(deps))
end

@testitem "ownership guard: the soc substrate is defined by Intonato itself" begin
    using Intonato
    using Piccolo      # for the duration pin's defining-module path below

    # Every formerly re-exported name, MockSoc, and the DC control class: in the
    # `using Intonato` scope, exported, and DEFINED by Intonato (not re-bound).
    for n in (
        :AbstractSoc,
        :execute!,
        :load_envelope!,
        :play_program!,
        :acquire,
        :dac_rate,
        :adc_rate,
        :QickChannelMap,
        :QickGenChannel,
        :pulse_to_envelopes,
        :QickProgram,
        :iq_to_measurements,
        :MockSoc,
        :set_gate!,
        :get_gate,
        :gate_snapshot,
        :DCAxis,
        :DCAxisMap,
        :gate_names,
        :axis_for,
    )
        @test n in names(Intonato)
        @test isdefined(@__MODULE__, n)
        @test getfield(@__MODULE__, n) === getfield(Intonato, n)
        @test parentmodule(getfield(Intonato, n)) === Intonato
    end
    @test MockSoc <: AbstractSoc

    # One Measurement type: the chassis record IS the readout conversion's record.
    @test Measurement === Intonato.Measurement
    @test parentmodule(Measurement) === Intonato
    ms = iq_to_measurements([ComplexF64[0.25, 0.75]], b -> real.(b), [3])
    @test ms isa Vector{Intonato.Measurement}

    # The pulse-sampling seam (unexported) is Intonato's too.
    @test parentmodule(Intonato.pulse_duration) === Intonato
    @test parentmodule(Intonato.sample_controls) === Intonato

    # This package's `duration` pin (src/Intonato.jl) is KEPT: Piccolo's top-level
    # reexport is still ambiguous against TimeWarp's (NamedTrajectories ≥ 0.9.3;
    # Piccolo #323), and this package reexports Piccolo's export list, so the
    # defining-module binding is what makes `Intonato.duration` resolvable — and
    # it is the binding the pulse-sampling seam reads. Once Piccolo ships its fix
    # the pin degrades to harmless shadowing; the invariants below hold either way.
    @test isdefined(Intonato, :duration)
    @test Intonato.duration === Piccolo.Quantum.Pulses.duration
    pulse = LinearSplinePulse(zeros(1, 3), [0.0, 1.5, 3.0])
    @test Intonato.pulse_duration(pulse) == Piccolo.Quantum.Pulses.duration(pulse)
end
