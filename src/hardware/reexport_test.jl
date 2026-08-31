# The reexport-surface pin: `using Intonato` is the one-stop import for the full
# convenience stack — the loop chassis AND, through the Strumento ≥ 0.2 dependency,
# the soc substrate and its verbs (the convenience flip that came with the seam's
# relocation: the substrate used to reexport the chassis, the chassis now reexports
# the substrate, matching the dependency direction).

@testitem "using Intonato exposes the soc layer; Measurement stays Intonato's" begin
    using Intonato
    import Strumento   # module handle only: no export surface brought in
    using Piccolo      # for the duration pin's defining-module path below

    # 1. The soc contract surface is in the using-scope (reexported from Strumento).
    for f in (:AbstractSoc, :execute!, :load_envelope!, :play_program!, :acquire,
        :dac_rate, :adc_rate, :QickChannelMap, :QickGenChannel,
        :pulse_to_envelopes, :QickProgram, :iq_to_measurements)
        @test isdefined(@__MODULE__, f)
        @test isdefined(Intonato, f)
    end

    # 2. NAME COLLISION RESOLUTION: both packages export `Measurement`. The
    #    reexport deliberately EXCLUDES Strumento's (documented at the reexport
    #    site in src/Intonato.jl): this package's `Measurement` is the chassis
    #    type the hardware seam feeds, so it governs. The bare name must be
    #    unambiguous in the using-scope and BE Intonato's.
    @test isdefined(@__MODULE__, :Measurement)
    @test Measurement === Intonato.Measurement
    @test Measurement !== Strumento.Measurement

    # 3. The extension-defined mock type is reachable through its canonical
    #    handle (extension exports do not surface on the parent namespace).
    ext = Base.get_extension(Strumento, :StrumentoPiccoloExt)
    @test ext !== nothing
    @test isdefined(ext, :MockSoc)
    @test ext.MockSoc <: AbstractSoc

    # 4. The `duration` reexport binding survives the NamedTrajectories ≥ 0.9.3
    #    collision (Piccolo's top-level `duration` reexport is ambiguous against
    #    TimeWarp's; harmoniqs/Piccolo.jl#323 tracks the upstream fix): pinned
    #    here to its defining module, the same binding StrumentoPiccoloExt uses.
    @test isdefined(Intonato, :duration)
    @test Intonato.duration === Piccolo.Quantum.Pulses.duration
end
