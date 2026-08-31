# The reexport-surface pin: `using Intonato` is the one-stop import for the full
# convenience stack — the loop chassis AND, through the Strumento ≥ 0.2 dependency,
# the soc substrate and its verbs (the convenience flip that came with the seam's
# relocation: the substrate reexported the chassis in Strumento ≤ 0.1; the chassis
# reexports the substrate now, matching the dependency direction).

# The resolution pin: the compat entry pins the exact minor (0.3), so a fresh
# resolution must land on Strumento 0.3.x — never fall back to the 0.2 line
# (the 0.2.0 tarball carried the fresh-resolution `duration` hazard the retired
# __init__ heal used to paper over; 0.3 fixes it at the substrate). This item
# FAILS if the resolver hands the suite a 0.2.x Strumento.
@testitem "fresh resolution pulls Strumento 0.3.x (compat pins the exact minor)" begin
    using Pkg
    deps = Pkg.dependencies()
    STRUMENTO_UUID = Base.UUID("9a96c2ab-728f-48d1-a688-c746ce4d362f")
    @test haskey(deps, STRUMENTO_UUID)
    dep = deps[STRUMENTO_UUID]
    @test dep.name == "Strumento"
    @test v"0.3" <= dep.version < v"0.4"
end

@testitem "using Intonato exposes the soc layer; Measurement stays Intonato's" begin
    using Intonato
    import Strumento   # module handle only: no export surface brought in
    using Piccolo      # for the duration pin's defining-module path below

    # 1. The soc contract surface is in the using-scope (reexported from Strumento).
    for f in (
        :AbstractSoc,
        :MockSoc,
        :StrumentoSoc,
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
    )
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

    # 3. The mock soc type is in the using-scope and is an AbstractSoc. Strumento
    #    0.2 (the registered tarball) defines MockSoc in its base and exports it,
    #    so the reexport exposes it directly; the extension-split move on
    #    Strumento's unreleased main would put it behind Base.get_extension —
    #    cover both so the item survives the substrate's next release.
    @test MockSoc <: AbstractSoc
    @test (
        Base.get_extension(Strumento, :StrumentoPiccoloExt) === nothing ?
        isdefined(Strumento, :MockSoc) :
        isdefined(Base.get_extension(Strumento, :StrumentoPiccoloExt), :MockSoc)
    )

    # 4. The `duration` bindings survive the NamedTrajectories ≥ 0.9.3 collision
    #    (Piccolo's top-level `duration` reexport is ambiguous against TimeWarp's;
    #    harmoniqs/Piccolo.jl#323 tracks the upstream fix): pinned to its defining
    #    module — in this package's own namespace and, via the __init__ heal in
    #    src/Intonato.jl, in the substrate's (whose typed translation resolves
    #    `duration` through Piccolo reexports and is otherwise unbound in fresh
    #    resolutions).
    @test isdefined(Intonato, :duration)
    @test Intonato.duration === Piccolo.Quantum.Pulses.duration
    @test isdefined(Strumento, :duration)
    @test Strumento.duration === Piccolo.Quantum.Pulses.duration
end
