# The reexport-surface pin: `using Intonato` is the one-stop import for the full
# convenience stack — the loop chassis AND, through the Strumento ≥ 0.3 dependency,
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

    # 1. The soc contract surface is in the using-scope (reexported from
    #    Strumento 0.3's BASE exports — the extension-split parent namespace
    #    carries exactly these names; the two concrete socs are covered in
    #    sections 3 and 3b below).
    for f in (
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
    #    0.3's extension split defines MockSoc in StrumentoPiccoloExt, and on
    #    Julia 1.12 extension exports NEVER surface on the parent module — so
    #    `using Strumento` alone cannot reach it. The verified mechanism (the
    #    Strumento 0.3 load-configuration checks document these semantics): the
    #    extension is attached because Piccolo — its trigger — is a hard dep of
    #    this package, so src/Intonato.jl binds the type once, at top level, via
    #    Base.get_extension, and re-exports it as Intonato's own binding.
    @test isdefined(@__MODULE__, :MockSoc)
    @test isdefined(Intonato, :MockSoc)
    @test MockSoc <: AbstractSoc
    ext = Base.get_extension(Strumento, :StrumentoPiccoloExt)
    @test ext !== nothing
    @test Intonato.MockSoc === ext.MockSoc
    # The parent namespace stays clean: extension types never surface on it.
    @test !isdefined(Strumento, :MockSoc)

    # 3b. The delegation soc (StrumentoSoc) is NOT reachable from this package
    #     under 0.3 — by design, not omission: it lives in the PythonCall
    #     extension, this package's manifest carries no PythonCall, so that
    #     extension never attaches. The documented reach for users who need the
    #     real board: load PythonCall in their own environment, then
    #     `Base.get_extension(Strumento, :StrumentoPythonCallExt).StrumentoSoc`.
    @test !isdefined(Intonato, :StrumentoSoc)
    @test !isdefined(@__MODULE__, :StrumentoSoc)
    @test Base.get_extension(Strumento, :StrumentoPythonCallExt) === nothing

    # 4. The `duration` seam at Strumento 0.3 — RE-POINTED off the retired
    #    __init__ heal (src/Intonato.jl): the substrate now binds its
    #    pulse-sampling seam from the DEFINING module inside StrumentoPiccoloExt,
    #    so it needs no `Strumento.duration` binding at all. The heal-retirement
    #    pin: if the eval were ever reintroduced (or lingered invisibly), the
    #    first assertion below fails.
    @test !isdefined(Strumento, :duration)
    @test hasmethod(Strumento.pulse_duration, Tuple{Piccolo.AbstractPulse})
    # This package's OWN `duration` pin (src/Intonato.jl) is KEPT, not retired:
    # Piccolo's top-level reexport is still ambiguous against TimeWarp's
    # (NamedTrajectories ≥ 0.9.3; Piccolo #323 — `Piccolo.duration` is unbound
    # in fresh resolutions today), and this package reexports Piccolo's export
    # list, so the defining-module binding is what makes `Intonato.duration`
    # resolvable at all. Once Piccolo ships its fix the pin degrades to harmless
    # shadowing; the invariant below holds in both worlds.
    @test isdefined(Intonato, :duration)
    @test Intonato.duration === Piccolo.Quantum.Pulses.duration
end
