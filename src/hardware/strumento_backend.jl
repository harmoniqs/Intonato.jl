# StrumentoBackend — the AbstractHardwareBackend over an AbstractSoc (Strumento ≥ 0.2),
# relocated here from Strumento.jl (its src/backend.jl @ 12a05b4) when Strumento's
# v0.2 release severed the inverted dependency edge: the calibration-loop chassis
# (this package) now sits ABOVE the soc substrate and owns the hardware seam.
#
# The four backend verbs — upload_pulse! / trigger! / readout / sample_rate — are
# declared AND exported by this package (types/hardware_backends.jl), so the adapter
# extends them natively: same module, same function objects. The identity testitem
# below pins that (the shadowing hazard behind Strumento PR #13 is
# location-independent, so the identity assertion travels with the seam).
#
# The soc owns *translation*: a MockSoc (Strumento's Piccolo extension) translates
# the pulse in Julia (board-free rollout), while a StrumentoSoc hands the pulse to
# Python `strumento`, which owns the pulse-IR → AveragerProgramV2 → acquire → reduce
# path. So `execute!` is the one verb each soc implements; the backend just chains
# upload → trigger → readout onto it. The QILC chassis never calls these directly;
# the StrumentoExperiment `run` closure (strumento_experiment.jl) does, once per
# experiment evaluation.

@testitem "StrumentoBackend upload/trigger/readout against MockSoc" begin
    using Intonato
    import Strumento  # module handle only: MockSoc is extension-defined
    using LinearAlgebra
    # Extension-defined types never surface on the parent namespace (Julia 1.12):
    # reach MockSoc through its canonical extension handle.
    MockSoc = Base.get_extension(Strumento, :StrumentoPiccoloExt).MockSoc
    σx = ComplexF64[0 1; 1 0]; σz = ComplexF64[1 0; 0 -1]
    sys = QuantumSystem(1.0 * σz, [σx], [1.0])
    N = 11
    pulse = LinearSplinePulse(0.1 .* randn(1, N), collect(range(0.0, 5.0, length = N)))
    map = QickChannelMap([QickGenChannel(0, 5e9; i_drive = 1)]; n_drives = 1)
    soc = MockSoc(sys, ComplexF64[1, 0], ComplexF64[0, 1]; dac_rate = 20.0)
    b = StrumentoBackend(soc, map, [N])

    @test b.last_raw === nothing
    upload_pulse!(b, pulse)
    trigger!(b)
    raw = readout(b)
    @test b.last_raw === raw
    @test sum(real.(raw[1])) ≈ 1.0 atol = 1e-6
    @test sample_rate(b) == 20.0
end

@testitem "Backend generics are Intonato's own function objects (seam contract)" begin
    using Intonato
    import Strumento
    # The AbstractHardwareBackend contract — upload_pulse! / trigger! / readout /
    # sample_rate — is declared AND exported by Intonato (types/hardware_backends.jl).
    # The relocated StrumentoBackend extends those very generics: same module, same
    # function objects. It must never gain shadow declarations — or the contract
    # silently splits into two function objects and works only by accident
    # (the Strumento PR #13 hazard; the identity assertion travels with the seam).
    for f in (:upload_pulse!, :trigger!, :readout, :sample_rate)
        @test f in names(Intonato)                # Intonato owns the export …
        @test isdefined(Intonato, f)              # … and the binding is live …
        @test !isdefined(Strumento, f)            # … with no substrate-side twin.
    end
    # The relocated adapter's methods live on Intonato's own function objects.
    @test hasmethod(Intonato.upload_pulse!, Tuple{StrumentoBackend, AbstractPulse})
    @test hasmethod(Intonato.trigger!, Tuple{StrumentoBackend})
    @test hasmethod(Intonato.readout, Tuple{StrumentoBackend})
    @test hasmethod(Intonato.sample_rate, Tuple{StrumentoBackend})
end
