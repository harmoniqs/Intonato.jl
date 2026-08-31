# StrumentoExperiment — wraps a StrumentoBackend as a `HardwareExperiment`, the
# seam the QILC `PulseTuningProblem` chassis consumes. The `run` closure chains
# translate→upload→trigger→readout→discriminate into a `Vector{Measurement}`.
# Relocated here from Strumento.jl (its src/experiment.jl @ 12a05b4) with the
# v0.2 convention flip: the substrate's `iq_to_measurements` now returns the
# substrate-owned `Strumento.Measurement` record, while this package's chassis
# consumes THIS package's `Measurement` — the factory converts (trivially; the
# two record types are duck-compatible .data/.index twins) rather than re-typing
# the chassis.

@testitem "StrumentoExperiment produces valid, deterministic measurements" begin
    using Intonato
    import Strumento
    using LinearAlgebra
    MockSoc = Base.get_extension(Strumento, :StrumentoPiccoloExt).MockSoc
    σx = ComplexF64[0 1; 1 0]; σz = ComplexF64[1 0; 0 -1]
    sys = QuantumSystem(1.0 * σz, [σx], [1.0])
    N = 11
    pulse = LinearSplinePulse(0.1 .* randn(1, N), collect(range(0.0, 5.0, length = N)))
    map = QickChannelMap([QickGenChannel(0, 5e9; i_drive = 1)]; n_drives = 1)
    # `populations` here is Intonato's export (Strumento ≥ 0.2 keeps a substrate-
    # local UNexported twin precisely so this single public binding governs).
    model = MeasurementModel(:ψ̃, [populations], [N])

    soc = MockSoc(sys, ComplexF64[1, 0], ComplexF64[0, 1]; dac_rate = 50.0)
    backend = StrumentoBackend(soc, map, [N])
    qexp = StrumentoExperiment(backend; measurement_model = model)

    y1 = run_experiment(qexp, pulse)
    y2 = run_experiment(qexp, pulse)
    @test y1 isa Vector{<:Measurement}                # the CHASSIS record type …
    @test all(m isa Intonato.Measurement for m in y1) # … is Intonato's own …
    @test !(y1[1] isa Strumento.Measurement)          # … not the substrate twin.
    @test length(y1) == 1 && y1[1].index == N
    @test sum(y1[1].data) ≈ 1.0 atol = 1e-6
    @test y1[1].data ≈ y2[1].data                      # deterministic
end
