# StrumentoExperiment — wraps a StrumentoBackend as a `HardwareExperiment`, the
# seam the QILC `PulseTuningProblem` chassis consumes. The `run` closure chains
# translate→upload→trigger→readout→discriminate into a `Vector{Measurement}`.
# Readout conversion (`iq_to_measurements`, src/hardware/readout.jl) returns this
# package's `Measurement` directly — there is one Measurement type, so the seam
# does no record translation.

"""
    StrumentoExperiment(backend; measurement_model) → HardwareExperiment

Build a `HardwareExperiment` whose `run(pulse)` uploads/plays `pulse` on
`backend`'s SoC and discriminates the readout into measurements. The
`measurement_model` annotates logged `ExperimentRecord`s with true provenance
(rather than the identity placeholder); its indices must match `backend.indices`.
Plug the result into `PulseTuningProblem(qcp, qexp, model; …)`.
"""
function StrumentoExperiment(backend::StrumentoBackend; measurement_model::MeasurementModel)
    run =
        pulse -> begin
            upload_pulse!(backend, pulse)
            trigger!(backend)
            raw = readout(backend)
            return iq_to_measurements(raw, backend.discriminator, backend.indices)
        end
    return HardwareExperiment(run, measurement_model)
end

@testitem "StrumentoExperiment produces valid, deterministic measurements" begin
    using Intonato
    using LinearAlgebra
    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    sys = QuantumSystem(1.0 * σz, [σx], [1.0])
    N = 11
    pulse = LinearSplinePulse(0.1 .* randn(1, N), collect(range(0.0, 5.0, length = N)))
    map = QickChannelMap([QickGenChannel(0, 5e9; i_drive = 1)]; n_drives = 1)
    # `populations` here is Intonato's export — also MockSoc's default forward model.
    model = MeasurementModel(:ψ̃, [populations], [N])

    soc = MockSoc(sys, ComplexF64[1, 0], ComplexF64[0, 1]; dac_rate = 50.0)
    backend = StrumentoBackend(soc, map, [N])
    qexp = StrumentoExperiment(backend; measurement_model = model)

    y1 = run_experiment(qexp, pulse)
    y2 = run_experiment(qexp, pulse)
    @test y1 isa Vector{Measurement}                  # the ONE record type …
    @test all(m isa Intonato.Measurement for m in y1) # … Intonato's own, as returned
    @test parentmodule(Measurement) === Intonato      # by the readout conversion.
    @test length(y1) == 1 && y1[1].index == N
    @test sum(y1[1].data) ≈ 1.0 atol = 1e-6
    @test y1[1].data ≈ y2[1].data                      # deterministic
end
