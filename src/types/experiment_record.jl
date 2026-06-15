# ============================================================================ #
#                          ExperimentRecord
# ============================================================================ #

"""
    ExperimentRecord

Provenance-rich record of a single experiment run: the pulse that was executed,
the measurement model used to read the device, the resulting measurements, the
optional raw readout, and a `metadata` NamedTuple capturing everything needed
to reproduce the run.

This is the public hardware-data-collection seam: a `SimulatedExperiment` or a
hardware backend produces a `Vector{Measurement}`, and an
[`AbstractExperimentLogger`](@ref) wraps it into an `ExperimentRecord` for
persistence (catalog/vault, TOML/JLD2, …).

# Fields
- `pulse::AbstractPulse`: the pulse that was run on the device.
- `measurement_model::MeasurementModel`: what was measured and at which knots.
- `measurements::Vector{Measurement}`: the measurement outputs (one per knot index).
- `raw::Any = nothing`: optional raw readout (shot counts, IQ blobs, …).
- `metadata::NamedTuple = (;)`: provenance. Recommended keys (none enforced):
    - `device`: device/backend id (e.g. `"sim"`, `"qick-rfsoc-01"`).
    - `shots`: per-measurement shot counts (e.g. `[1000, 1000]`).
    - `basis`: measurement basis (e.g. `:Z`).
    - `timestamp`: ISO-8601 acquisition time.
    - `config_hash`: hash of the hardware/backend configuration.
    - `calibration`: calibration label (e.g. `"nominal"`).
    - `seed`: RNG seed used for any stochastic (shot-noise) sampling.
    - `versions`: package/Manifest version string for reproducibility.
    - `pulse_hash`: hash of the pulse knot data.

`metadata` is a free-form `NamedTuple` so backends can attach device-specific
provenance without a schema change; the recommended keys above are what the
follow-on vault/catalog logger (separate plan) will read.
"""
Base.@kwdef struct ExperimentRecord
    pulse::AbstractPulse
    measurement_model::MeasurementModel
    measurements::Vector{Measurement}
    raw::Any = nothing
    metadata::NamedTuple = (;)
end

@testitem "ExperimentRecord captures pulse + measurements + provenance" begin
    using Intonato

    # Single-qubit Rabi fixture (mirrors src/device_models/abstract.jl).
    N = 11
    T = 5.0
    times = range(0.0, T, length=N) |> collect
    somepulse = LinearSplinePulse(0.1 * ones(1, N), times)
    model = MeasurementModel(:ψ̃, [populations], [N])
    ms = [Measurement([0.7, 0.3], N)]

    rec = ExperimentRecord(
        pulse = somepulse,
        measurement_model = model,
        measurements = ms,
        raw = nothing,
        metadata = (; device="sim", shots=[1000, 1000], basis=:Z,
                      timestamp="2026-06-14T00:00:00", config_hash="abc",
                      calibration="nominal", seed=42, versions="Manifest:def",
                      pulse_hash="ghi"),
    )

    @test rec.pulse === somepulse
    @test rec.measurement_model === model
    @test length(rec.measurements) == length(ms)
    @test rec.measurements[1].data == [0.7, 0.3]
    @test rec.raw === nothing
    @test rec.metadata.seed == 42
    @test rec.metadata.shots == [1000, 1000]

    # `raw` and `metadata` are optional (kwdef defaults).
    rec2 = ExperimentRecord(
        pulse = somepulse,
        measurement_model = model,
        measurements = ms,
    )
    @test rec2.raw === nothing
    @test rec2.metadata == (;)
end
