# QickProgram + pulse_to_envelopes — the pulse → QICK waveform translation.
#
# Samples a Piccolo pulse onto each generator channel's DAC grid (via Piccolo
# `sample`) and packs the routed controls into complex (idata, qdata) envelopes
# plus the metadata a board needs to play and read them back.
#
# Moved from Strumento.jl (its src/translate.jl + the typed method from its
# Piccolo extension, Intonato #55) — behaviour unchanged. The pulse duration is
# read through the package-level `duration` pin (src/Intonato.jl, Piccolo #323):
# the defining-module binding, the same function object the golden pins were
# captured against.

"""
    QickProgram

A device-agnostic description of a played pulse:
- `times` — the DAC-grid sample times (s).
- `envelopes` — `gen_ch => (idata, qdata)` complex envelope samples.
- `carrier_freqs` — `gen_ch => carrier frequency (Hz)`.
- `routing` — `(gen_ch, i_drive, q_drive)` per channel (so a SoC can invert
  envelopes back to drive controls — used by `MockSoc`).
- `n_drives` — control count of the source pulse.
- `indices` — measurement knot indices (into `1:N`) the readout should produce.
"""
struct QickProgram
    times::Vector{Float64}
    envelopes::Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}
    carrier_freqs::Dict{Int,Float64}
    routing::Vector{Tuple{Int,Int,Union{Int,Nothing}}}
    n_drives::Int
    indices::Vector{Int}
end

# Default per-gen-channel envelope sample-memory cap (typical QICK firmware is
# O(few k) samples per generator). Configurable per call.
const DEFAULT_MAX_ENVELOPE_LEN = 16_384

"""
    pulse_to_envelopes(pulse, map, dac_rate, indices; max_len=DEFAULT_MAX_ENVELOPE_LEN) → QickProgram

Sample `pulse` (a Piccolo `AbstractPulse`) onto the DAC grid
(`0 : 1/dac_rate : duration`) and route each control onto its generator
channel's I/Q envelope per `map`. Errors if the envelope would exceed `max_len`
samples (envelope-memory limit).
"""
function pulse_to_envelopes(
    pulse::AbstractPulse,
    map::QickChannelMap,
    dac_rate::Real,
    indices::Vector{Int};
    max_len::Int = DEFAULT_MAX_ENVELOPE_LEN,
)
    Piccolo.Quantum.Pulses.n_drives(pulse) == map.n_drives || error(
        "pulse has $(Piccolo.Quantum.Pulses.n_drives(pulse)) drives but channel map " *
        "expects $(map.n_drives)",
    )
    T = duration(pulse)
    nsamp = floor(Int, T * dac_rate) + 1
    nsamp ≤ max_len ||
        error("envelope length $nsamp exceeds max $max_len at dac_rate=$dac_rate, T=$T")
    times = collect(range(0.0, T, length = nsamp))
    ctrls = Piccolo.Quantum.Pulses.sample(pulse, times)    # (n_drives, nsamp)

    envelopes = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
    carrier_freqs = Dict{Int,Float64}()
    routing = Tuple{Int,Int,Union{Int,Nothing}}[]
    for ch in map.channels
        idata = Vector{Float64}(ctrls[ch.i_drive, :])
        qdata =
            ch.q_drive === nothing ? zeros(Float64, nsamp) :
            Vector{Float64}(ctrls[ch.q_drive, :])
        envelopes[ch.gen_ch] = (idata, qdata)
        carrier_freqs[ch.gen_ch] = ch.carrier_freq
        push!(routing, (ch.gen_ch, ch.i_drive, ch.q_drive))
    end
    return QickProgram(times, envelopes, carrier_freqs, routing, map.n_drives, indices)
end

# Anything that is not a Piccolo pulse is named actionably (not a bare MethodError).
pulse_to_envelopes(pulse, args...; kwargs...) = error(
    "Intonato.pulse_to_envelopes: no translation method for $(typeof(pulse)) — " *
    "the translation samples a Piccolo AbstractPulse",
)

# Reconstruct the played drive controls from a soc's loaded per-channel
# envelopes (the inversion MockSoc and TwinSoc both perform, so the played pulse
# is identical across the socs).
function _played_pulse(prog::QickProgram, env::AbstractDict)
    nsamp = length(prog.times)
    ctrls = zeros(Float64, prog.n_drives, nsamp)
    for (gen_ch, i_drive, q_drive) in prog.routing
        idata, qdata = env[gen_ch]
        ctrls[i_drive, :] .= idata
        q_drive === nothing || (ctrls[q_drive, :] .= qdata)
    end
    return LinearSplinePulse(ctrls, prog.times)
end

@testitem "pulse_to_envelopes errors actionably on a non-pulse; QickProgram contract" begin
    using Intonato
    @test_throws ErrorException pulse_to_envelopes("not a pulse", nothing, 0.0, [0])
    @test fieldtypes(QickProgram) == (
        Vector{Float64},
        Dict{Int,Tuple{Vector{Float64},Vector{Float64}}},
        Dict{Int,Float64},
        Vector{Tuple{Int,Int,Union{Int,Nothing}}},
        Int,
        Vector{Int},
    )
end

@testitem "pulse_to_envelopes samples + routes correctly" begin
    using Intonato
    using LinearAlgebra
    N = 11
    T = 5.0
    times = collect(range(0.0, T, length = N))
    vals = 0.1 .* randn(2, N)
    pulse = LinearSplinePulse(vals, times)
    # drive 1 → ch0 I, drive 2 → ch0 Q (one complex drive on one channel)
    map = QickChannelMap([QickGenChannel(0, 5e9; i_drive = 1, q_drive = 2)]; n_drives = 2)
    dac_rate = 10.0   # 10 Hz → 51 samples over T=5
    prog = pulse_to_envelopes(pulse, map, dac_rate, [N])
    @test length(prog.times) == 51
    @test haskey(prog.envelopes, 0)
    idata, qdata = prog.envelopes[0]
    @test length(idata) == 51 && length(qdata) == 51
    # idata/qdata equal the pulse's two controls sampled at the DAC grid.
    @test idata ≈ [pulse(t)[1] for t in prog.times]
    @test qdata ≈ [pulse(t)[2] for t in prog.times]
    @test prog.carrier_freqs[0] == 5e9
end

@testitem "pulse_to_envelopes enforces envelope memory cap" begin
    using Intonato
    N = 11
    T = 5.0
    pulse = LinearSplinePulse(0.1 .* randn(1, N), collect(range(0.0, T, length = N)))
    map = QickChannelMap([QickGenChannel(0, 5e9; i_drive = 1)]; n_drives = 1)
    @test_throws ErrorException pulse_to_envelopes(pulse, map, 1e6, [N]; max_len = 1000)
end

@testitem "pulse_to_envelopes rejects drive-count mismatch" begin
    using Intonato
    pulse = LinearSplinePulse(0.1 .* randn(1, 11), collect(range(0.0, 5.0, length = 11)))
    map = QickChannelMap([QickGenChannel(0, 5e9; i_drive = 1, q_drive = 2)]; n_drives = 2)
    @test_throws ErrorException pulse_to_envelopes(pulse, map, 10.0, [11])
end
