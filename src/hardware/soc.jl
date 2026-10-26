# AbstractSoc — the abstraction over a QICK board controller. The mock
# (`MockSoc`, pure Julia), the twin face (`Intonato.Twins.TwinSoc`) and the real
# proxy (`StrumentoSoc`, delegating to Python strumento) all implement it, so
# `StrumentoBackend` is written once against this interface.
#
# Moved from Strumento.jl (its src/soc.jl, Intonato #55) — behaviour unchanged.

"""
    AbstractSoc

Abstraction over a board controller. The one verb every soc implements is
**`execute!`** — it owns *translation*, so the soc kinds differ exactly where
they should:

- `execute!(soc, pulse, channel_map, indices) → raw` — translate + run `pulse`,
  returning raw per-measurement IQ blobs (aligned with `indices`). `MockSoc`
  translates in Julia and rolls out a `QuantumSystem`; a delegation soc hands the
  pulse to Python `strumento` (which owns the pulse-IR → program → acquire path).
- `dac_rate(soc) → Float64`, `adc_rate(soc) → Float64` — sample rates (Hz).

The lower-level `load_envelope!` / `play_program!` / `acquire` verbs are the
MockSoc's internal machinery (a board-free QICK-shaped translation); a soc that
delegates translation elsewhere need not implement them.
"""
abstract type AbstractSoc end

# Generic fallbacks give an actionable error if a subtype forgets a method.
execute!(soc::AbstractSoc, args...) = error("execute! not implemented for $(typeof(soc))")
load_envelope!(soc::AbstractSoc, args...) =
    error("load_envelope! not implemented for $(typeof(soc))")
play_program!(soc::AbstractSoc, args...) =
    error("play_program! not implemented for $(typeof(soc))")
acquire(soc::AbstractSoc, args...) = error("acquire not implemented for $(typeof(soc))")
dac_rate(soc::AbstractSoc) = error("dac_rate not implemented for $(typeof(soc))")
adc_rate(soc::AbstractSoc) = error("adc_rate not implemented for $(typeof(soc))")

@testitem "AbstractSoc interface fallbacks error" begin
    using Intonato
    struct _BareSoc <: Intonato.AbstractSoc end
    s = _BareSoc()
    @test_throws ErrorException load_envelope!(s, 0, [1.0], [0.0])
    @test_throws ErrorException play_program!(s)
    @test_throws ErrorException acquire(s, [0])
    @test_throws ErrorException adc_rate(s)
    @test_throws ErrorException dac_rate(s)
    @test_throws ErrorException execute!(s, 0)
end

# ──── Pulse-sampling seam ─────────────────────────────────────────────────────
# A delegation soc samples a played pulse through this seam rather than naming
# Piccolo's sampling API itself. The typed methods bind `duration` from its
# DEFINING module (the package-level pin in src/Intonato.jl — Piccolo #323: the
# top-level reexport of `duration` is ambiguous against NamedTrajectories'
# TimeWarp export). Anything that is not a Piccolo pulse errors actionably.

pulse_duration(pulse::AbstractPulse) = duration(pulse)
sample_controls(pulse::AbstractPulse, times) = Piccolo.Quantum.Pulses.sample(pulse, times)

pulse_duration(pulse) = error(
    "Intonato.pulse_duration: no pulse-sampling method for $(typeof(pulse)) — " *
    "the seam samples a Piccolo AbstractPulse",
)

sample_controls(pulse, times) = error(
    "Intonato.sample_controls: no pulse-sampling method for $(typeof(pulse)) — " *
    "the seam samples a Piccolo AbstractPulse",
)

@testitem "pulse-sampling seam: typed on AbstractPulse, actionable otherwise" begin
    using Intonato
    struct _DuckPulse end
    @test_throws ErrorException Intonato.pulse_duration(_DuckPulse())
    @test_throws ErrorException Intonato.sample_controls(_DuckPulse(), 0.0:1.0)
    @test hasmethod(Intonato.pulse_duration, Tuple{AbstractPulse})
    @test hasmethod(Intonato.sample_controls, Tuple{AbstractPulse,Vector{Float64}})
    pulse = LinearSplinePulse(
        0.1 .* permutedims(collect(range(0.0, 1.0, length = 5))),
        collect(range(0.0, 2.0, length = 5)),
    )
    @test Intonato.pulse_duration(pulse) == 2.0
    @test Intonato.sample_controls(pulse, [0.0, 2.0]) ≈ [0.0 0.1]
end
