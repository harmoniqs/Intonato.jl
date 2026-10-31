# MockSoc — a pure-Julia "board" that simulates QICK execution by rolling the
# played pulse through a known `QuantumSystem` (Piccolo-native propagation:
# `rollout` on a `KetTrajectory`) and emitting synthetic IQ. The forward model is
# explicit:
#   state → IQ blob = measurement_fn(state) packed as a real-valued complex vector,
# which the trivial discriminator `real` inverts EXACTLY. So a QILC loop run
# through `StrumentoBackend{MockSoc}` reproduces the same measurements a direct
# simulated rollout would — the loop is validated without a board.
#
# The user passes the "true" (optionally mismatched) system as the mock's system;
# the nominal QCP is solved against the nominal system separately.
#
# Moved from Strumento.jl (its Piccolo extension, Intonato #55) — behaviour
# unchanged, including the known limitation tracked in Intonato #51 (the ket
# rollout ignores any dissipators). The default forward model is this package's
# `populations` (the identical function the substrate kept a private twin of).

"""
    MockSoc(system, ψ_init, ψ_goal; measurement_fn=populations, dac_rate=1.0, adc_rate=1.0)

A simulated QICK SoC backed by `system`. `play_program!`/`acquire` reconstruct the
played pulse from the loaded envelopes and roll it out via Piccolo's `rollout`
(`KetTrajectory` propagation through `system`), returning IQ blobs
`measurement_fn(state)` (packed as complex; invert with `real`).
"""
mutable struct MockSoc <: AbstractSoc
    system::QuantumSystem
    ψ_init::Vector{ComplexF64}
    ψ_goal::Vector{ComplexF64}
    measurement_fn::Function
    dac_rate::Float64
    adc_rate::Float64
    _env::Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}
    _program::Union{Nothing,QickProgram}
end

function MockSoc(
    system::QuantumSystem,
    ψ_init::AbstractVector,
    ψ_goal::AbstractVector;
    measurement_fn::Function = populations,
    dac_rate::Real = 1.0,
    adc_rate::Real = 1.0,
)
    return MockSoc(
        system,
        ComplexF64.(ψ_init),
        ComplexF64.(ψ_goal),
        measurement_fn,
        Float64(dac_rate),
        Float64(adc_rate),
        Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}(),
        nothing,
    )
end

dac_rate(soc::MockSoc) = soc.dac_rate
adc_rate(soc::MockSoc) = soc.adc_rate

load_envelope!(soc::MockSoc, gen_ch::Int, idata, qdata) =
    (soc._env[gen_ch] = (Vector{Float64}(idata), Vector{Float64}(qdata)); nothing)

play_program!(soc::MockSoc, program::QickProgram) = (soc._program = program; nothing)

# execute! — the AbstractSoc verb: translate the pulse in Julia (QICK-shaped
# envelopes), then roll it out. This is the board-free path the QILC loop tests.
function execute!(
    soc::MockSoc,
    pulse::AbstractPulse,
    channel_map::QickChannelMap,
    indices::Vector{Int},
)
    prog = pulse_to_envelopes(pulse, channel_map, dac_rate(soc), indices)
    for (gen_ch, (idata, qdata)) in prog.envelopes
        load_envelope!(soc, gen_ch, idata, qdata)
    end
    play_program!(soc, prog)
    return acquire(soc, channel_map.readout_chs)
end

function acquire(soc::MockSoc, _ro_chs)
    prog = soc._program
    prog === nothing && error("MockSoc.acquire: no program played")
    # Reconstruct the drive controls from the loaded per-channel envelopes.
    recon = _played_pulse(prog, soc._env)
    # Piccolo-native propagation: propagate ψ through the system under the
    # reconstructed controls, then evaluate the forward model at each
    # measurement knot (indices index into the reconstructed pulse's DAC-grid
    # knot times). Pinned by the golden testitem below.
    qtraj = rollout(KetTrajectory(soc.system, recon, soc.ψ_init, soc.ψ_goal), recon)
    knot_times = Piccolo.Quantum.Pulses.get_knot_times(recon)
    # Forward model state→IQ: pack each measurement's data as a complex blob.
    return [
        ComplexF64.(soc.measurement_fn(ket_to_iso(qtraj(knot_times[k])))) for
        k in prog.indices
    ]
end

@testitem "MockSoc round-trips a pulse to valid populations IQ" begin
    using Intonato
    using LinearAlgebra
    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    sys = QuantumSystem(1.0 * σz, [σx], [1.0])
    N = 11
    T = 5.0
    times = collect(range(0.0, T, length = N))
    pulse = LinearSplinePulse(0.1 .* randn(1, N), times)
    map = QickChannelMap([QickGenChannel(0, 5e9; i_drive = 1)]; n_drives = 1)

    soc = MockSoc(sys, ComplexF64[1, 0], ComplexF64[0, 1]; dac_rate = 20.0)
    prog = pulse_to_envelopes(pulse, map, dac_rate(soc), [N])
    for (gen_ch, (idata, qdata)) in prog.envelopes
        load_envelope!(soc, gen_ch, idata, qdata)
    end
    play_program!(soc, prog)
    raw = acquire(soc, [0])

    @test length(raw) == 1                 # one measurement (final knot)
    pops = real.(raw[1])
    @test length(pops) == 2                # dim-2 populations
    @test sum(pops) ≈ 1.0 atol = 1e-6      # valid probability vector
    @test all(pops .≥ -1e-9)
    # acquiring before any program is played errors actionably
    @test_throws ErrorException acquire(
        MockSoc(sys, ComplexF64[1, 0], ComplexF64[0, 1]),
        [0],
    )
end

@testitem "MockSoc IQ forward model is pinned to golden values" begin
    using Intonato
    # GOLDEN PIN — captured from the SimulatedExperiment rollout (the
    # pre-re-grounding forward model, Piccolo 2.0.2, Julia 1.12) for this FIXED
    # fixture: 2-drive system, deterministic analytic I/Q pulse, one
    # complex-envelope gen channel, two measurement knots (DAC-grid samples 11
    # and 101), dac_rate = 20 Hz. A real behaviour change must fail loudly here.
    #
    # TRANSPORT TOLERANCE: the literals are bit-exact on the capture box; CI
    # runner hardware rotation (multi-microarch vectorized paths, FMA-fusion
    # differences) has drifted them 1–5 ulp. The literal comparison therefore
    # carries isapprox(rtol=1e-13, atol=1e-15): ~200× above the largest observed
    # drift (5.1e-16 relative), ~4 orders of magnitude below any semantic
    # forward-model change (≥1e-9).
    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    sys = QuantumSystem(1.0 * σz, [σx, σx], [1.0, 1.0])
    N = 11
    T = 5.0
    times = collect(range(0.0, T, length = N))
    vals =
        0.1 .* permutedims(
            hcat(sin.(range(0.0, 2.4π, length = N)), cos.(range(0.3π, 1.7π, length = N))),
        )
    pulse = LinearSplinePulse(vals, times)
    map = QickChannelMap([QickGenChannel(0, 5e9; i_drive = 1, q_drive = 2)]; n_drives = 2)

    soc = MockSoc(sys, ComplexF64[1, 0], ComplexF64[0, 1]; dac_rate = 20.0)
    raw = execute!(soc, pulse, map, [11, 101])

    @test isapprox(
        raw,
        [
            ComplexF64[0.9987748357943047+0.0im, 0.0012251642056963555+0.0im],
            ComplexF64[0.9552991750369558+0.0im, 0.044700824963045074+0.0im],
        ];
        rtol = 1e-13,
        atol = 1e-15,
    )
end
