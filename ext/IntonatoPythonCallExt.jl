# IntonatoPythonCallExt — the PythonCall-triggered package extension (Intonato #56).
#
# The real-board path: `StrumentoSoc` hands a solved pulse to the Python
# `strumento` framework over PythonCall, and `StrumentoDCSource` maps the slow-DC
# verbs onto Python strumento's DC seam (`strumento.core.dc`). Python strumento
# is the single authority for the device model, pulse IR, compiler and wire
# format; this extension is the thin Julia binding to it.
#
# The extension loads exactly when PythonCall is loaded; the base package
# references no PythonCall name, so `using Intonato` never needs Python. Julia's
# extension semantics (1.12): extension exports do NOT surface on the parent
# module — the soc contract verbs the base declares (`execute!`, `dac_rate`,
# `adc_rate`, `set_gate!`, `get_gate`, `gate_snapshot`) gain their methods here,
# so bare-name calls work in every configuration; the extension-defined TYPES are
# reached through `Base.get_extension(Intonato, :IntonatoPythonCallExt)`.
#
# The pulse is sampled through the base pulse-sampling seam
# (`Intonato.pulse_duration` / `Intonato.sample_controls`, typed on Piccolo's
# `AbstractPulse`; anything else errors actionably).
#
# Moved from Strumento.jl (its StrumentoPythonCallExt, Intonato #56) — behaviour
# unchanged. Testitems in this file guard on `Base.identify_package("PythonCall")`
# and on Python `strumento` being importable, skipping cleanly otherwise.
module IntonatoPythonCallExt

import Intonato
import Intonato: AbstractSoc, execute!, dac_rate, adc_rate, QickChannelMap,
    DCAxisMap, set_gate!, get_gate, gate_snapshot
using TestItems

using PythonCall

export StrumentoSoc
export StrumentoDCSource

# ──── StrumentoSoc ───────────────────────────────────────────────────────────
# The real-board path, delegating to the Python `strumento` package over
# PythonCall. Julia does NOT assemble an AveragerProgramV2 itself; it hands the
# solved pulse to Python `strumento`, which owns the pulse-IR → compile → acquire
# → reduce path. One source of truth (Python), a thin Julia binding here.
#
# `strumento` is imported LAZILY inside the constructor (runtime, not load-time),
# so loading this extension never initializes the interpreter and the whole
# MockSoc path runs without Python.

"""
    StrumentoSoc(device; drive_map, dac_rate, adc_rate, board=nothing)

Real board reached by delegating to the Python `strumento` framework via PythonCall
(available when PythonCall is loaded, as
`Base.get_extension(Intonato, :IntonatoPythonCallExt).StrumentoSoc`).
`device` is a strumento device-instance YAML path (or a Python `Device` handle);
`drive_map` maps each Piccolo drive index to a `(drive_index, line, role, carrier_mhz)`
tuple naming the strumento wiring line/role and its carrier. On `execute!`, the played
pulse is handed to `strumento.from_solution` and run through a `StrumentoProgram` on the
board — Julia never assembles the program.

`strumento` (and `qick`) are imported lazily here; an actionable error is raised if the
Python package is unavailable. The delegation path is exercised end-to-end against the
Python-side mock (`MockQickSocV2`) by a test that runs when Python + strumento are
importable and skips cleanly otherwise — the pure-Julia CI lane stays pure. Real-board
acquisition remains a hardware session.
"""
mutable struct StrumentoSoc <: AbstractSoc
    strumento::Py          # the imported `strumento` module
    device::Py             # a strumento Device (Python)
    board::Py              # the board soc (a qick soc / strumento MockQickSocV2), or py-None
    drive_map::Vector{Tuple{Int,String,String,Float64}}   # (drive_index, line, role, carrier_mhz)
    dac_rate::Float64
    adc_rate::Float64
end

function StrumentoSoc(device; drive_map::Vector{<:Tuple}, dac_rate::Real, adc_rate::Real,
                      board = nothing)
    st = try
        pyimport("strumento")
    catch e
        error("StrumentoSoc requires the Python `strumento` package in the board's " *
              "Python environment (the open QICK tProc-v2 framework this soc delegates " *
              "to). `pyimport(\"strumento\")` failed: $e")
    end
    dev = device isa Py ? device : st.Device.load(string(device))
    bd = board === nothing ? pybuiltins.None : board
    dm = Tuple{Int,String,String,Float64}[
        (Int(d), String(line), String(role), Float64(f)) for (d, line, role, f) in drive_map
    ]
    return StrumentoSoc(st, dev, bd, dm, Float64(dac_rate), Float64(adc_rate))
end

dac_rate(soc::StrumentoSoc) = soc.dac_rate
adc_rate(soc::StrumentoSoc) = soc.adc_rate

# execute! — hand the pulse to Python `strumento` (from_solution → program → acquire →
# reduce): strumento owns translation. Sampling goes through the base pulse-sampling seam.
#
# Knot semantics (the AbstractSoc contract; see MockSoc): `indices` are MeasurementModel
# knots into 1:N, and each knot's blob must be the IQ measured AT that knot's time. The
# delegated path honors that with ONE PROGRAM PER KNOT: the pulse is handed to
# `from_solution` TRUNCATED at the knot (`times[1:k]`, `controls[:, 1:k]`), a `Measure` op
# is appended (`from_solution` plays pulses only — the compiler declares readout channels
# only when it sees one), the program compiles and acquires, and the averaged IQ becomes
# that knot's blob.
function execute!(soc::StrumentoSoc, pulse, ::QickChannelMap,
                  indices::Vector{Int})
    T = Intonato.pulse_duration(pulse)
    nsamp = floor(Int, T * soc.dac_rate) + 1
    times = collect(range(0.0, T, length = nsamp))
    ctrls = Intonato.sample_controls(pulse, times)    # (n_drives, nsamp)

    LineRef = soc.strumento.LineRef
    py_drive_map = pydict()
    for (d, line, role, carrier) in soc.drive_map
        py_drive_map[line] = pytuple((LineRef(line, role), carrier))
    end

    StrumentoProgram = pyimport("strumento.core.program").StrumentoProgram
    from_solution = pyimport("strumento.core.pulses").from_solution
    gain_calib = _pi_pulse_reference(soc)

    blobs = Vector{Vector{ComplexF64}}(undef, length(indices))
    for (j, k) in enumerate(indices)
        (2 ≤ k ≤ nsamp) || error("StrumentoSoc.execute!: knot index $k outside the " *
            "playable pulse ($nsamp samples); knot 1 (t=0, before any evolution) has no " *
            "delegated readout — the mock path owns that case")
        knot_solution = pydict([
            "times" => pylist(times[1:k]),
            "controls" => pydict([line => pylist(ctrls[d, 1:k])
                                  for (d, line, role, _) in soc.drive_map]),
        ])
        seq = from_solution(knot_solution; drive_map = py_drive_map,
                            gain_calib = gain_calib, dac_rate = soc.dac_rate)
        seq.measure()   # pulses-only seq compiles with no readout channels otherwise
        prog = StrumentoProgram(soc.device; seq = seq, soc = soc.board)
        result = prog.acquire(soc.board, progress = false)
        blobs[j] = [_averaged_iq(result)]
    end
    return blobs
end

# The physical-amplitude → v2-fraction reference `from_solution` needs: read from the
# loaded device's own calibration, never a Julia-supplied constant.
#
# The gain FRACTION comes from the device's own factory (`dev.qubit.ge_pi()`): Python owns
# the int-code → fraction conversion and the soccfg `maxv` resolution, so this side never
# re-implements them. The Rabi rate uses the mean-rate convention from the stored π-pulse
# timing: a π rotation over the played window gives Ω/2π = 0.5 / T_π (MHz). This fixes a
# LINEAR ruler (frac per MHz) — exact for pulses of the calibration pulse's shape class;
# the shape-exact reference is a hardware-session refinement.
function _pi_pulse_reference(soc::StrumentoSoc)
    pi_pulse = soc.device.qubit.ge_pi()
    gain_frac = pyconvert(Float64, pi_pulse.gain)
    pc = soc.device.calib.qubit.pulses["pi_ge"]
    length_us = pyconvert(Float64, pc.length)
    sigma_us = pyconvert(Float64, pc.sigma)
    T_pi = length_us > 0 ? length_us : 6 * sigma_us   # gauss: played window ≈ 6σ
    T_pi > 0 || error("StrumentoSoc: the calibration π pulse has no usable duration " *
                      "(length=$length_us µs, sigma=$sigma_us µs)")
    return pyimport("strumento.packs.cqed.calibration").PiPulseReference(
        pi_gain_frac = gain_frac, pi_rabi_mhz = 0.5 / T_pi)
end

# A single readout channel's averaged acquisition → one IQ sample. With no swept axis the
# result is 1-D — shape (n_reads,) = (1,) on this path; the readout channel exists because
# `Measure` was appended (its absence fails earlier, at compile). astype first: the
# acquisition may carry float32 (the ADC's native width), which pyconvert will not widen.
function _averaged_iq(result::Py)
    np = pyimport("numpy")
    avgi = pyconvert(Vector{Float64}, np.asarray(result.avgi, dtype = "float64").reshape(-1))
    avgq = pyconvert(Vector{Float64}, np.asarray(result.avgq, dtype = "float64").reshape(-1))
    length(avgi) == 1 || error("StrumentoSoc: expected a single averaged readout, got " *
                               "avgi with $(length(avgi)) entries")
    return ComplexF64(avgi[1], avgq[1])
end

@testitem "StrumentoSoc is an AbstractSoc (type only; no Python/board needed)" begin
    using Intonato
    if Base.identify_package("PythonCall") === nothing
        @info "skipping: no PythonCall in this environment (PythonCall-extension surface)"
        @test true
    else
        using PythonCall
        # The delegation type is extension-defined: extension exports do not
        # surface on the parent module, so reach it through its canonical handle.
        StrumentoSoc = Base.get_extension(Intonato, :IntonatoPythonCallExt).StrumentoSoc
        @test StrumentoSoc <: Intonato.AbstractSoc
        @test hasmethod(Intonato.execute!,
                        Tuple{StrumentoSoc,Any,QickChannelMap,Vector{Int}})
        @test hasmethod(Intonato.dac_rate, Tuple{StrumentoSoc})
        @test hasmethod(Intonato.adc_rate, Tuple{StrumentoSoc})
        # Not constructed here: that needs the Python `strumento` package (the
        # end-to-end item below) and would initialize PythonCall's interpreter.
    end
end

@testitem "StrumentoSoc delegation runs end-to-end on the Python mock (skips without Python strumento)" begin
    using Intonato
    if Base.identify_package("PythonCall") === nothing
        @info "skipping: the delegation end-to-end needs PythonCall in this environment"
        @test true
    else
        using PythonCall
        StrumentoSoc = Base.get_extension(Intonato, :IntonatoPythonCallExt).StrumentoSoc
        # Embedded-Python hygiene: a bare interpreter (no shell locale) defaults to ASCII and
        # chokes on the µ/– in device YAMLs — force UTF-8 mode before the interpreter starts.
        ENV["PYTHONUTF8"] = "1"
        st = try
            pyimport("strumento")
        catch e
            @info "skipping: Python `strumento` not importable in this environment ($e)"
            nothing
        end
        if st === nothing
            @test true   # vacuous pass: the pure-Julia CI lane carries no Python strumento
        else
            fixtures = joinpath(pkgdir(Intonato), "test", "fixtures")
            board = pyimport("strumento.core.mock").MockQickSocV2.from_snapshot(
                joinpath(fixtures, "_fixtures", "soccfg_v2_testbench.json"))
            dev = st.Device.load(joinpath(fixtures, "loopback_demo", "device.yaml"))

            # a small smooth pulse on one drive, sized in the testbench gen's REAL clocking:
            # f_fabric 599.04 MHz with samps_per_clk 16 → envelopes need ≥48 samples to clear
            # qick's 3-fabric-cycle minimum. T = 0.3 µs at 599.04 samples/µs → 180 samples.
            times = collect(range(0.0, 0.3, length = 180))
            pulse = LinearSplinePulse(0.05 .* sin.(range(0, π, length = 180))', times)
            cmap = QickChannelMap([QickGenChannel(0, 5e3; i_drive = 1)]; n_drives = 1)
            soc = StrumentoSoc(dev; drive_map = [(1, "qubit", "drive", 4000.0)],
                               dac_rate = 599.04, adc_rate = 599.04, board = board)
            @test dac_rate(soc) == 599.04
            @test adc_rate(soc) == 599.04

            blobs = execute!(soc, pulse, cmap, [180])   # final knot only
            @test length(blobs) == 1
            @test length(blobs[1]) == 1
            @test isfinite(real(blobs[1][1])) && isfinite(imag(blobs[1][1]))
            # MockQickSocV2 yields zero-valued IQ: the assertion that matters is that the
            # pipeline produced a SHAPED single-read result — i.e. the Measure op made the
            # compiler declare a readout channel, and acquire returned through it.
            @test blobs[1][1] == 0.0 + 0.0im

            # per-knot truncation: two knots on the same pulse produce two acquisitions
            blobs2 = execute!(soc, pulse, cmap, [90, 180])
            @test length(blobs2) == 2
            # knot 1 (t=0) is refused loudly on the delegated path
            @test_throws ErrorException execute!(soc, pulse, cmap, [1])
        end
    end
end

# ──── StrumentoDCSource — the DC verbs over Python strumento's DC seam ────────
# strumento.core.dc's DCSource (named-axis DC control over a VoltageSource driver,
# with max_v clamping — the spin-qubit electrostatics seam). The binding carries
# the instrument-side facts as a Julia DCAxisMap (base's DC data surface) and
# delegates every verb 1:1 to the Python surface — Python owns clamping and axis
# addressing, Julia owns the contract.

"""
    StrumentoDCSource(axes::DCAxisMap; source = nothing) -> StrumentoDCSource

The Julia DC control-class face of Python strumento's DC seam
`strumento.core.dc.DCSource` (available when PythonCall is loaded, as
`Base.get_extension(Intonato, :IntonatoPythonCallExt).StrumentoDCSource`).
`axes` is base's instrument-side `DCAxisMap` (gate name → voltage-source
channel + the hard safety clamp); it builds the Python `DCAxis` set, and
`source` is a Python `VoltageSource` driver — the seam's own
`MockVoltageSource` by default (offline work; wrap a real instrument per the
seam's Protocol).

The verbs delegate 1:1: `set_gate!` → `DCSource.set` (the |v| ≤ max_v clamp
fires through the seam — Python owns instrument safety), `get_gate` →
`DCSource.get`, `gate_snapshot` → `DCSource.snapshot`. An unknown gate's
KeyError and a clamp's ValueError surface as actionable Julia errors with
the seam's own message riding through. `strumento` is imported lazily, so
loading this extension never initializes the interpreter.
"""
mutable struct StrumentoDCSource
    dc::Py             # the Python strumento.core.dc.DCSource (named-axis wrapper)
    axes::DCAxisMap    # the Julia-side instrument facts (the binding's input)
end

function StrumentoDCSource(axes::DCAxisMap; source = nothing)
    st = try
        pyimport("strumento")
    catch e
        error("StrumentoDCSource requires the Python `strumento` package (its DC " *
              "seam, strumento.core.dc). `pyimport(\"strumento\")` failed: $e")
    end
    dc = pyimport("strumento.core.dc")
    src = source === nothing ? dc.MockVoltageSource() : source
    pyaxes = pydict([name => dc.DCAxis(channel = ax.channel, max_v = ax.max_v)
                     for (name, ax) in axes.axes])
    return StrumentoDCSource(dc.DCSource(src, pyaxes), axes)
end

# Re-raise a Python-seam failure as an actionable Julia error with the seam's
# own message riding through (the Python surface owns the wording: the
# unknown-axis list, the clamp's bound).
function _pyseam_error(what::AbstractString, e)
    return error("StrumentoDCSource: $what failed through the Python DC seam: " *
                 sprint(showerror, e))
end

function set_gate!(bridge::StrumentoDCSource, gate::AbstractString, volts::Real)
    try
        bridge.dc.set(gate, Float64(volts))
    catch e
        _pyseam_error("set_gate!($(repr(gate)), $volts V)", e)
    end
    return bridge
end

function get_gate(bridge::StrumentoDCSource, gate::AbstractString)
    try
        return pyconvert(Float64, bridge.dc.get(gate))
    catch e
        _pyseam_error("get_gate($(repr(gate)))", e)
    end
end

function gate_snapshot(bridge::StrumentoDCSource)
    return pyconvert(Dict{String,Float64}, bridge.dc.snapshot())
end

@testitem "StrumentoDCSource delegates the DC verbs to Python strumento's DC seam (python-optional)" begin
    using Intonato
    if Base.identify_package("PythonCall") === nothing
        @info "skipping: no PythonCall in this environment (PythonCall-extension surface)"
        @test true
    else
        using PythonCall
        ENV["PYTHONUTF8"] = "1"
        st = try
            pyimport("strumento")
        catch e
            @info "skipping: Python `strumento` not importable in this environment ($e)"
            nothing
        end
        if st === nothing
            @test true   # vacuous pass: the pure-Julia CI lane carries no Python strumento
        else
            StrumentoDCSource =
                Base.get_extension(Intonato, :IntonatoPythonCallExt).StrumentoDCSource

            # built from the instrument-side data surface; defaults to the
            # seam's own MockVoltageSource
            axes = DCAxisMap("L" => DCAxis(0, 1.0), "R" => DCAxis(1, 0.5))
            bridge = StrumentoDCSource(axes)

            # set/get round-trip through the Python DCSource
            @test set_gate!(bridge, "L", 0.25) === bridge
            @test get_gate(bridge, "L") == 0.25
            set_gate!(bridge, "R", -0.1)
            @test get_gate(bridge, "R") == -0.1
            @test gate_snapshot(bridge) == Dict("L" => 0.25, "R" => -0.1)

            # an unwritten axis reads the source's 0.0 default (the seam's own)
            fresh = StrumentoDCSource(axes)
            @test get_gate(fresh, "L") == 0.0

            # the unknown axis is named actionably, the Python seam's message riding
            err = try
                set_gate!(bridge, "barrier", 0.1); nothing
            catch e
                e
            end
            @test err isa ErrorException
            msg = sprint(showerror, err)
            @test occursin("barrier", msg)
            @test occursin("unknown DC axis", msg)

            # the safety clamp fires THROUGH the seam (dc.py owns |v| ≤ max_v)
            err = try
                set_gate!(bridge, "L", 1.5); nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("max_v", sprint(showerror, err))

            # the seam's own state records the delegated writes (the
            # MockVoltageSource history — the binding adds no bookkeeping of its own)
            hist = pyconvert(Vector{Tuple{Int,Float64}}, bridge.dc.source.history)
            @test (0, 0.25) in hist && (1, -0.1) in hist
        end
    end
end

@testitem "PythonCall extension placement: StrumentoSoc / StrumentoDCSource live in the extension; base gains nothing" begin
    using Intonato
    using Pkg
    # UNguarded (the placement pin must hold in EVERY configuration): the base
    # package defines neither name, and PythonCall is a weak dependency only.
    @test !isdefined(Intonato, :StrumentoSoc)
    @test !isdefined(Intonato, :StrumentoDCSource)
    project = Pkg.TOML.parsefile(joinpath(pkgdir(Intonato), "Project.toml"))
    @test !haskey(project["deps"], "PythonCall")
    @test haskey(project["weakdeps"], "PythonCall")
    @test project["extensions"]["IntonatoPythonCallExt"] == "PythonCall"
    if Base.identify_package("PythonCall") === nothing
        @info "skipping the extension side: no PythonCall in this environment"
        @test true
    else
        using PythonCall
        ext = Base.get_extension(Intonato, :IntonatoPythonCallExt)
        @test ext !== nothing
        @test isdefined(ext, :StrumentoSoc)
        @test isdefined(ext, :StrumentoDCSource)
    end
end

end # module IntonatoPythonCallExt
