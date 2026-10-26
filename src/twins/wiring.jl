# TwinWiringMap — device-channel → twin-drive wiring.
#
# The channel-map concept extended one rung DOWN the stack: `QickChannelMap`
# routes Piccolo drive CONTROLS onto QICK generator channels (the pulse side);
# `TwinWiringMap` routes DEVICE generator channels onto the TWIN's drive
# quadratures (which family drive each device line feeds), so a played channel
# that would silently land on the wrong twin drive is named, not executed.
#
# Moved from Strumento.jl (its src/channel_map.jl, Intonato #55) — behaviour
# unchanged.

export TwinWiringMap, TwinGenWiring, wiring_for

"""
    TwinGenWiring(gen_ch, i_drive, q_drive; line="")

One device generator channel's wiring into the twin: gen channel `gen_ch`
(the 0-based generator channel) plays the twin's `i_drive` (I quadrature) and
`q_drive` (Q quadrature) controls. `line` documents which device wiring line the
channel serves (it rides the error messages).
"""
struct TwinGenWiring
    gen_ch::Int
    i_drive::Int
    q_drive::Int
    line::String
end

TwinGenWiring(gen_ch::Int, i_drive::Integer, q_drive::Integer; line::AbstractString = "") =
    TwinGenWiring(gen_ch, Int(i_drive), Int(q_drive), String(line))

"""
    TwinWiringMap(wirings; n_drives) -> TwinWiringMap

Validated device-channel → twin-drive wiring: every drive index referenced must
be in `1:n_drives`, no drive may be wired twice, gen channels must be distinct,
and the wirings must be ordered by gen channel (played generators arrive
sorted; the map's order is the wiring's declaration).
"""
struct TwinWiringMap
    wirings::Vector{TwinGenWiring}
    n_drives::Int
end

function TwinWiringMap(wirings::Vector{TwinGenWiring}; n_drives::Integer)
    used = Int[]
    for w in wirings
        for d in (w.i_drive, w.q_drive)
            (1 ≤ d ≤ n_drives) || error(
                "TwinWiringMap: twin drive index $d (gen channel $(w.gen_ch)) " *
                "out of range 1:$n_drives",
            )
            d in used && error("TwinWiringMap: twin drive index $d wired more than once")
            push!(used, d)
        end
    end
    gen_chs = [w.gen_ch for w in wirings]
    allunique(gen_chs) || error("TwinWiringMap: generator channels must be distinct")
    issorted(gen_chs) || error(
        "TwinWiringMap: wirings must be ordered by gen channel (got $gen_chs) — " *
        "played generators arrive sorted, and the map's order is the wiring's " *
        "declaration",
    )
    return TwinWiringMap(wirings, Int(n_drives))
end

"""The twin drive pair a wired gen channel feeds, or `nothing` when the channel
is unwired (an unmapped device line — the actionable-error case)."""
function wiring_for(map::TwinWiringMap, gen_ch::Integer)
    for w in map.wirings
        w.gen_ch == gen_ch && return w
    end
    return nothing
end

@testitem "TwinWiringMap — the device-channel → twin-drive wiring, validated" begin
    using Intonato.Twins
    # The canonical wiring: the transmon line's generator feeds the twin's
    # ancilla quadratures (drives 1/2), the cavity line's the cavity quadratures
    # (drives 3/4) — the bosonic family's documented drive order.
    m = TwinWiringMap(
        [
            TwinGenWiring(2, 1, 2; line = "qubit.drive"),
            TwinGenWiring(3, 3, 4; line = "manipulate.main"),
        ];
        n_drives = 4,
    )
    @test m.wirings[1].gen_ch == 2 && m.wirings[2].gen_ch == 3
    @test wiring_for(m, 3) === m.wirings[2]
    @test wiring_for(m, 5) === nothing           # an unmapped device line

    # validation, each named: drive out of range, drive wired twice, duplicate
    # or unsorted gen channels.
    @test_throws ErrorException TwinWiringMap([TwinGenWiring(2, 1, 5)]; n_drives = 4)
    @test_throws ErrorException TwinWiringMap(
        [TwinGenWiring(2, 1, 3), TwinGenWiring(3, 3, 4)];
        n_drives = 4,
    )
    @test_throws ErrorException TwinWiringMap(
        [TwinGenWiring(2, 1, 2), TwinGenWiring(2, 3, 4)];
        n_drives = 4,
    )
    err = try
        TwinWiringMap([TwinGenWiring(3, 3, 4), TwinGenWiring(2, 1, 2)]; n_drives = 4)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("ordered by gen channel", sprint(showerror, err))
end
