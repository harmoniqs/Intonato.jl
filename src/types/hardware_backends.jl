"""
    AbstractHardwareBackend

Abstract supertype for structured hardware interfaces. A backend is the
low-level device handle; the four interface generics below define the contract a
concrete backend implements, and a backend-specific experiment factory chains
them into a measurement.

# Interface (extend these generics for your backend `B <: AbstractHardwareBackend`)
- [`upload_pulse!`](@ref)`(backend::B, pulse)` — translate + load the pulse onto the device
- [`trigger!`](@ref)`(backend::B)` — play the loaded pulse / start acquisition
- [`readout`](@ref)`(backend::B)` — return the raw acquired data
- [`sample_rate`](@ref)`(backend::B) -> Float64` — the device sample rate

Turning raw readout into `Vector{Measurement}` is backend-specific (it needs the
device's discrimination / knot-index layout), so it is **not** part of this
generic contract — a backend ships its own experiment factory that builds a
[`HardwareExperiment`](@ref) whose `run` closure chains
upload → trigger → readout → discriminate (see this package's Strumento seam —
`StrumentoBackend` / `StrumentoExperiment` in `src/hardware/`, over Strumento ≥ 0.2's
`AbstractSoc` — for the reference pattern).

These are declared as generic functions (not merely documented prose) so a
backend `import`s and extends them; nothing in Intonato calls them directly.
"""
abstract type AbstractHardwareBackend end

"""
    upload_pulse!(backend::AbstractHardwareBackend, pulse)

Translate `pulse` and load it onto `backend`'s device. Mutating; returns nothing
by convention. Part of the [`AbstractHardwareBackend`](@ref) interface — extend
it for your backend type.
"""
function upload_pulse! end

"""
    trigger!(backend::AbstractHardwareBackend)

Play the currently-loaded pulse / start acquisition on `backend`'s device.
Mutating. Part of the [`AbstractHardwareBackend`](@ref) interface.
"""
function trigger! end

"""
    readout(backend::AbstractHardwareBackend) -> raw

Return the raw acquired data from `backend`'s device (e.g. IQ blobs). Converting
raw data into `Vector{Measurement}` is backend-specific and lives in the
backend's experiment factory, not here. Part of the
[`AbstractHardwareBackend`](@ref) interface.
"""
function readout end

"""
    sample_rate(backend::AbstractHardwareBackend) -> Float64

The device sample rate. Part of the [`AbstractHardwareBackend`](@ref) interface.
"""
function sample_rate end
