# ──── Device-model interface ──────────────────────────────────────────────────

"""
    AbstractDeviceModel

Predictive model of device dynamics the tuning loop plans against.

Concrete models implement:

    predict(model, pulse, measurement_model) -> Vector{Measurement}-like
    adapt!(model, pulse, y_exp)              -> model        # refine from hardware data
"""
abstract type AbstractDeviceModel end

function predict end
function adapt! end

"""
    NominalModel{S, K, G} <: AbstractDeviceModel

Fixed nominal physics model (Piccolo `QuantumSystem`). `adapt!` is a no-op
dispatched away at compile time — `NominalModel` carries no learned parameters.

# Fields
- `system::S`: `QuantumSystem` / `OpenQuantumSystem` (the nominal dynamics)
- `ψ_init::K`: initial state
- `ψ_goal::G`: goal state
"""
struct NominalModel{S,K,G} <: AbstractDeviceModel
    system::S            # QuantumSystem / OpenQuantumSystem
    ψ_init::K
    ψ_goal::G
end

"""
    predict(m::NominalModel, pulse::AbstractPulse, model::MeasurementModel)

Roll the `pulse` through the wrapped nominal system, then evaluate the
measurement model at each knot index — mirroring
[`run_experiment`](@ref) on a `SimulatedExperiment`, but with the nominal
(rather than mismatched) system. Returns a `Vector{Measurement}`.

!!! note
    This propagates the pulse. `model_predict` only reads an
    already-populated trajectory and does **not** propagate — do not use it here.
"""
function predict(m::NominalModel, pulse::AbstractPulse, model::MeasurementModel)
    qtraj_template = KetTrajectory(m.system, pulse, m.ψ_init, m.ψ_goal)
    qtraj = rollout(qtraj_template, pulse)
    knot_times = get_knot_times(pulse)

    measurements = Vector{Measurement}(undef, length(model.indices))
    for (j, k) in enumerate(model.indices)
        t_k = knot_times[k]
        x_k = _state_at_time(qtraj, t_k, model.state_name)
        measurements[j] = Measurement(model.measurements[j](x_k), k)
    end

    return measurements
end

adapt!(m::NominalModel, ::AbstractPulse, ::Any) = m   # no-op via dispatch

@testitem "AbstractDeviceModel + NominalModel predict/adapt!" begin
    using Intonato
    using Intonato: predict, adapt!, AbstractDeviceModel
    using LinearAlgebra

    # Single-qubit Rabi fixture (mirrors src/problems/test.jl QILC items).
    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    sys = QuantumSystem(1.0 * σz, [σx], [1.0])

    N = 11
    T = 5.0
    times = range(0.0, T, length = N) |> collect
    ψ0 = ComplexF64[1.0, 0.0]
    ψg = ComplexF64[0.0, 1.0]
    pulse = LinearSplinePulse(0.1 * ones(1, N), times)
    model = MeasurementModel(:ψ̃, [full_state for _ = 1:N], collect(1:N))

    # A NominalModel wraps a QuantumSystem; predict rolls the pulse through it
    # and evaluates the measurement model. adapt! is a no-op.
    nm = NominalModel(sys, ψ0, ψg)
    y = predict(nm, pulse, model)
    @test y isa Vector{<:Real} || y isa Vector
    @test length(y) == length(model.indices)

    # adapt! is a no-op: it returns the same model object and mutates nothing.
    # (`QuantumSystem` has no value `==`, so a deepcopy/`==` check is defeated by
    # struct identity; assert the no-op via identity of the model and its fields.)
    sys_before, ψ0_before, ψg_before = nm.system, nm.ψ_init, nm.ψ_goal
    out = adapt!(nm, pulse, y)
    @test out === nm
    @test nm.system === sys_before
    @test nm.ψ_init === ψ0_before
    @test nm.ψ_goal === ψg_before
end
