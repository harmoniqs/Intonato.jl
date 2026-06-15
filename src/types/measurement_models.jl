"""
    MeasurementModel

Defines what to measure and where (which knot points).

# Fields
- `state_name::Symbol`: trajectory component to read (:ψ̃, :Ũ⃗, :ρ̃)
- `measurements::Vector{AbstractMeasurement}`: measurement objects (callable + noise info)
- `indices::Vector{Int}`: knot indices π_1, …, π_M ⊂ 1:N
"""
struct MeasurementModel
    state_name::Symbol
    measurements::Vector{AbstractMeasurement}
    indices::Vector{Int}

    function MeasurementModel(
        state_name::Symbol,
        measurements::Vector{<:AbstractMeasurement},
        indices::Vector{Int},
    )
        @assert length(measurements) == length(indices) (
            "MeasurementModel: lengths must match — " *
            "got measurements=$(length(measurements)), " *
            "indices=$(length(indices))"
        )
        return new(state_name, convert(Vector{AbstractMeasurement}, measurements), indices)
    end
end

"""
    MeasurementModel(state_name, functions, indices)

Backward-compatible constructor: wraps bare `Function`s as `DeterministicMeasurement`.
"""
function MeasurementModel(
    state_name::Symbol,
    functions::Vector{<:Function},
    indices::Vector{Int},
)
    measurements = DeterministicMeasurement.(functions)
    return MeasurementModel(state_name, measurements, indices)
end

"""
    model_predict(traj::NamedTrajectory, model::MeasurementModel)

Evaluate measurement functions on the model-side trajectory at each knot index.
"""
function model_predict(traj::NamedTrajectory, model::MeasurementModel)
    return [
        Measurement(
            model.measurements[j](traj[k][model.state_name]),
            k
        )
        for (j, k) in enumerate(model.indices)
    ]
end
