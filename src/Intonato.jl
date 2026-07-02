module Intonato

using Reexport
@reexport using Piccolo
@reexport using NamedTrajectories

using LinearAlgebra
using ForwardDiff   # used by measurement_functions/wigner.jl + pulse_ops/truncation.jl
using TestItems

# ──── Types ──────────────────────────────────────────────────────────────────
include("types/noise_models.jl")
include("types/measurements.jl")
include("types/measurement_models.jl")
# experiment_record + experiment_logger define the types referenced by
# run_experiment's additive `logger` kwarg, so they must precede experiments.jl.
include("types/experiment_record.jl")
include("types/experiment_logger.jl")
include("types/experiments.jl")
include("types/hardware_backends.jl")
include("types/test.jl")

# ──── Noise statistics + whitening (GLS) ─────────────────────────────────────
include("noise/noise_stats.jl")
include("noise/noise_stats_test.jl")
include("noise/whitening.jl")
include("noise/whitening_test.jl")

# ──── Measurement functions ──────────────────────────────────────────────────
include("measurement_functions/state_measurements.jl")
include("measurement_functions/wigner.jl")
include("measurement_functions/displaced_parity.jl")
include("measurement_functions/partial_trace.jl")
include("measurement_functions/test.jl")

# ──── Pulse operations ───────────────────────────────────────────────────────
include("pulse_ops/truncation.jl")
include("pulse_ops/interpolation.jl")
include("pulse_ops/test.jl")

# ──── Optimizers (line search) ────────────────────────────────────────────────
include("optimizers/line_search.jl")
include("optimizers/test.jl")

# ──── Device models (the AbstractDeviceModel seam + NominalModel) ─────────────
include("device_models/abstract.jl")

# ──── Problems (the tuning chassis + strategy interface) ──────────────────────
include("problems/abstract.jl")
include("problems/pulse_tuning_problem.jl")
include("problems/test.jl")

# ──── Exports ────────────────────────────────────────────────────────────────

# Types
export Measurement, MeasurementModel
export AbstractExperiment, SimulatedExperiment, HardwareExperiment
export AbstractMeasurement,
    DeterministicMeasurement, ShotNoiseMeasurement, KnownCovarianceMeasurement
export pauli_covariance, population_covariance, wigner_covariance
export pauli, pop
export AbstractHardwareBackend
export ExperimentRecord
export AbstractExperimentLogger, NullExperimentLogger, InMemoryExperimentLogger, record!

# Noise statistics + whitening (GLS)
export noise_floor, debiased_cost, cost_std, diff_std, whiten

# Core interface
export run_experiment, model_predict, measurement_error
export phase_max_fidelity

# Measurement functions
export populations, populations_density, full_state, density_matrix_measurement
export observable_expectation, observable_expectations, expect
export wigner, wigner_at
export displaced_parity, displaced_parity_at
export partial_trace_B

# Pulse operations
export truncate_pulse, interpolate_pulse

# Optimizers (public: line search)
export armijo_line_search

# Device-model interface (the AbstractDeviceModel slot + nominal stand-in)
export AbstractDeviceModel, NominalModel, predict, adapt!

# Tuning-problem + strategy interfaces (chassis/strategy split). The public
# surface ships the chassis + the generic strategy interface + the
# IdentityStrategy stand-in. Concrete tuning strategies plug in via this
# interface.
export AbstractPulseTuningProblem, AbstractTuningStrategy, IdentityStrategy, step
export prepare_strategy,
    tuning_goal, candidate_trajectory, last_timings, accepts_global_data

# Closed-loop tuning chassis. The chassis is strategy-generic; its
# result/record types are public.
export PulseTuningProblem, TuningResult, IterationRecord

end
