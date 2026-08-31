module Intonato

using Reexport
@reexport using Piccolo
@reexport using NamedTrajectories

using LinearAlgebra
using ForwardDiff   # used by measurement_functions/wigner.jl + pulse_ops/truncation.jl
using Random: AbstractRNG   # SimulatedExperiment noise-sampling rng slot
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
include("types/experiments_noise_test.jl")
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
include("measurement_functions/density_encoding.jl")
include("measurement_functions/parity_reconstruction.jl")
include("measurement_functions/partial_trace.jl")
include("measurement_functions/fidelity.jl")
include("measurement_functions/test.jl")
include("measurement_functions/bosonic_composite_test.jl")

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
# Learnable-parameter declaration (`Learn`) + the `learnables` strategy seam.
# Included after abstract.jl (needs AbstractTuningStrategy) and before
# pulse_tuning_problem.jl (whose constructor calls `_wire_learnables!`).
include("problems/learnable.jl")
include("problems/acceptance.jl")
include("problems/selectors.jl")
include("problems/pulse_tuning_problem.jl")
include("problems/test.jl")
include("problems/acceptance_test.jl")
include("problems/selectors_test.jl")

# ──── Strategies (public concrete strategies for the chassis) ─────────────────
include("strategies/low_rank_hessian_strategy.jl")

# ──── Hardware seam (Strumento ≥ 0.2: backend adapter + experiment factory) ───
# The closed-loop hardware seam, relocated from Strumento.jl's v0.1.x tree when
# its v0.2 standalone release severed the inverted dependency edge. The soc
# substrate lives below this package; the backend adapter and the experiment
# factory that bridge pulse → soc → measurement live here, at the chassis.
include("hardware/strumento_backend.jl")
include("hardware/strumento_experiment.jl")
include("hardware/integration_test.jl")
include("hardware/reexport_test.jl")

# ──── Exports ────────────────────────────────────────────────────────────────

# Types
export Measurement, MeasurementModel
export AbstractExperiment, SimulatedExperiment, HardwareExperiment
export AbstractMeasurement,
    DeterministicMeasurement, ShotNoiseMeasurement, KnownCovarianceMeasurement
export pauli_covariance, population_covariance, wigner_covariance
export pauli, pop
export AbstractHardwareBackend
export upload_pulse!, trigger!, readout, sample_rate
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
export goal_fidelity, goal_fidelity_at, phase_max_fidelity_at
export qubit_sigma_z, qubit_sigma_z_at
export rho_triangle, rho_to_measvec, measvec_to_rho, reduced_cavity_rho
export rho_measurement_functions
export reconstruct_rho_from_parity

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

# Learnable-parameter seam: declare a device parameter learnable once (`Learn`);
# a strategy exposes its set through `learnables`, and the PulseTuningProblem
# constructor validates the QCP against it.
export Learn, learnables

# Closed-loop tuning chassis. The chassis is strategy-generic; its
# result/record types are public.
export PulseTuningProblem, TuningResult, IterationRecord

# Acceptance-policy seam (chassis-owned step acceptance + trust-scale schedule)
export AcceptancePolicy, LineSearchAcceptance, OneShotAcceptance
export decide, reset_acceptance!

# Iterate-selector seam (end-of-run iterate policy)
export IterateSelector, FinalIterate, NoiseCorrectedBestJ, TopKRemeasure, PolyakAverage
export select_iterate!

# Strategy diagnostic hook (recorded as IterationRecord.F_model)
export last_f_model

# Public concrete strategies: measured-gradient Newton in the low-rank
# principal subspace of the model cost Hessian (Liu et al. 2026,
# arXiv:2606.05060).
export LowRankHessianStrategy

end
