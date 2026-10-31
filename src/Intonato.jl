module Intonato

using Reexport
@reexport using Piccolo
@reexport using NamedTrajectories

# NamedTrajectories ≥ 0.9.3 collides with Piccolo's top-level `duration` reexport
# (TimeWarp's `duration` in the same reexport chain), which leaves `Piccolo.duration`
# — and, through the reexport above, `Intonato.duration` — UNBOUND in fresh
# resolutions while still listed among the exports. Pin the binding to its
# defining module until Piccolo ships its fix (harmoniqs/Piccolo.jl#323). The
# pulse-sampling seam (src/hardware/soc.jl, src/hardware/translate.jl) reads the
# pulse duration through this pin; the ownership-guard testitem pins it here.
const duration = Piccolo.Quantum.Pulses.duration

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
# Binomial Fisher / IRLS weighting (exact-likelihood statistics on the
# measurement surface — the A1 thin split's public increment).
include("noise/fisher_irls.jl")
include("noise/fisher_irls_test.jl")

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

# ──── Hardware substrate (the soc contract; moved from Strumento.jl, #55) ─────
# The soc contract + its verbs, the QICK channel map, the slow-DC control class,
# the pulse → envelope translation, readout conversion, and the board-free
# MockSoc. Defined here — this package owns the substrate.
include("hardware/soc.jl")
include("hardware/channel_map.jl")
include("hardware/dc.jl")
include("hardware/translate.jl")
include("hardware/readout.jl")
include("hardware/mock_soc.jl")

# ──── Hardware seam (backend adapter + experiment factory) ────────────────────
# The backend adapter and the experiment factory that bridge pulse → soc →
# measurement. `StrumentoBackend` / `StrumentoExperiment` keep their names: they
# name the Python strumento stack the real-board soc talks to.
include("hardware/strumento_backend.jl")
include("hardware/strumento_experiment.jl")
include("hardware/integration_test.jl")
include("hardware/ownership_test.jl")

# ──── Digital twins (Intonato.Twins; moved from Strumento.jl) ─────────────────
# Drift processes, the twin-record loader, the truth/belief/record twin, the
# twin's soc face (TwinSoc), the twin wiring map, and the physics families.
# Included after the hardware substrate: TwinSoc implements the soc contract.
# A submodule, NOT exported at top level: it keeps generic verbs (`instantiate`,
# `advance!`, `calibrate!`) out of this namespace — `using Intonato.Twins`.
include("twins/Twins.jl")

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
# The soc substrate: the contract + its verbs, the channel map, the slow-DC
# control class, the translation record + verb, readout conversion, the mock soc.
export AbstractSoc
export execute!, load_envelope!, play_program!, acquire, dac_rate, adc_rate
export set_gate!, get_gate, gate_snapshot
export QickChannelMap, QickGenChannel
export DCAxis, DCAxisMap, gate_names, axis_for
export pulse_to_envelopes, QickProgram
export iq_to_measurements
export MockSoc
# The hardware seam: the backend adapter over an AbstractSoc and its experiment
# factory.
export StrumentoBackend, StrumentoExperiment
export ExperimentRecord
export AbstractExperimentLogger, NullExperimentLogger, InMemoryExperimentLogger, record!

# Noise statistics + whitening (GLS)
export noise_floor, debiased_cost, cost_std, diff_std, whiten
# Binomial Fisher / IRLS measurement weighting (exact-likelihood statistics)
export binomial_fisher_information, binomial_fisher_weights

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
