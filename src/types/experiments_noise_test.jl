# Tests for the SimulatedExperiment noise-sampling contract (seeded shot noise)
# and the run_experiment n_shots override.

@testitem "SimulatedExperiment noise sampling contract" begin
    using Intonato
    using Random: MersenneTwister

    # Single-qubit Rabi fixture (mirrors src/problems/test.jl QILC items).
    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    sys = QuantumSystem(1.0 * σz, [σx], [1.0])
    N = 11
    times = collect(range(0.0, 5.0; length = N))
    ψ0 = ComplexF64[1.0, 0.0]
    ψg = ComplexF64[0.0, 1.0]
    pulse = LinearSplinePulse(0.1 * ones(1, N), times)
    qtraj = KetTrajectory(sys, pulse, ψ0, ψg)

    m_shot = ShotNoiseMeasurement(populations, 400, population_covariance)
    m_det = DeterministicMeasurement(populations)
    model = MeasurementModel(:ψ̃, AbstractMeasurement[m_shot, m_det], [N, N])

    # (a) rng = nothing (default): deterministic — identical across runs, and
    # the legacy positional constructor still works (regression).
    exp_det = SimulatedExperiment(qtraj, model)
    @test exp_det.rng === nothing
    y1 = run_experiment(exp_det, pulse)
    y2 = run_experiment(exp_det, pulse)
    @test all(a.data == b.data for (a, b) in zip(y1, y2))

    # (b) seeded rng: shot-noise elements get ε ~ N(0, Σ(y)) added,
    # deterministic elements are untouched, and the same seed reproduces.
    exp_a = SimulatedExperiment(qtraj, model; rng = MersenneTwister(1))
    exp_b = SimulatedExperiment(qtraj, model; rng = MersenneTwister(1))
    ya = run_experiment(exp_a, pulse)
    yb = run_experiment(exp_b, pulse)
    @test ya[1].data != y1[1].data          # noise added on the shot element
    @test ya[2].data == y1[2].data          # deterministic element untouched
    @test ya[1].data == yb[1].data          # same seed ⇒ same draw
end

@testitem "run_experiment n_shots override rebuilds shot measurements" begin
    using Intonato
    using Random: MersenneTwister

    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    sys = QuantumSystem(1.0 * σz, [σx], [1.0])
    N = 11
    times = collect(range(0.0, 5.0; length = N))
    ψ0 = ComplexF64[1.0, 0.0]
    ψg = ComplexF64[0.0, 1.0]
    pulse = LinearSplinePulse(0.1 * ones(1, N), times)
    qtraj = KetTrajectory(sys, pulse, ψ0, ψg)

    m_shot = ShotNoiseMeasurement(populations, 400, population_covariance)
    m_det = DeterministicMeasurement(populations)
    model = MeasurementModel(:ψ̃, AbstractMeasurement[m_shot, m_det], [N, N])

    y_det = run_experiment(SimulatedExperiment(qtraj, model), pulse)

    # Same seed ⇒ same randn sequence, and σ ∝ 1/√n, so the boosted-shots noise
    # realization is exactly the n=400 realization scaled by 1/√10.
    y400 = run_experiment(SimulatedExperiment(qtraj, model; rng = MersenneTwister(7)), pulse)
    y4000 = run_experiment(
        SimulatedExperiment(qtraj, model; rng = MersenneTwister(7)),
        pulse;
        n_shots = 4000,
    )
    ε400 = y400[1].data .- y_det[1].data
    ε4000 = y4000[1].data .- y_det[1].data
    @test isapprox(ε4000, ε400 ./ sqrt(10); atol = 1e-12)

    # Deterministic element unaffected by the override.
    @test y4000[2].data == y_det[2].data

    # The override is a value-type copy — the original model is untouched.
    @test model.measurements[1].n_shots == 400
end
