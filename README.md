# Intonato.jl

Closed-loop pulse tuning of quantum hardware, built on the
[Piccolo](https://github.com/harmoniqs/Piccolo.jl) ecosystem.

`Intonato` provides a composable framework for refining control pulses against
an experiment in the loop. The core is a small **chassis** — `PulseTuningProblem`
— with two pluggable slots:

- an **`AbstractTuningStrategy`** — the inner optimization step (the policy that
  proposes the next pulse from the latest measurements), and
- an **`AbstractDeviceModel`** — the predictive model of the device the loop
  tunes against (a fixed `NominalModel`, or an adaptive model).

Around those it ships the substrate they need: measurement functions,
measurement models, an `AbstractExperiment` interface (simulated or hardware),
an `AbstractHardwareBackend` contract, pulse operations, a line search, and an
`ExperimentRecord` / `AbstractExperimentLogger` seam for capturing
provenance-rich hardware data.

## Status

Early-stage. The chassis, slot interfaces, device model, and data-collection
seam are in place; concrete tuning strategies are under active development.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/harmoniqs/Intonato.jl")
```

## License

MIT — see [LICENSE](LICENSE).
