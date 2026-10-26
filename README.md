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

## Hardware stack

Intonato owns its hardware substrate and talks to the board through the Python
[`strumento`](https://github.com/harmoniqs/strumento) framework directly:

```
Intonato (Julia)  ──PythonCall──▶  strumento (Python)  ──▶  qick / QICK board
```

- **Soc contract** — `AbstractSoc` and its verbs (`execute!`, `dac_rate`,
  `adc_rate`, …), the `QickChannelMap`, the pulse → envelope translation
  (`pulse_to_envelopes` / `QickProgram`), readout conversion
  (`iq_to_measurements` → `Measurement`), the slow-DC control class
  (`set_gate!` / `get_gate` / `gate_snapshot` over a `DCAxisMap`), and the
  board-free `MockSoc`. `StrumentoBackend` / `StrumentoExperiment` turn any soc
  into a `HardwareExperiment` for the tuning chassis.
- **Digital twins** — `using Intonato.Twins`: drift processes, twin records,
  the truth/belief `DigitalTwin`, its soc face `TwinSoc`, and the physics
  families. Drift moves truth only; calibration moves belief only.
- **Real board** — the `IntonatoPythonCallExt` package extension, loaded when
  `PythonCall` is loaded. It provides `StrumentoSoc` (hands the solved pulse to
  Python `strumento`: `from_solution` → `StrumentoProgram` → acquire → averaged
  IQ) and `StrumentoDCSource` (the DC verbs over `strumento.core.dc`). Extension
  types do not surface on `Intonato` itself; reach them through the extension
  module:

  ```julia
  using Intonato, PythonCall
  ext = Base.get_extension(Intonato, :IntonatoPythonCallExt)
  soc = ext.StrumentoSoc("device.yaml"; drive_map = [(1, "qubit", "drive", 4000.0)],
                         dac_rate = 599.04, adc_rate = 599.04,
                         board = qick_soc)   # a qick soc (or strumento's MockQickSocV2)
  ```

  Python `strumento` (and `qick`) must be installed in the interpreter
  PythonCall uses — Intonato declares no Python dependencies. To use an
  existing environment, set `JULIA_CONDAPKG_BACKEND=Null` and
  `JULIA_PYTHONCALL_EXE=/path/to/python`. Without PythonCall, Intonato loads
  with no Python at all.

## Migrating from Strumento.jl

Strumento.jl, the former intermediate Julia package, is retired as of Intonato
0.5.0; its binding and twin core moved here, and its Python-duplicating layers
were dropped in favour of Python `strumento`. Drop the `Strumento` dependency
and map names as follows:

| Strumento.jl | Intonato 0.5 |
|---|---|
| `AbstractSoc`, `execute!`, `load_envelope!`, `play_program!`, `acquire`, `dac_rate`, `adc_rate` | `Intonato` (same names) |
| `QickChannelMap`, `QickGenChannel`, `pulse_to_envelopes`, `QickProgram`, `iq_to_measurements`, `Measurement` | `Intonato` (one `Measurement` type) |
| `set_gate!`, `get_gate`, `gate_snapshot`, `DCAxis`, `DCAxisMap`, `gate_names`, `axis_for` | `Intonato` |
| `MockSoc` (Piccolo extension) | `Intonato.MockSoc` |
| twin core (`DigitalTwin`, drift processes, `TwinRecord` / `load_record`), `TwinSoc`, `TwinWiringMap`, `TwinGenWiring`, `wiring_for`, physics families | `Intonato.Twins` |
| `StrumentoSoc`, `StrumentoDCSource` (PythonCall extension) | `Base.get_extension(Intonato, :IntonatoPythonCallExt)` |
| `BringupBridge`, `TwinJobServer`, wire client, bring-up procedures | removed — use Python `strumento` directly |

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/harmoniqs/Intonato.jl")
```

## License

MIT — see [LICENSE](LICENSE).
