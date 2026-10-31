# API

```@meta
CollapsedDocStrings = true
```

The central entry point. `solve!(::PulseTuningProblem)` is defined as a
`Piccolo.solve!` method (the function-name binding is owned by the `Piccolo` /
`DirectTrajOpt` module, in scope here via Intonato's `@reexport using Piccolo`).
It is documented explicitly so it is guaranteed to appear even if a future
refactor changes which module owns the binding; the `@autodocs` block below is
filtered to skip this method so it is not documented twice.

```@docs
solve!(::Intonato.PulseTuningProblem)
```

```@autodocs
Modules = [Intonato]
Order = [:type, :function]
Filter = b -> b !== solve!
```

## Real-board extension

The real-board path lives in the `IntonatoPythonCallExt` package extension,
loaded when `PythonCall` is loaded: `StrumentoSoc` (an `AbstractSoc` that
delegates the played pulse to the Python `strumento` framework — Intonato →
Python strumento, no intermediate Julia package) and `StrumentoDCSource` (the
`set_gate!` / `get_gate` / `gate_snapshot` verbs over `strumento.core.dc`).
Extension types are reached through the extension module:

```julia
using Intonato, PythonCall
ext = Base.get_extension(Intonato, :IntonatoPythonCallExt)
ext.StrumentoSoc, ext.StrumentoDCSource
```

Their docstrings are in `ext/IntonatoPythonCallExt.jl`. Python `strumento` must
be importable by PythonCall's interpreter; Intonato declares no Python
dependencies.
