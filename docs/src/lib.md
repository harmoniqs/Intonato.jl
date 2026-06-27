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
