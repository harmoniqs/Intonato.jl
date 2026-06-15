module IntonatoOptimExt

# Placeholder extension reserving the black-box tuning-strategy seam.
#
# When `Optim` is loaded alongside `Intonato`, this extension will provide a
# black-box `AbstractTuningStrategy` (e.g. NelderMead) that plugs into the
# public `PulseTuningProblem` chassis via the generic strategy interface
# (prepare_strategy / step / tuning_goal / candidate_trajectory / …).
#
# The actual strategy lands in its own follow-on plan (spec → plan); this stub
# only fixes the package structure so the [weakdeps]/[extensions] wiring exists
# and Intonato precompiles with or without Optim present. It defines nothing
# and extends nothing yet — intentionally a no-op.

using Intonato
using Optim

end
