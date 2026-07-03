# Local helper: run a name-filtered subset of the package testitems.
# Usage: TEST_FILTER="substring" julia --project=. test/run_filtered.jl
using TestItemRunner
const PAT = get(ENV, "TEST_FILTER", "")
@run_package_tests filter = ti -> occursin(PAT, ti.name)
