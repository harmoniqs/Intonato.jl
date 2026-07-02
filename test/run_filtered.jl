using TestItemRunner
const PAT = get(ENV, "TEST_FILTER", "")
@run_package_tests filter = ti -> occursin(PAT, ti.name)
