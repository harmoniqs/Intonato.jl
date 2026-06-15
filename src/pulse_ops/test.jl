# This file contains tests for the surrounding source directory.

# ============================================================================ #
# Tests for interpolation.jl
# ============================================================================ #

@testitem "interpolate_pulse LinearSplinePulse" begin
    using Intonato

    N = 11; T = 5.0
    times = range(0.0, T, length=N) |> collect
    u1 = randn(1, N)
    u2 = randn(1, N)
    p1 = LinearSplinePulse(u1, times; drive_name=:Ω)
    p2 = LinearSplinePulse(u2, times; drive_name=:Ω)

    # α = 0 recovers p1
    p0 = interpolate_pulse(p1, p2, 0.0)
    @test get_knot_values(p0) ≈ u1
    @test get_knot_times(p0) ≈ times

    # α = 1 recovers p2
    p1_end = interpolate_pulse(p1, p2, 1.0)
    @test get_knot_values(p1_end) ≈ u2

    # α = 0.5 gives midpoint
    pmid = interpolate_pulse(p1, p2, 0.5)
    @test get_knot_values(pmid) ≈ 0.5 * (u1 + u2)

    # drive_name preserved
    @test drive_name(pmid) == :Ω
end

@testitem "interpolate_pulse ZeroOrderPulse" begin
    using Intonato

    N = 11; T = 5.0
    times = range(0.0, T, length=N) |> collect
    u1 = randn(1, N)
    u2 = randn(1, N)
    p1 = ZeroOrderPulse(u1, times; drive_name=:Ω)
    p2 = ZeroOrderPulse(u2, times; drive_name=:Ω)

    # α = 0 recovers p1
    p0 = interpolate_pulse(p1, p2, 0.0)
    @test p0.controls.u ≈ u1
    @test get_knot_times(p0) ≈ times

    # α = 1 recovers p2
    p1_end = interpolate_pulse(p1, p2, 1.0)
    @test p1_end.controls.u ≈ u2

    # α = 0.5 gives midpoint
    pmid = interpolate_pulse(p1, p2, 0.5)
    @test pmid.controls.u ≈ 0.5 * (u1 + u2)

    # drive_name preserved
    @test drive_name(pmid) == :Ω
end

@testitem "interpolate_pulse CubicSplinePulse" begin
    using Intonato

    N = 11; T = 5.0
    times = range(0.0, T, length=N) |> collect
    u1 = randn(1, N)
    u2 = randn(1, N)
    du1 = randn(1, N)
    du2 = randn(1, N)
    p1 = CubicSplinePulse(u1, du1, times; drive_name=:Ω)
    p2 = CubicSplinePulse(u2, du2, times; drive_name=:Ω)

    # α = 0 recovers p1
    p0 = interpolate_pulse(p1, p2, 0.0)
    @test get_knot_values(p0) ≈ u1
    @test get_knot_derivatives(p0) ≈ du1
    @test get_knot_times(p0) ≈ times

    # α = 1 recovers p2
    p1_end = interpolate_pulse(p1, p2, 1.0)
    @test get_knot_values(p1_end) ≈ u2
    @test get_knot_derivatives(p1_end) ≈ du2

    # α = 0.5 gives midpoint for both values and derivatives
    pmid = interpolate_pulse(p1, p2, 0.5)
    @test get_knot_values(pmid) ≈ 0.5 * (u1 + u2)
    @test get_knot_derivatives(pmid) ≈ 0.5 * (du1 + du2)

    # drive_name preserved
    @test drive_name(pmid) == :Ω
end

# ============================================================================ #
# Tests for truncation.jl
# ============================================================================ #

@testitem "truncate_pulse LinearSplinePulse at knot" begin
    using Intonato

    N = 11; T = 5.0
    times = range(0.0, T, length=N) |> collect
    u = randn(1, N)
    pulse = LinearSplinePulse(u, times; drive_name=:Ω)

    # Truncate at the 6th knot (index 6, time = times[6])
    t_end = times[6]
    p_trunc = truncate_pulse(pulse, t_end)

    t_out = get_knot_times(p_trunc)
    u_out = get_knot_values(p_trunc)

    @test length(t_out) == 6
    @test t_out ≈ times[1:6]
    @test u_out ≈ u[:, 1:6]
    @test drive_name(p_trunc) == :Ω
end

@testitem "truncate_pulse LinearSplinePulse between knots" begin
    using Intonato

    N = 11; T = 5.0
    times = range(0.0, T, length=N) |> collect
    u = randn(1, N)
    pulse = LinearSplinePulse(u, times; drive_name=:Ω)

    # Truncate between knots 6 and 7
    t_end = (times[6] + times[7]) / 2.0
    p_trunc = truncate_pulse(pulse, t_end)

    t_out = get_knot_times(p_trunc)
    u_out = get_knot_values(p_trunc)

    # Should have the first 6 knots plus one interpolated endpoint
    @test length(t_out) == 7
    @test t_out[1:6] ≈ times[1:6]
    @test t_out[end] ≈ t_end

    # Interpolated endpoint matches original pulse evaluation
    @test u_out[:, end] ≈ pulse(t_end)
end

@testitem "truncate_pulse CubicSplinePulse between knots" begin
    using Intonato
    using ForwardDiff

    N = 11; T = 5.0
    times = range(0.0, T, length=N) |> collect
    u = randn(1, N)
    du = randn(1, N)
    pulse = CubicSplinePulse(u, du, times; drive_name=:Ω)

    # Truncate between knots 6 and 7
    t_end = (times[6] + times[7]) / 2.0
    p_trunc = truncate_pulse(pulse, t_end)

    t_out = get_knot_times(p_trunc)
    u_out = get_knot_values(p_trunc)
    du_out = get_knot_derivatives(p_trunc)

    # Should have the first 6 knots plus one interpolated endpoint
    @test length(t_out) == 7
    @test t_out[1:6] ≈ times[1:6]
    @test t_out[end] ≈ t_end

    # Interpolated value matches original pulse
    @test u_out[:, end] ≈ pulse(t_end)

    # Endpoint derivative matches ForwardDiff derivative of original pulse
    du_expected = ForwardDiff.derivative(t -> pulse(t), Float64(t_end))
    @test du_out[:, end] ≈ du_expected
end
