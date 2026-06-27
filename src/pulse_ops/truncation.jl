"""
    truncate_pulse(pulse, t_end)

Truncate a pulse to the interval [0, t_end].
For mid-pulse hardware measurements. Dispatches on pulse type.
"""
function truncate_pulse end

function truncate_pulse(pulse::ZeroOrderPulse, t_end::Real)
    times = get_knot_times(pulse)
    mask = times .≤ t_end
    return ZeroOrderPulse(
        get_knot_values(pulse)[:, mask],
        times[mask];
        drive_name = pulse.drive_name,
    )
end

function truncate_pulse(pulse::LinearSplinePulse, t_end::Real)
    times = get_knot_times(pulse)
    mask = times .≤ t_end
    u = get_knot_values(pulse)[:, mask]
    t = times[mask]

    # Add interpolated endpoint if t_end falls between knots
    if t[end] < t_end
        u = hcat(u, pulse(t_end))
        t = vcat(t, Float64(t_end))
    end

    return LinearSplinePulse(u, t; drive_name = pulse.drive_name)
end

function truncate_pulse(pulse::CubicSplinePulse, t_end::Real)
    times = get_knot_times(pulse)
    mask = times .≤ t_end
    u = get_knot_values(pulse)[:, mask]
    du = get_knot_derivatives(pulse)[:, mask]
    t = times[mask]

    # Add interpolated endpoint with derivative if t_end falls between knots
    if t[end] < t_end
        u = hcat(u, pulse(Float64(t_end)))
        du_end = ForwardDiff.derivative(t -> pulse(t), Float64(t_end))
        du = hcat(du, du_end)
        t = vcat(t, Float64(t_end))
    end

    return CubicSplinePulse(u, du, t; drive_name = pulse.drive_name)
end

# ============================================================================ #
# Tests
# ============================================================================ #
