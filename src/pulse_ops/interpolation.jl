"""
    interpolate_pulse(p1, p2, α)

Linear interpolation between two pulses: (1 - α) p₁ + α p₂ on knot values.
Precondition: p1 and p2 must have the same knot times and same type.
"""
function interpolate_pulse end

function interpolate_pulse(p1::ZeroOrderPulse, p2::ZeroOrderPulse, α::Real)
    u = (1 - α) * p1.controls.u + α * p2.controls.u
    return ZeroOrderPulse(u, get_knot_times(p1); drive_name = p1.drive_name)
end

function interpolate_pulse(p1::LinearSplinePulse, p2::LinearSplinePulse, α::Real)
    u = (1 - α) * get_knot_values(p1) + α * get_knot_values(p2)
    return LinearSplinePulse(u, get_knot_times(p1); drive_name = p1.drive_name)
end

function interpolate_pulse(p1::CubicSplinePulse, p2::CubicSplinePulse, α::Real)
    u = (1 - α) * get_knot_values(p1) + α * get_knot_values(p2)
    du = (1 - α) * get_knot_derivatives(p1) + α * get_knot_derivatives(p2)
    return CubicSplinePulse(u, du, get_knot_times(p1); drive_name = p1.drive_name)
end

# ============================================================================ #
# Tests
# ============================================================================ #
