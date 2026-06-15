"""
    Measurement

Singular data container: one measurement function evaluated at one knot point.

# Fields
- `data::Vector{Float64}`: measurement output g_j(x_{π_j}), variable length
- `index::Int`: knot point index π_j ∈ 1:N
"""
struct Measurement
    data::Vector{Float64}
    index::Int
end

Base.length(m::Measurement) = length(m.data)

"""
    measurement_error(y1, y2)

Sum of squared differences between two measurement collections.
"""
function measurement_error(y1::Vector{Measurement}, y2::Vector{Measurement})
    return sum(sum((m1.data .- m2.data).^2) for (m1, m2) in zip(y1, y2))
end
