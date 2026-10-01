"""
    wrap_to_box(x, unitcell, unitcell_inv) -> (wrapped_x, n_cross)

Wrap position `x` into the periodic box. Returns the wrapped position and the number of
box vectors crossed along each direction (to be added to the image counter).
"""
@inline function wrap_to_box(x, unitcell, unitcell_inv)
    # Map Cartesian to fractional coordinates
    frac = unitcell_inv * x
    n_cross = floor.(frac)
    # Map back to Cartesian
    wrapped_x = unitcell * (frac - n_cross)
    return wrapped_x, Int32.(n_cross)
end

"""
    wrap_positions!(positions, images, unitcell, unitcell_inv)

Wrap all positions into the box in place, updating the image counters.
"""
function wrap_positions!(positions, images, unitcell, unitcell_inv)
    @inbounds for i in eachindex(positions, images)
        (positions[i], n_cross) = wrap_to_box(positions[i], unitcell, unitcell_inv)
        images[i] += n_cross
    end
    return nothing
end

"""
    wrapped_positions(state) -> (positions, images)

Copies of the positions wrapped into the box and their matching image counters, without
modifying `state` (which would invalidate the neighbor list).
"""
function wrapped_positions(state::SimulationState)
    positions = copy(state.positions)
    images = copy(state.images)
    wrap_positions!(positions, images, state.unitcell, inv(state.unitcell))
    return positions, images
end
