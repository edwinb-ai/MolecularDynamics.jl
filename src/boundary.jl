"""
    wrap_to_box(x, image, unitcell, unitcell_inv)

Wrap position x (vector) into the periodic box defined by unitcell.
Also updates the image vector.
"""
function wrap_to_box(x, image, unitcell, unitcell_inv)
    # Map Cartesian to fractional coordinates
    frac = unitcell_inv * x
    n_cross = floor.(frac)
    frac_mod = frac .- n_cross
    # Update image
    @. image += Int(n_cross)
    # Map back to Cartesian
    wrapped_x = unitcell * frac_mod
    return wrapped_x
end

"""
    minimum_image(rvec, unitcell, unitcell_inv)

Given a displacement vector `rvec = xj - xi`, returns the minimum-image displacement
according to the periodic box described by `unitcell` and `unitcell_inv`.
All vectors/matrices are StaticArrays.
"""
function minimum_image(rvec, unitcell, unitcell_inv)
    # Convert to fractional coordinates
    frac = unitcell_inv * rvec
    # Shift by nearest integer to put in [-0.5, 0.5)
    frac = frac .- round.(frac)
    # Back to cartesian
    return unitcell * frac
end
