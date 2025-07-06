"""
    wrap_to_box!(x, image, unitcell, unitcell_inv)

Wrap position x (vector) into the periodic box defined by unitcell.
Also updates the image vector.
"""
function wrap_to_box!(x, image, unitcell, unitcell_inv)
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