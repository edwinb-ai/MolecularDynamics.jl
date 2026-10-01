"""
    QuinticSwitch

Switch `S(x) = 1 - 10x³ + 15x⁴ - 6x⁵`, with `x = (r - r_on) / (r_cut - r_on)`. Its first
and second derivatives vanish at both ends, so a switched potential keeps continuous
energy, force and second derivative.
"""
struct QuinticSwitch end

"""
    XPLORSwitch

The XPLOR (CHARMM) switch, as in [`LennardJonesXPLOR`](@ref). Energy and force stay
continuous, but the second derivative jumps at `r_on` and `r_cut`.
"""
struct XPLORSwitch end

"""
    switch_value(switch, r, r_on, r_cut) -> (S, dS/dr)

Value and derivative of `switch` at `r`, for `r_on <= r < r_cut`.
"""
@inline function switch_value(::QuinticSwitch, r, r_on, r_cut)
    width = r_cut - r_on
    x = (r - r_on) / width
    S = 1 - x^3 * (10 - 15x + 6x^2)
    dS = -30 * x^2 * (1 - x)^2 / width
    return S, dS
end

@inline switch_value(::XPLORSwitch, r, r_on, r_cut) = xplor_switch(r, r_on, r_cut)

"""
    Smoothed(potential; r_on, r_cut, switch=:quintic)

Any pair potential brought smoothly to zero between `r_on` and `r_cut`: the energy is
`u(r) S(r)`, unchanged below `r_on` and zero from `r_cut` on. The default quintic switch
keeps the energy, force and second derivative continuous, as the Hessian needs;
`switch=:xplor` uses the XPLOR switch, whose second derivative jumps at both ends.

Distances are absolute. The `cutoff` given to `initialize_state` must be at least `r_cut`,
and `potential` itself must not be cut off before `r_cut`. Long-range tail corrections of
the wrapped potential are not applied.

```julia
params = Parameters(density, n_particles, dt, Smoothed(MyPotential(); r_on=1.8, r_cut=2.0))
```
"""
struct Smoothed{P<:Potential,S} <: Potential
    potential::P
    r_on::Float64
    r_cut::Float64
    switch::S
end

function Smoothed(potential::Potential; r_on::Real, r_cut::Real, switch::Symbol=:quintic)
    if !(0 < r_on < r_cut)
        throw(ArgumentError("need 0 < r_on < r_cut, got r_on = $r_on and r_cut = $r_cut"))
    end
    switch_type = if switch === :quintic
        QuinticSwitch()
    elseif switch === :xplor
        XPLORSwitch()
    else
        throw(ArgumentError("unknown switch :$switch, use :quintic or :xplor"))
    end
    return Smoothed(potential, Float64(r_on), Float64(r_cut), switch_type)
end

"""
    evaluate(pot::Smoothed, r, sigma1, sigma2)

Energy `u S` and force `f S - u dS/dr` of the switched potential.
"""
function evaluate(pot::Smoothed, r::Real, sigma1::Real, sigma2::Real)
    if r >= pot.r_cut
        return zero(float(r)), zero(float(r))
    end
    (u, f) = evaluate(pot.potential, r, sigma1, sigma2)
    if r < pot.r_on
        return u, f
    end
    (S, dS) = switch_value(pot.switch, r, pot.r_on, pot.r_cut)
    return u * S, f * S - u * dS
end

# Uses the wrapped potential's own fast path; the switch needs r only beyond r_on
@inline function evaluate_r2(pot::Smoothed, r2, sigma1, sigma2)
    if r2 >= pot.r_cut^2
        return zero(float(r2)), zero(float(r2))
    end
    (u, f_over_r) = evaluate_r2(pot.potential, r2, sigma1, sigma2)
    if r2 < pot.r_on^2
        return u, f_over_r
    end
    r = sqrt(r2)
    (S, dS) = switch_value(pot.switch, r, pot.r_on, pot.r_cut)
    return u * S, f_over_r * S - u * dS / r
end

"""
    check_cutoff(potential, cutoff)

Throw if the neighbor list `cutoff` is too short for `potential`.
"""
check_cutoff(::Potential, cutoff) = nothing

function check_cutoff(pot::Smoothed, cutoff)
    if cutoff < pot.r_cut
        throw(
            ArgumentError(
                "the cutoff ($cutoff) is shorter than r_cut = $(pot.r_cut) of the smoothed " *
                "potential; pass cutoff=$(pot.r_cut) or larger to `initialize_state`",
            ),
        )
    end
    return nothing
end
