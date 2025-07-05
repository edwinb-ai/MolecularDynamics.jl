const sqthree = sqrt(3.0)

"""
    register_images_and_wrap!(x, image, unitcell)

Generic periodic wrap for any simulation box shape (cubic, orthorhombic, or triclinic).

- `x`: position vector (SVector, Vector)
- `image`: image vector (MVector, Vector{Int}) to be updated in-place
- `unitcell`: simulation box (scalar, vector, or 2D matrix of box vectors)

Returns:
    wrapped_x: wrapped Cartesian position (same type as x)
"""
function register_images_and_wrap!(x, image, unitcell)
    # Convert box to matrix form
    boxmat = if isa(unitcell, Number)
        unitcell * I(length(x))
    elseif isa(unitcell, AbstractVector)
        Diagonal(unitcell)
    elseif isa(unitcell, AbstractMatrix)
        unitcell
    else
        throw(ArgumentError("Unsupported unitcell type"))
    end

    # Compute fractional coordinates: x_frac = boxmat \ x
    x_frac = boxmat \ x

    # Number of box crossings (integer part)
    n_cross = floor.(x_frac)

    # Update images
    @. image += Int(n_cross)

    # Wrap: back to [0,1) in each direction
    x_frac_wrapped = x_frac .- n_cross

    # Convert back to Cartesian
    wrapped_x = boxmat * x_frac_wrapped

    return wrapped_x
end

function integrate_half!(positions, images, velocities, forces, dt, boxl)
    @threads for i in eachindex(positions, forces, velocities)
        f = forces[i]
        x = positions[i]
        v = velocities[i]
        # ! Important: There is a mass in the force term
        velocities[i] = @. v + (f * dt / 2.0)
        positions[i] = @. x + (velocities[i] * dt)
        positions[i] = register_images_and_wrap!(positions[i], images[i], boxl)
    end

    return nothing
end

function integrate_second_half!(velocities, forces, dt)
    @threads for i in eachindex(velocities, forces)
        f = forces[i]
        v = velocities[i]
        velocities[i] = @. v + (f * dt / 2.0)
    end

    return nothing
end

function ensemble_step!(
    ::NVE, velocities, params::Parameters, state::SimulationState, step::Int
)
    return compute_temperature(velocities, state.nf)
end

function ensemble_step!(
    ensemble::NVT, velocities, params::Parameters, state::SimulationState, step::Int
)
    temperature = ensemble.ktemp(step)
    # Apply thermostat, e.g., Bussi thermostat
    bussi!(velocities, temperature, state.nf, params.dt, ensemble.tau, state.rng)
    return compute_temperature(velocities, state.nf)
end

@inline function sample_uniform!(vector, rng)
    rand!(rng, vector)
    @. vector = (2.0 * vector - 1.0) * sqthree

    return nothing
end

function integrate_brownian!(
    positions, images, forces, dt, boxl, rng, dimension, ktemp, sigma
)
    noise = zeros(MVector{dimension,Float64})

    @inbounds for i in eachindex(positions, forces)
        f = forces[i]
        x = positions[i]
        sample_uniform!(noise, rng)
        positions[i] = @. x + (f * dt / ktemp) + (noise * sigma)
        positions[i] = register_images_and_wrap!(positions[i], images[i], boxl)
    end

    return nothing
end
