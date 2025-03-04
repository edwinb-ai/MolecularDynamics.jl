function register_images_and_wrap!(x, image, boxl)
    # Compute number of box crossings for each dimension
    n_cross = floor.(x ./ boxl)

    # Update the image vector
    @. image += Int(n_cross)

    # Wrap the position
    wrapped_x = @. x - boxl * n_cross

    return wrapped_x
end

function integrate_half!(positions, images, velocities, forces, dt::Float64, boxl::Float64)
    for i in eachindex(positions, forces, velocities)
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
    for i in eachindex(velocities, forces)
        f = forces[i]
        v = velocities[i]
        velocities[i] = @. v + (f * dt / 2.0)
    end

    return nothing
end