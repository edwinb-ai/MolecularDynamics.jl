function sum_noises(nf, rng)
    result = 0.0

    if nf == 0.0
        result = 0.0
    elseif nf == 1.0
        result = randn(rng)^2
    elseif mod(nf, 2) == 0
        gamma_dist = Gamma(nf ÷ 2)
        result = 2.0 * rand(rng, gamma_dist)
    else
        gamma_dist = Gamma((nf - 1) ÷ 2)
        result = 2.0 * rand(rng, gamma_dist)
        result += randn(rng)^2
    end

    return result
end

function bussi!(velocities, ktemp, nf, dt, τ, rng)
    dt_ratio = dt / τ

    # Compute kinetic energy
    kinetic_energy = 0.0
    @inbounds for i in eachindex(velocities)
        kinetic_energy += sum(abs2, velocities[i])
    end
    kinetic_energy /= 2.0
    current_temperature = 2.0 * kinetic_energy / nf

    # Compute random numbers
    r1 = randn(rng)
    r2 = sum_noises(nf - 1, rng)

    # Compute the parameters from the thermostat
    term_1 = exp(-dt_ratio)
    c2 = (1.0 - term_1) * ktemp / (current_temperature * nf)
    term_2 = c2 * (r2 + r1^2)
    term_3 = 2.0 * r1 * sqrt(term_1 * c2)
    scale = sqrt(term_1 + term_2 + term_3)

    # Apply velocity rescaling
    @inbounds for i in eachindex(velocities)
        velocities[i] = velocities[i] * scale
    end

    return nothing
end

function compute_kinetic(velocities)
    kinetic_energy = 0.0

    @inbounds for i in eachindex(velocities)
        kinetic_energy += sum(abs2, velocities[i])
    end

    kinetic_energy /= 2.0

    return kinetic_energy
end

function compute_temperature(velocities, nf)
    kinetic_energy = compute_kinetic(velocities)
    temperature = 2.0 * kinetic_energy / nf

    return temperature
end
