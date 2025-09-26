"""
    LinearRamp(T_initial, T_final, n_steps)

A callable struct representing a linear temperature ramp from `T_initial` to `T_final` over `n_steps` steps.
Calling the ramp with a step index returns the temperature at that step.
"""
struct LinearRamp
    T_initial::Float64
    T_final::Float64
    n_steps::Int
end

function (ramp::LinearRamp)(step::Int)
    # Handle the post-ramp period correctly - return exact final temperature
    if step > ramp.n_steps
        return ramp.T_final
    end
    
    # Clamp step to valid range [1, n_steps] for 1-indexed steps (as called from simulation)
    step = clamp(step, 1, ramp.n_steps)
    
    # Linear interpolation - adjusted for 1-indexed steps
    if ramp.n_steps == 1
        return ramp.T_final
    end
    
    progress = (step - 1) / (ramp.n_steps - 1)
    return ramp.T_initial + (ramp.T_final - ramp.T_initial) * progress
end

"""
    ExponentialRamp(T_initial, T_final, n_steps)

A callable struct representing an exponential temperature ramp from `T_initial` to `T_final` over `n_steps` steps.
"""
struct ExponentialRamp
    T_initial::Float64
    T_final::Float64
    n_steps::Int
end

function (ramp::ExponentialRamp)(step::Int)
    # Handle the post-ramp period correctly - return exact final temperature
    if step > ramp.n_steps
        return ramp.T_final
    end
    
    # Clamp step to valid range [1, n_steps] for 1-indexed steps (as called from simulation)
    step = clamp(step, 1, ramp.n_steps)
    
    # Handle edge cases
    if ramp.n_steps == 1 || ramp.T_initial == ramp.T_final
        return ramp.T_final
    end
    
    # Calculate exponential factor - adjusted for 1-indexed steps
    progress = (step - 1) / (ramp.n_steps - 1)
    α = log(ramp.T_final / ramp.T_initial)
    return ramp.T_initial * exp(α * progress)
end

"""
    initial_temperature_for_velocities(ktemp)

Returns the initial temperature for velocities based on the provided `ktemp`.
"""
function initial_temperature_for_velocities(ktemp)
    if hasproperty(ktemp, :T_initial) && hasproperty(ktemp, :T_final)
        return max(ktemp.T_initial, ktemp.T_final)
    else
        return ktemp
    end
end
