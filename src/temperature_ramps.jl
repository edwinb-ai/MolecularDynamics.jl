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
    # Ensure we handle the post-ramp period correctly
    if step >= ramp.n_steps
        return ramp.T_final
    end
    
    # Clamp step to valid range [0, n_steps-1] for 0-indexed steps
    step = clamp(step, 0, ramp.n_steps - 1)
    
    # Linear interpolation - adjusted for 0-indexed steps
    if ramp.n_steps == 1
        return ramp.T_final
    end
    
    progress = step / (ramp.n_steps - 1)
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
    # Ensure we handle the post-ramp period correctly
    if step >= ramp.n_steps
        return ramp.T_final
    end
    
    # Clamp step to valid range [0, n_steps-1] for 0-indexed steps
    step = clamp(step, 0, ramp.n_steps - 1)
    
    # Handle edge cases
    if ramp.n_steps == 1 || ramp.T_initial == ramp.T_final
        return ramp.T_final
    end
    
    # Calculate exponential factor - adjusted for 0-indexed steps
    progress = step / (ramp.n_steps - 1)
    α = log(ramp.T_final / ramp.T_initial)
    return ramp.T_initial * exp(α * progress)
end
