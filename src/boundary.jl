# For cubic/square (orthorhombic) box
struct OrthoBoxSolver{N}
    boxlengths::SVector{N,Float64}
end

# For general box matrix
struct WrappedBoxSolver{N}
    boxmat::SMatrix{N,N,Float64}
    boxinv::SMatrix{N,N,Float64}
end

# Convenience constructors
OrthoBoxSolver(boxl::Float64, N::Int) = OrthoBoxSolver{N}(SVector{N,Float64}(boxl))

function OrthoBoxSolver(lengths::AbstractVector{<:Real})
    return OrthoBoxSolver{length(lengths)}(SVector{length(lengths),Float64}(lengths))
end

function WrappedBoxSolver(boxmat::SMatrix{N,N,Float64}) where {N}
    return WrappedBoxSolver{N}(boxmat, inv(boxmat))
end

# For cubic/square box (fast, per component)
function register_images_and_wrap!(
    x::SVector{N,Float64}, image::MVector{N,T}, solver::OrthoBoxSolver{N}
) where {N,T<:Integer}
    n_cross = floor.(x ./ solver.boxlengths)
    @. image += Int(n_cross)
    x_frac_wrapped = x .- n_cross .* solver.boxlengths
    return x_frac_wrapped
end

# For general box matrix
function register_images_and_wrap!(
    x::SVector{N,Float64}, image::MVector{N,T}, solver::WrappedBoxSolver{N}
) where {N,T<:Integer}
    x_frac = solver.boxinv * x
    n_cross = floor.(x_frac)
    @. image += Int(n_cross)
    x_frac_wrapped = x_frac .- n_cross
    wrapped_x = solver.boxmat * x_frac_wrapped
    return wrapped_x
end