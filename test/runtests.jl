using Test

include("helpers.jl")

@testset "MolecularDynamics.jl" begin
    include("test_potentials.jl")
    include("test_neighborlist.jl")
    include("test_dynamics.jl")
    include("test_minimize.jl")
    include("test_io.jl")
end
