include("src/MolecularDynamicsCells.jl")

function main()
    # densities = [0.776, 0.78, 0.82, 0.84, 0.86, 0.9]
    # reps = 10
    densities = [0.9]
    ktemp = 0.85

    for d in densities
        params = Parameters(d, ktemp, 4000)
        # Create a new directory with these parameters
        pathname = joinpath(@__DIR__, "density=$(@sprintf("%.4g", d))")
        mkpath(pathname)
        simulation(params, pathname; eq_steps=100_000, prod_steps=1_000_000)
    end

    return nothing
end

main()
