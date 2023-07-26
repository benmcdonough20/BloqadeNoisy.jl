using Kronecker
using LinearAlgebra
"""
    struct ErrorModel

This struct  holds all of the information used to simulate coherent noise,
incoherent noise, and readout error for a specific error model. The
structure is as follows:

# Arguments

- confusion_mat: `(Int)->(T)` where `T`` is Matrix-like
- collapse_ops: `(Int)->Vector{SparseMatrixCSC}`
- coherent_noise: `(RydbergHamiltonian)->(()->RydbergHamiltonian)`
"""
struct ErrorModel
    confusion_mat::Function #used for noisy readout simulation
    collapse_ops::Function #collapse operators included in the Lindblaian
    coherent_noise::Function #method to modify the Hamiltonian each shot
end

"""
function measure_noisy

    This function mimicks YaoAPI.measure with noisy readout

#Arguments

- reg: statevector representing the quantum state to be measured
- noise_model: an ErrorModel struct which contains a method to create the confusion matrix
- site (optional): site to be measured
- nshots (optional kwarg): number of measurements to return
"""
function measure_noisy(
    u::Vector, 
    noise_model::ErrorModel, 
    sites=nothing; 
    nshots::Int = 1
    )
    nqubits = round(Int,log2(length(u)))
    cmat = noise_model.confusion_mat(nqubits) #generate confusion matrix
    w = Weights(cmat * abs.(u).^2) #create weights representing measurement probabilities
    return if site === nothing
        [DitStr{2}(digits(sample(w) .- 1; base = 2, pad = nqubits)) for i in 1:nshots]
    else
        try
            [DitStr{2}(digits(sample(w) .- 1; base = 2, pad = nqubits))[sites] for i in 1:nshots]
        catch BoundsError
            error("Site(s) not in range")
        end
    end
end

function rydberg_density_noisy_shots(
   u::Vector,
   noise_model::ErrorModel,
   op::Diagonal,
   shots
)
    nqubits = round(Int,log2(length(u)))
    cmat = noise_model.confusion_mat(nqubits) #generate confusion matrix
    w = Weights(cmat * abs.(u).^2) #create weights representing measurement probabilities
    sum([op.diag[sample(w)] for i in 1:shots])/shots
end