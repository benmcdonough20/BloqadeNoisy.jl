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
    noise_model::ErrorModel, 
    amps::Vector{T} where T <: Real, 
    sites=nothing; 
    nshots::Int = 1
    )
    nqubits = round(Int,log2(length(amps)))
    cmat = noise_model.confusion_mat(nqubits) #generate confusion matrix
    w = Weights(cmat * amps) #create weights representing measurement probabilities
    if sites === nothing; sites = 1:length(amps); end
    [DitStr{2}(digits(sample(w) .- 1; base = 2, pad = nqubits)) for i in 1:nshots]
end

function expectation_value_noisy(
   noise_model::ErrorModel,
   amps::Vector{T} where T <: Real,
   op::Diagonal;
   errs = nothing
)
    nqubits = round(Int,log2(length(amps)))
    cmat = noise_model.confusion_mat(nqubits) #generate confusion matrix
    expec = sum([a * real(n) for (a,n) in zip(cmat * amps, op.diag)])
    return if errs === nothing
        expec
    else
        (
            expectation = expec,
            propagated_err = sqrt(sum([(err * real(n))^2 for (err,n) in zip(cmat * errs, op.diag)]))
        )
    end
end

function expectation_value_noisy(
   noise_model::ErrorModel,
   amps::Vector{T} where T <: Real,
   op::Diagonal,
   shots;
   errs = false
)
    nqubits = round(Int,log2(length(amps)))
    cmat = noise_model.confusion_mat(nqubits) #generate confusion matrix
    w = Weights(cmat * amps) #create weights representing measurement probabilities
    S = [real(op.diag[sample(w)]) for i in 1:shots]
    return if errs === true
        mean(S)
    else
        (
            expectation = mean(S),
            sample_err = 2*std(S)/sqrt(shots)
        )
    end
end