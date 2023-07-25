#TODO: These constants should be configurable, maybe via JSON?
p01 = .01
p10 = .08
relaxation_rate = 1/10.0
relax_op = (X+im*Y)/2
δΩrel = 0.008
δΔ = 0.18
δx = 0.05
δy = 0.05
δΔ_inhom = 0.37
δΩ_inhom = 0.02

function _aquila_coherent_noisy(h)
    (atoms,ϕ,Ω,Δ) = get_rydberg_params(h)

    function sample_noisy_ham() #return a function that modifies h
        randomize((x,y)) = (x+δx * randn(), y + δy*randn())
        atoms_noisy = randomize.(atoms) #randomize atom positions

        #add coherent drift in Ω and inhomogeneity
        Ω_noisy = Ω .* (1+δΩrel*randn() .+ δΩ_inhom * randn(length(atoms)))
        #add coherent drift in Δ and inhomogeneity
        Δ_noisy = δΔ*randn()+Δ .+ δΔ_inhom * randn(length(atoms))
        return rydberg_h(
            atoms_noisy;
            Ω = Ω_noisy,
            Δ = Δ_noisy,
            ϕ = ϕ
        )
    end
end

function _aquila_collapse_operators(nqubits)
    [
        sqrt(relaxation_rate) * mat(put(nqubits, q => relax_op)) #single-qubit relaxation
        for q in 1:nqubits
    ]
end

function _aquila_confusion_mat(N)
    M = [[1-p01 p10]; [p01 1-p10]]
    return kronecker([M for i in 1:N]...) #readout
end

"""
function Aquila

    Create an ErrorModel representing the noise model of Aquila.
"""
Aquila() = ErrorModel(
    _aquila_confusion_mat,
    _aquila_collapse_operators,
    _aquila_coherent_noisy,
)