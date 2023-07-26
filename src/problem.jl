"""
    struct NoiseSchrodingerEquation

Decorator object for SchrodingerEquation. Implements the f(dstate, state, p, t)
interface to fit into a standard ODE Problem.

# Arguments

- equation: The coherent equation for the problem
- imag_evo: ∑_L L^†L for collapse operators L which defines the non-hermitian part of the Hamiltonian
- num_cops: number of collapse operators (for displaying)
"""
struct NoisySchrodingerEquation{T<:SparseMatrixCSC}
    equation::SchrodingerEquation
    imag_evo::T
    num_cops::Int
end

function (eq::NoisySchrodingerEquation)(dstate, state, p, t::Number)
    eq.equation(dstate, state, p, t)
    mul!(dstate, eq.imag_evo, state, -.5, one(t))
    return
end

function Base.show(io::IO, mime::MIME"text/plain", eq::NoisySchrodingerEquation)
    Base.show(io, mime, eq.equation) 
    printstyled(io, "collapse operators: ", length(eq.num_cops); color = :yellow)
    return println(io)
end

"""
    struct NoisySchrodingerEquation
    NoisySchrodingerEquation(reg, tspan, hamiltonian, c_ops; kw...)
    NoisySchrodingerEquation(reg, tspan, hamiltonian, noise_model; kw...)

Define an ODE Problem representing a single noisy trajectory of an open system

# Arguments

- reg: evolution register and initial state of the problem
- tspan: either (start, stop) or the end time
- hamiltonian: AbstractBlock representing evolution hamiltonian, must be compatible with noise model
"""
struct NoisySchrodingerProblem{Reg,EquationType<:ODEFunction,uType,tType,Algo,Kwargs} <:
       SciMLBase.AbstractODEProblem{uType,tType,true}
    reg::Reg
    f::EquationType
    c_ops::Vector{T} where T <: SparseMatrixCSC
    state::uType
    u0::uType
    tspan::tType
    algo::Algo
    coherent_noise::Function
    confusion_mat #Matrix-like
    save_times
    kwargs::Kwargs
    p::Real
end

#internal constructor
function NoisySchrodingerProblem(
    reg::AbstractRegister,
    save_times,
    expr,
    c_ops::Vector{T} where T <: SparseMatrixCSC,
    coherent_noise::Function,
    confusion_mat; #TODO: type checking
    algo=DP8()
)
    nqudits(reg) == nqudits(expr) || throw(ArgumentError("number of qubits/sites does not match!"))

    tspan = (first(save_times), last(save_times))

    state = statevec(reg)
    space = YaoSubspaceArrayReg.space(reg)

    T = real(eltype(state))
    T = isreal(expr) ? T : Complex{T}
    eq = NoisySchrodingerEquation(
        SchrodingerEquation(
            expr, Hamiltonian(T, expr,space)
        ),
        sum([l'*l for l in c_ops]),
        length(c_ops)
    )
    ode_f = ODEFunction(eq)

    tspan_type = promote_type(real(eltype(state)), eltype(tspan))
    tspan = tspan_type.(tspan) # promote tspan to T so Dual number works

    jump_cb = ContinuousCallback( #trigger quantum jumps and collapse the state
        collapse_condition,
        (integrator)->collapse!(integrator, c_ops),
    ) 

    #remove save_start and sate_on options to allow saving at specified times
    ode_options = pairs((
        save_everystep = false,
        dense = false,
        reltol=1e-10,
        abstol=1e-10,
        saveat=save_times,
        callback = jump_cb #quantum jumps
    ))

    return NoisySchrodingerProblem(
        reg,
        ode_f,
        c_ops,
        state,
        copy(state),
        tspan,
        algo,
        coherent_noise,
        confusion_mat,
        save_times,
        ode_options,
        rand(), #random initial condition
    )
end

function NoisySchrodingerProblem(
    reg,
    save_times,
    h, 
    c_ops::Array;
    kw...
)
    return NoisySchrodingerProblem(
       reg,
       save_times,
       h,
       c_ops,
       () -> h,
       () -> nothing;
       kw...
    )
end

function NoisySchrodingerProblem(
    reg::ArrayReg,
    save_times,
    h::AbstractBlock,
    noise_model::ErrorModel;
    kw...
)
    return NoisySchrodingerProblem(
        reg,
        save_times,
        h,
        noise_model.collapse_ops(nqubits(reg)),
        noise_model.coherent_noise(h),
        noise_model.confusion_mat(nqubits(reg));
        kw...
    )
end

function collapse_condition(u, t, integrator)
    norm(u)^2 - integrator.p
end

function collapse!(integrator, L_ops)
    dp = 0
    l = length(L_ops)
    probs = Vector{Float64}(undef, l)
    for i in 1:l
        dp += norm(L_ops[i] * integrator.u)^2 #normalization
        probs[i] = dp #cumulative distribution
    end
    r = rand()
    for i in 1:l
        if r <= probs[i]/dp #choose jump based on r
            copy!(integrator.u, L_ops[i]*integrator.u) #jump
            normalize!(integrator.u) #normalize
            break
        end
    end
    integrator.p = rand()
end

function rebuild(prob::NoisySchrodingerProblem, p, h)
    space = YaoSubspaceArrayReg.space(prob.reg)
    T = real(eltype(prob.state))
    T = isreal(h) ? T : Complex{T}
    eq = NoisySchrodingerEquation(
        SchrodingerEquation(
            h, Hamiltonian(T, h, space)
        ),
        sum([l'*l for l in prob.c_ops]),
        length(prob.c_ops)
    )
    ode_f = ODEFunction(eq)
    return NoisySchrodingerProblem(
        copy(prob.reg),
        ode_f,
        prob.c_ops,
        copy(prob.state),
        copy(prob.u0),
        prob.tspan,
        prob.algo,
        prob.coherent_noise,
        prob.confusion_mat,
        prob.save_times,
        prob.kwargs,
        p
    )
end

function Base.show(io::IO, mime::MIME"text/plain", prob::NoisySchrodingerProblem)
    indent = get(io, :indent, 0)
    tab(indent) = " "^indent

    println(io, tab(indent), "NoisySchrodingerProblem:")
    # state info
    println(io, tab(indent + 2), "register info:")
    print(io, tab(indent + 4), "type: ")
    printstyled(io, typeof(prob.reg); color = :green)
    println(io)

    print(io, tab(indent + 4), "storage size: ")
    printstyled(io, Base.format_bytes(storage_size(prob.reg)); color = :yellow)
    println(io)
    println(io)

    # tspan info
    println(io, tab(indent + 2), "time span (μs): ", prob.tspan)
    println(io)

    # equation info
    println(io, tab(indent + 2), "equation: ")
    show(IOContext(io, :indent => indent + 4), mime, prob.f.f)
    println(io, tab(indent + 4), "algorithm: ", repr(prob.algo))
end

function DiffEqBase.solve(prob::NoisySchrodingerProblem, args...; sensealg = nothing, initial_state = nothing, kw...)
    if sensealg === nothing && haskey(prob.kwargs, :sensealg)
        sensealg = prob.kwargs[:sensealg]
    end
    # update initial state
    if initial_state !== nothing
        initial_state isa AbstractRegister ||
            throw(ArgumentError("initial_state must be a register, got $(typeof(initial_state))"))
        u0 = statevec(initial_state)
    else
        u0 = prob.u0
    end
    return DiffEqBase.solve_up(prob, sensealg, u0, nothing, args...; kw...)
end

DiffEqBase.get_concrete_problem(prob::NoisySchrodingerProblem, isadapt; kw...) = prob

"""
    function emulate

Emulate the evolution of a noisy system

# Arguments
- prob: NoisySchrodingerProblem to emulate
- ntraj: number of trajectories to use for simulation
- expectations: Matrices representing observables
- output_func: Vector{Complex}->Any - Transformation of the statevector to save
- ensemble_algo: See EnsembleProblem documentation. Common choices are EnsembleSerial and EnsembleThreaded
"""
function emulate(
    prob::NoisySchrodingerProblem,
    ntraj::Int;
    expectations = [],
    output_func = u -> u,
    ensemble_algo = EnsembleSerial(),
)

    @assert all(isreal.(expectations)) #check that expectation operators are real

    prob_func(prob, i, repeat) = rebuild(prob, rand(), prob.coherent_noise())

    if !isempty(expectations)
        output_func = (u) -> [real(u' * e * u) for e in expectations]
    end

    #IMPORTANT: see ContinuousCallback documentation. Because integrator state is saved before and after callback is triggered
    #for accuracy, the timeseries end up being different lengths. This is why interpolation is used, but maybe there is a better solution
    ensemble_prob = EnsembleProblem(prob, prob_func = prob_func, output_func = (sol, i) -> ([output_func(normalize(sol(t))) for t in prob.save_times], false))
    solve(ensemble_prob, prob.algo, ensemble_algo, trajectories = ntraj)
end

"""
    function emulate
Emulate the evolution of a noisy system
# Arguments
- prob: NoisySchrodingerProblem to emulate
- ntraj: number of trajectories to use for simulation
- noisy_expectations: Vector of operators that are diagonal in computational basis 
- shots: If specified, the noisy_expectation values will be taken with simulated shots
"""
function emulate(
    prob::NoisySchrodingerProblem,
    ntraj::Int,
    noisy_expectations::Vector{T} where T <: Diagonal;
    shots = nothing,
    kw...
)

    @assert all(isreal.(noisy_expectations)) #check that expectation operators are real

    if shots === nothing
        output_func = (u) -> [sum([real(e.diag[n]) * p for (n,p) in enumerate(prob.confusion_mat * abs.(u).^2)]) for e in noisy_expectations]
        emulate(prob, ntraj; output_func = output_func, kw...)
    else
        sim = emulate(prob, ntraj; output_func = u -> abs.(u)^2, kw...) #get probability amplitudes
        prob_amps = expec_series_mean(sim, 1:length(prob.u0)) #average probabilitiy amplitdues
        expec_shots(e, p) = (
            w = Weights(prob.confusion_mat * p); #create weights and apply confusion matrix
            mean([real(e.diag[sample(w)]) for i in 1:shots]) #sample expectation value
        )
        #output format is a timeseries list for each expectation value
        return [[expec_shots(e, p) for p in prob_amps] for e in noisy_expectations]
    end 

end

"""
    function expec_series_mean

Convenience method to access the expectation value over the ensemble
over the series of save times.

# Arguments
- sim: EnsembleSolution, result of calling `emulate`
- index: index of the expectation value in the array provided
"""
function expec_series_mean(sim, index)
    ntraj = length(sim)
    times = length(sim[1])
    [mean([sim[i][t][index] for i in 1:ntraj]) for t in 1:times]
end

function expec_series_err(sim, index)
    ntraj = length(sim)
    times = length(sim[1])
    [std([sim[i][t][index] for i in 1:ntraj])/sqrt(ntraj) for t in 1:times]
end
