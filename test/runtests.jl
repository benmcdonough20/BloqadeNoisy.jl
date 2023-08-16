using DifferentialEquations
using BloqadeNoisy
using Test
using Bloqade
using Yao
using LaTeXStrings
using LinearAlgebra
using Kronecker

@testset "BloqadeNoisy.jl" begin
    # Write your tests here.
    reg = zero_state(1)
    h = rydberg_h([(0,0)], Ω = 15, Δ = 0)
    save_times = LinRange(0, 10, 500)
    ns = NoisySchrodingerProblem(reg, save_times, h, Aquila())
    sim = emulate(ns, 1)
    @test length(sim) == length(save_times)
    @test length(first(sim)) == 2
    end_state = sim[end]
    @test length(measure_noisy(Aquila(), end_state; nshots = 1)) == 1
    sim = emulate(ns, 1, mat.([X,Z]))
    @test length(sim) == 2
    @test length(first(sim)) == length(save_times)
    @test sim[1][1] == 0.0
    @test sim[2][1] == 1.0
    p01 = Aquila().confusion_mat(1)[2,1]
    sim = emulate(ns, 1, [mat(Z)]; readout_error = true)
    @test sim[1][1] == 1-2*p01
    h = rydberg_h([(0,0), (8,0), (18,0)], Ω = 15, Δ = 0)
    save_times = LinRange(0, 4, 200) #choose the times at which to save the solution
    reg = zero_state(3)
    ns = NoisySchrodingerProblem(reg, save_times, h, Aquila())
    sim1 = emulate(ns, 1, [mat(put(3, i=>Z)) for i in 1:3]; report_error = true)
    sim2 = emulate(ns, 2, [mat(put(3, 1=>Z))]; report_error = true)
    sim_ro = emulate(ns, 2, [mat(put(3, 1=>Z))]; readout_error = true, report_error = true)
    sim = emulate(ns, 1, [mat(put(3, i=>Z)) for i in 1:3]; readout_error = true, report_error = true, shots = 10)
    sim = emulate(ns, 1)
    measure_noisy(Aquila(), sim[100]; nshots = 10)
    sim = emulate(ns, 1, sol -> [abs(u[1])^2 for u in sol])
    values = simulation_series_mean(sim)
    error = simulation_series_err(sim) 
    ep = EnsembleProblem(ns, prob_func = (prob, i, repeat)->randomize(ns))
    solve(ep, trajectories = 1)
    rate = 1/10
    reg = zero_state(1)
    c_ops = [sqrt(rate)*mat((X+im*Y)/2)] #this noise model will not have any coherent error
    h = rydberg_h([(0,0)], Ω = 15)
    ns = NoisySchrodingerProblem(reg, save_times, h, c_ops)
    trivial_error_model = ErrorModel(
        n -> I,
        n -> [],
        h -> (() -> h)
    )
    ns = NoisySchrodingerProblem(reg, [0, 4], h, trivial_error_model)
    confusion_matrix(n) = kronecker([[[.9 .1];[.1 .9]] for i in 1:n]...)
    bitflip_model(n) = [SparseMatrixCSC(sqrt(1/10)*mat(put(n, i=>X))) for i in 1:n]
    coherent_noise(h) = () -> ((atoms,ϕ,Ω,Δ)=get_rydberg_params(h); rydberg_h(atoms; Ω = Ω*(1+.08*randn()), Δ = Δ, ϕ = ϕ))
    better_error_model = ErrorModel(
        confusion_matrix,
        bitflip_model,
        coherent_noise
    )
    ns = NoisySchrodingerProblem(zero_state(2), 0:1f-2:1, rydberg_h([(0,),(8)]; Ω = 15), better_error_model)
    sim = emulate(ns, 1, [mat(put(2, 1=>X)), mat(kron(X, X))])
end
