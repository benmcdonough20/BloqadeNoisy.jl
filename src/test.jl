using Bloqade
using Plots
using Kronecker
using StatsBase
using DifferentialEquations.EnsembleAnalysis
import BloqadeExpr.Hamiltonian
using Yao
using LinearAlgebra
using LaTeXStrings
include("error_model.jl")
include("noise_models.jl")
include("problem.jl")

reg = zero_state(1)
#works with constants and waveforms
h = rydberg_h([(0.0,0.0)], Ω = 2.5*2π, Δ = 2.5*2π)
tend = 10
save_times = LinRange(0, tend, 500)

#Add collapse operators to simulation
ns = NoisySchrodingerProblem(reg, tend, h, Aquila(); saveat = save_times)

sp = SchrodingerProblem(copy(reg), tend, h; save_on = true, saveat = save_times, save_start = true)
integrator = init(sp, DP8())
solve!(integrator)
plot(save_times, [abs(u[2])^2 for u in integrator.sol.u])

#get noisy expectation values
ntraj = 640
expecs = [mat(Op.n)]
output_func = u->convert(Int, measure_noisy(u, Aquila())[1])

sim = emulate(ns, ntraj; output_func = output_func, ensemble_algo = EnsembleSerial())

plot(save_times,
        expec_series_mean(sim, 1),
        ribbon = expec_series_err(sim, 1), color = :blue,
)
ylims!(0,1)

N = 1
cmat = Aquila().confusion_mat(N)
ind = [[1,1] for i in 1:N]
ind[1] = [0, 1]
sum((cmat*abs.(reg).^2)[site])

using CSV
using DataFrames
qutip_vals = CSV.read("/Users/queraintern/Documents/GitHub/manybody_fidelity/opensystem_simulation/3q_expect.csv",DataFrame, header = false, delim = ",")
qutip_vals_noiseless = CSV.read("/Users/queraintern/Documents/GitHub/manybody_fidelity/opensystem_simulation/3q_expect_nonoise.csv",DataFrame, header = false, delim = ",")
e = [collect(qutip_vals[i,:]) for i in 1:3]
f = [collect(qutip_vals_noiseless[i,:]) for i in 1:3]

plot!(0:1f-2:3.99, e[1], color = :red, linestyle = :dash, label = "QuTiP")
plot!(0:1f-2:3.99, e[2], color = :red, linestyle = :dash, label = "")
plot!(0:1f-2:3.99, e[3], color = :red, linestyle = :dash, label = "")

plot!(0:1f-2:3.99, f[1], color = :black, label = "noiseless", alpha = .3)
plot!(0:1f-2:3.99, f[2], color = :black,  label = "", alpha = .3)
plot!(0:1f-2:3.99, f[3], color = :black,  label = "", alpha = .3)
png("qutip_comparison.png")


timepoint_mean(sim, .5)
timeseries_point_mean(sim, save_times)
sim[1]
sim(1.0)
@edit sim[1](1.0)

#Using the EnsenbleAnalysis tools
ntraj = 100
output_func = (sol)->([[abs(u[1])^2, abs(u[3])^2] for u in sol.u])
sim = emulate(ns, ntraj)

fieldnames(typeof(sim))
sim.elapsedTime
sim.converged
length(sim[2])

timepoint_mean(sim, 1.0)

plot(save_times, [timestep_mean(sim, t)[1] for t in 1:length(save_times)], ribbon = [timestep_meanvar(sim,t)[2][1]/sqrt(ntraj) for t in 1:length(save_times)])
plot!(save_times, [timestep_mean(sim, t)[2] for t in 1:length(save_times)], ribbon = [timestep_meanvar(sim,t)[2][1]/sqrt(ntraj) for t in 1:length(save_times)])

sum([length(s)!=201 for s in sim])

timeseries_point_mean(sim[1], save_times).u
sim[1](1.0)
sim[1].u
[get_timepoint(sim[1], i) for i in save_times]


get_timepoint(sim, 1.0)