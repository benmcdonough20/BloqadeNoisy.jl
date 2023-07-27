# BloqadeNoisy

A framework for stochastic wavefunction simulations of open quantum systems in Bloqade.

## Features
- Coherent noise
- Readout error
- Incoherent noise
- Custom code injection
- Custom noise models
- Parallel processing

## Usage

See `tutorials/tutorial.ipynb` for a demonstration of functionality.

## Background

More information about the stochastic wavefunction method implemented can be found at the following resources:
- [qutip mcsolve documentation](https://qutip.org/docs/latest/guide/dynamics/dynamics-monte.html)
- [Lukin course notes ch. 6](https://lukin.physics.harvard.edu/files/lukin/files/physics_285b_lecture_notes.pdf)

## Further development
- Improve memory allocation
- Run on GPU

## Dependencies
- Julia 1.9.1
- Bloqade v0.1.24
- StatsBase v0.33.21
- SparseArrays
- Kronecker v0.5.4
- DifferentialEquations v7.8.0
- StatsBase v0.33.21
- Yao v0.8.9
