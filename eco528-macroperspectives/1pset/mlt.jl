# Carolina Kowalski Piazza
# ECO 528 - Macroeconomic Perspectives on Inequality
# Assignment 1: Solve numerically the McGee-Livshits-Tertilt economy described on the slides in partial equilibrium.
#

include("tauchen.jl")

u(c) = log(c)
u(c::Vector) = log.(c)

# PARAMS

β = 0.9
r = 0.02
q_ = 1 / 1.02
γ = 0

# AR1 process
ρ = 0.9
σ = 0.1
states = 3

MC = tauchen(ρ, σ, states; m = 2)

