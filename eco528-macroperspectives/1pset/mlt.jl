# Carolina Kowalski Piazza
# ECO 528 - Macroeconomic Perspectives on Inequality
# Assignment 1: Solve numerically the McGee-Livshits-Tertilt economy described
# on the slides in partial equilibrium.
#

cd(dirname(@__FILE__))

using UnPack
include("tauchen.jl")

u(c) = log(c)
u(c::Vector) = log.(c)


mutable struct Params
    β::Real
    r::Real
    q̅::Real
    γ::Real
    a::AbstractVector{Real}
    y::AbstractVector{Real}
    N::Integer
    MC::MarkovChain
 
    function Params(MC::MarkovChain; β = 0.9, r = 0.02, q_ = 1 / 1.02, γ = 0, N = 100)
	ygrid = exp.(MC.state_values)
	amin = -ygrid[1]/r
	amax = ygrid[end]/r
	a = range(amin, stop = amax, length = N)

	return new(β, r, q̅, γ, a, ygrid, N, MC)
    end

end


mutable struct ValueAndPolicy
	V::Array{Float64,2} # Value function when in the mkt
	W::Array{Float64,2} # Value function when in autarky (default)
	savings_pol::Array{Int64,2} # savings policy function
	c_pol::Array{Float64,2} # consumption policy function
	D_pol::Array{Int8, 2} # default decision

	function ValueAndPolicy(P::Params)
		@unpack β, r, q̅, γ, a, y, N, MC = P

		V = Array{Float64,2}(undef, (N, states)) # Vij = V(a[i], y[j])
		W = Array{Float64,2}(undef, (1, states)) # W1j = W(0, y[j])
		for i in 1:N
			for j in 1:states
				c = max(y[j] + a[i], 1 )
				V[i, j] = (1 / (1 - β) * u(c)) # Initial guess
				c̃ = y[j]
				W[1, j] = (1 / (1 - β)) * u(c̃) 
			end
		end
		c_pol = Array{Float64,2}(undef, (N, states)) #cij = c(a[i], y[j])
		savings_pol = round.(Int, zeros((N, states))) # savings policy fuction (in indices)
		D_pol = zeros(Int8, N, states)
		return new(V, W, savings_pol, c_pol, D_pol)

	end
end

function OneStepUpdate()
	for i in 1:N
		for j in 1:states
			for k in 1:N
				objective[k] = u(y[j] + (1 + r)*a[i] -a[k], P) + β*dot(transition[j,:], old_V[k, :])
			end
			next_a[i,j] = round.(Int,argmax(objective)) 
			new_V[i,j] = maximum(objective)
			c_pol[i,j] = y[j] + (1 + r)*a[i] - a[next_a[i,j]]
		end
	end
	return()
end

function ValueFunctionIteration!(P::Params, VP::ValueAndPolicy; max_iter = 1000, tolerance = 1e-5)
	@unpack β, r, q̅, γ, a, y, N, MC = P

	Π = MC.transition
	ε = 10.0
	iterations = 0

	while ε > tolerance && iterations < max_iter
		EV = Π * VP.V
		EW = Π * VP.W

		old_V = copy(VP.V)
		OneStepUpdate!(VP, ExpV, ExpW)	
		FindPrice(VP)

		ε = maximum(abs.(VP.V - old_V))
		iterations += 1
		if iterations % 10 == 0
			veps = round(ε, digits = 6)
			println("iteration: $iterations, ε: $veps")
		end
	end
	println("Converged")
end


# PARAMS
# AR1 process
ρ = 0.9
σ = 0.1
states = 3

MC = Tauchen(ρ, σ, states; m = 2)

P = Params(MC)


