# Carolina Kowalski Piazza
# ECO 528 - Macroeconomic Perspectives on Inequality
# Assignment 1: Solve numerically the McGee-Livshits-Tertilt economy described
# on the slides in partial equilibrium.
#

cd(dirname(@__FILE__))

using UnPack, LinearAlgebra, StatsBase, Plots
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
 
    function Params(MC::MarkovChain; β = 0.9, r = 0.02, q̅ = 1 / 1.02, γ = 0, N = 150)
	ygrid = exp.(MC.state_values)
	amin = -ygrid[end]
	amax = ygrid[end]
	a = range(amin, stop = amax, length = N)

	return new(β, r, q̅, γ, a, ygrid, N, MC)
    end

end


mutable struct ValueAndPolicy
	V::Array{Float64,2} # Value function
	value_market::Array{Float64,2} # Value function when in the mkt
	value_default::Array{Float64,2} # Value function when in autarky (default)
	q::Array{Float64,2} # Prices (inverse return to savings)
	savings_pol::Array{Int64,2} # savings policy function
	D_pol::Array{Int8, 2} # default decision

	function ValueAndPolicy(P::Params)
		@unpack β, r, q̅, γ, a, y, N, MC = P

		V = Array{Float64,2}(undef, (N, states)) # Vij = V(a[i], y[j])
		value_market =	 Array{Float64,2}(undef, (N, states))
		value_default = Array{Float64,2}(undef, (1, states))
		for i in 1:N
			for j in 1:states
				c = max(y[j] + a[i], 1e-5 )
				value_market[i, j] = (1 / (1 - β) * u(c)) # Initial guess
				V[i, j] = (1 / (1 - β) * u(c)) # Initial guess
				c̃ = y[j]
				value_default[1, j] = (1 / (1 - β)) * u(c̃) 
			end
		end

		q = q̅ * ones(Float64, N, states)
		savings_pol = round.(Int, ones((N, states))) # savings policy fuction (in indices)
		D_pol = zeros(Int8, N, states)
		return new(V, value_market, value_default, q, savings_pol, D_pol)
	end
end


function OneStepUpdate!(VP, EV, Evalue_market, Evalue_default)
	@unpack β, r, q̅, γ, a, y, N, MC = P
	objective = Array{Float64}(undef, N)
	objectivew = Array{Float64}(undef, N)
	for j in 1:states
		for k in 1:N
			if a[k] >= 0
				c̃ = max(y[j] - q̅ * a[k], 1e-10)
				objectivew[k] = u(c̃) + β * Evalue_market[k, j]
			else
				objectivew[k] = -Inf
			end
		end
		VP.value_default[1, j] = maximum(objectivew)
		# valuedefault =  u(y[j]) + β * Evalue_default[1, j] 
		# VP.value_default[1, j] = valuedefault

		for i in 1:N
			for k in 1:N
				if a[k] < 0
					consumption = max(y[j] + a[i] - VP.q[k, j] * a[k], 1e-10) # for each choice of savings, consumption is residual; can't be negative
				else
					consumption = max(y[j] + a[i] - q̅ * a[k], 1e-10)
				end
				objective[k] = u(consumption) + β * Evalue_market[k, j]
			end

			VP.savings_pol[i,j] = round.(Int,argmax(objective)) 
			# VP.value_market[i,j] = maximum(objective)
			newvalue = maximum(objective)
			if newvalue < VP.value_default[1, j]
				# VP.V[i, j] = VP.value_default[1, j]
				VP.value_market[i, j] = VP.value_default[1, j]
				VP.D_pol[i, j] = 1
			else
				# VP.V[i, j] = VP.value_market[i, j]
				VP.value_market[i, j] = newvalue
				VP.D_pol[i, j] = 0
			end
		end
	end
end

function FindPrice!(P::Params, VP::ValueAndPolicy)
	@unpack β, r, q̅, γ, a, y, N, MC = P
	default_states = VP.D_pol

   	# update price
	Π = transpose(MC.transition)
    	θ = default_states * Π # prob of default
    	copyto!(VP.q, (1 .- θ) * q̅ )

end

function ValueFunctionIteration!(P::Params, VP::ValueAndPolicy; max_iter = 1000, tolerance = 1e-5)
	@unpack β, r, q̅, γ, a, y, N, MC = P

	Π = transpose(MC.transition)
	ε = 10.0
	iterations = 0

	while ε > tolerance && iterations < max_iter
		EV = VP.V * Π
		Evalue_market = VP.value_market * Π
		Evalue_default = VP.value_default * Π
		old_value = copy(VP.value_market)
		OneStepUpdate!(VP, EV, Evalue_market, Evalue_default)	
		FindPrice!(P, VP)
		ε = maximum(abs.(VP.value_market - old_value))
		iterations += 1
		if iterations % 10 == 0
			veps = round(ε, digits = 6)
			println("iteration: $iterations, ε: $veps")
		end
	end
	println("Converged")
end

#------------------------------------------------------------------------------
# PARAMS
# AR1 process
ρ = 0.9
σ = 0.1
states = 3

#------------------------------------------------------------------------------
# Solve for the value function

MC = Tauchen(ρ, σ, states; m = 2)
P = Params(MC)
VP = ValueAndPolicy(P)
ValueFunctionIteration!(P, VP)


#------------------------------------------------------------------------------
# Plots


#------------------------------------------------------------------------------
# Assess Discretization with plots and moments

 println("Simulating paths...")
 path_3 = SimulateMC(MC; T = 5000)
 path_3 = MC.state_values[path_3]
 path = SimulateProcess(ρ, σ; T = 5000)

 T = length(path)

 println("Now plotting...")
 using Plots
 theme(:dark)
 plot(1:501, [path[T-500:end], path_3[T-500:end]],
 	label=["continuous" "3 states"], title="Process Discretization", legend=:best, size = (1000,500), lw = 1.5) # :outerright
 png("figures/1_simulatedprocess")

 pathw_list =  hcat(path[1001:end], path_3[1001:end])
 path_list = exp.(pathw_list)

 println("Calculate a few moments for comparison...")
 path_list = convert(Array{Float64}, path_list)
 moments = Array{Float64}(undef, (3,2))
 for j in 1:2
 	x = path_list[:,j]
 	moments[1,j] = mean(x)
 	moments[2,j] = var(x)
 	moments[3,j] = autocor(x, [1])[1]
 end


 display(moments)



# plot(P.a, VP.V, title = "Value Function", legend = :best, lw = 1.5)
# png("figures/valuefunction")

plot(P.a, VP.value_market, title = "Value Function when Participating in the Market", legend = :best, lw = 1.5)
png("figures/marketvaluefunction")

plot(P.a, VP.q, title = "Price Function (q)", legend = :best, lw = 1.5)
png("figures/price")

plot(P.a, P.a[VP.savings_pol], title = "Savings Policy", legend = :best, lw = 1.5)
png("figures/savings")




display(VP.value_default)



