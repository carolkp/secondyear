
# Function to discretize an AR(1) process
# Returns a Markov Chain transition matrix and the grid of y
#
using Distributions, Random


mutable struct MarkovChain # Tauchen method function will return this kind of object
	states::Integer 
 	transition::AbstractArray{Real}
    	state_values::AbstractVector{Real}
end

function Tauchen(ρ::T, σ::T, number_states::Integer; m::Integer = 2) where{T<:Real}
	# Choose grid
	ymax = m * sqrt((σ^2) / (1 - ρ^2))
	ymin = -ymax
	y = range(ymin, stop = ymax, length = number_states)
	d = step(y)
	
	# Transition matrix
	transition = Matrix{Float64}(undef, number_states, number_states)
	
	for j in 1:number_states
		for k in 1:number_states
			if k == 1
				transition[j, k] = cdf(Normal(0, σ), y[1] + 0.5 * d - ρ * y[j])
			elseif k == number_states
				transition[j, k] = 1 - cdf(Normal(0, σ), y[number_states] - 0.5 * d - ρ * y[j])
			else
				transition[j, k] = cdf(Normal(0, σ), y[k] + 0.5 * d - ρ * y[j]) - cdf(Normal(0, σ), y[k] - 0.5 * d - ρ * y[j])
			end
		end
	end

	return MarkovChain(number_states, transition, y) 
end

# Simulate Markov Chain trajectory
# Returns history of indices of visited states, instead of values
function SimulateMC(MC::MarkovChain; T::Integer = 10000)
	state_values = MC.state_values
	transition = MC.transition
	states = MC.states

	draws = rand(T)
	history = Array{Int64}(undef, T)
	idx = ceil(median(1:states))
	history[1] = idx
	cdf_transition = cumsum(transition; dims = 2)

	for t in 2:T
		current_state = history[t-1]
		next_state = findfirst(x -> x > draws[t], cdf_transition[current_state, :] )
		history[t] = next_state
	end

	return(history)
end


function SimulateProcess(ρ, σ; T = 1000)
	wbar = -(σ^2) / (2 * (1 + ρ))

	draws = rand(Normal(0, σ), T)
	history = Array{Float64}(undef, T)
	history[1] = wbar

	for t in 2:T
		history[t] = wbar + ρ * history[t-1] + draws[t]
	end

	path = history
	return(path)
end


