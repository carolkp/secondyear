# Carolina Kowalski Piazza
# ECO 528 - Macroeconomic Perspectives on Inequality
# Assignment 2: Solve the Life-Cycle model
#

cd(dirname(@__FILE__))

using UnPack, LinearAlgebra, SparseArrays

δ(t) = 1/(1 + 500*exp(-0.2*t))
x(t) = 1 + 0.04*t -0.0006*t^2
# y(t) = z(t)*x(t)

mutable struct MarkovChain # Tauchen method function will return this kind of object
	states::Integer 
 	transition::AbstractArray{Real}
    	state_values::AbstractVector{Real}

	function MarkovChain(;states = 3, state_values = [0.7, 1, 1.3], transition = [0.7 0.2 0.1; 0.15 0.7 0.15; 0.1 0.2 0.7])
		return new(states, transition, state_values)
	end
		

end

mutable struct Model
    r::Real
    ρ::Real 	# annual discount rate
    γ::Real 	# relative risk aversion
    T::Integer 	# max period
    Na::Integer # number of points in asset grid
    NT::Integer # number of points in T grid
    a::StepRangeLen{Real, Real, Real}
    timegrid::StepRangeLen{Real, Real, Real}
    MC::MarkovChain
 
    function Model(MC::MarkovChain; ρ = 0.03, r = 0.025, γ = 2, T = 40, Na = 50, NT = 120)
	amin = 0
	amax = 30
	a = range(amin, stop = amax, length = Na)
	timegrid = range(0, stop = T, length = NT)
	return new(r, ρ, γ, T, Na, NT, a, timegrid, MC)
    end
end


mutable struct ValueAndPolicy
	V::Array{Float64,3} # Value function
	consumption_pol::Array{Float64, 3} # consumption policy
	savings_pol::Array{Int64,3} # savings policy function

	function ValueAndPolicy(M::Model)
		@unpack r, ρ, γ, T, Na, NT, a, timegrid, MC = M
		dt = step(timegrid)
		x_t = x.(timegrid)
		V = zeros(NT, Na, MC.states) # Vtij = V(t, a[i], z[j])
		consumption_pol = zeros(NT, Na, MC.states)
		for t in 1:NT # initial guess is consuming whole income + return on savings
			for i in 1:Na
				c =  r * a[i] .+ x_t[t] .* MC.state_values[:]
				consumption_pol[NT, i, :] = c
				V[t, i, :] = utility.(c; γ)
			end
		end
		savings_pol = round.(Int, ones((NT, Na, MC.states))) # savings policy fuction (in indices)
		return new(V, consumption_pol, savings_pol)
	end
end

function utility(c; γ = 2)
	return((c^(1 - γ))/(1 - γ))
end

function Diff_utility(c; γ = 2)
	return(c^(-γ))
end

function Diff_utility_inverse(u; γ = 2)
	return(u^(-1/γ))
end

function OnePeriodUpdate!(t, da, dt, IncomeTransition, NaJ, VP::ValueAndPolicy, M::Model)
	@unpack r, ρ, γ, T, Na, NT, a, timegrid, MC = M
	
	# from here on depend on t
	# have V in last period (same as initial guess)
	# will to iteration to find second to last period
	#t = NT-1
	x_t = x(timegrid[t])
	δ_t = δ(timegrid[t])
	nextperiodV = reshape(copy(VP.V[t+1, :, :]), (NaJ,1))
	oldV = copy(VP.V[t, :, :]) # get initial guess for current period

	# Here will start iteration on l (VFI)
	maxiter = 10
	tolerance = 10e-5
	iter = 0
	ε = 1.0
	# VFI
	V_forward = similar(oldV)
	V_backward = similar(oldV)
	c_central = similar(oldV)
	s_central = zeros(size(c_central))
	s_forward = similar(c_central)
	s_backward = similar(c_central)
	
	while iter < maxiter && ε > tolerance

		# Need forward and backward difference approx of V'
		V_forward[1:(Na - 1), :] = (oldV[2:Na, :] .- oldV[1:(Na - 1), :]) ./ da
		V_backward[2:Na, :] = (oldV[2:Na, :] .- oldV[1:(Na-1), :]) ./ da
		# Treat savings as 0 in the boundaries
		V_forward[Na, :] = Diff_utility.(r * a[Na] .+ x_t .* MC.state_values; γ)
		V_backward[1, :] = Diff_utility.(r * a[1] .+ x_t .* MC.state_values; γ)
		# Now calculate corresponding forward and backward consumption and savings
		# From FOC, c = inverse of u' evaluated at V'
		c_forward = Diff_utility_inverse.(V_forward; γ)
		c_backward = Diff_utility_inverse.(V_backward; γ)
	
		for i in 1:Na
			c_central[i, :] = @. x_t * MC.state_values + r * a[i]
			s_forward[i, :] = @. x_t * MC.state_values + r * a[i] - c_forward[i, :]
			s_backward[i, :] = @. x_t * MC.state_values + r * a[i] - c_backward[i, :]
		end
		V_central = Diff_utility.(c_central; γ)
		
		Ind_forward = s_forward .> 0
		Ind_backward = s_backward .< 0
		Ind_central = @. (1 - Ind_forward - Ind_backward)
		Vprime = Ind_forward .* V_forward + Ind_backward .* V_backward + Ind_central .* V_central
		#current_consumption =  Ind_forward .* c_forward + Ind_backward .* c_backward + Ind_central .* c_central
		current_consumption = Diff_utility_inverse.(Vprime)
		current_utility = reshape(utility.(current_consumption; γ), (NaJ,1))

		lowerdiag = [zeros(1, MC.states) ; @. - min(s_backward[2:Na, :], 0) / da]
		lowerdiag = reshape(lowerdiag[2:NaJ], (NaJ - 1,))
	
		upperdiag = [@. max(s_forward[1:Na-1, :], 0) / da ; zeros(1, MC.states)]
		upperdiag = reshape(upperdiag[1:NaJ - 1], (NaJ - 1,))
	
		diag = @. (min(s_backward, 0) -  max(s_forward, 0)) / da
		diag = reshape(diag, (NaJ,))
		A = spdiagm(-1 => lowerdiag, 0 => diag, 1 => upperdiag)
		TransitionMatrix = A + IncomeTransition

		b = current_utility .+ (1 / dt) .* nextperiodV
		B  = ( (ρ + δ_t + 1 / dt) .* sparse(I(NaJ)) - TransitionMatrix)
		newV = reshape(B \ b, Na, MC.states)
		
		ε = maximum(abs.(newV - oldV))
		oldV = newV
		if iter % 1 == 0
			veps = round(ε, digits = 6)
			println("Iteration: $iter; Error $veps")
		end
		iter +=1
	end
	if iter == maxiter && ε > tolerance
		println("Failed to converge :(")
	else
		VP.V[t, :, :] = oldV
		println("Converged for period $t !")
	end
end

function BackwardInduction!(VP::ValueAndPolicy, M::Model)
	@unpack r, ρ, γ, T, Na, NT, a, timegrid, MC = M
	
	da = step(a)
	dt = step(timegrid)
	IncomeTransition = sparse(kron( (MC.transition - I(MC.states)), I(Na))) # Kronecker product
	NaJ = MC.states * Na
	
	for t in (NT-1):-1:1
		println("\nPeriod $t out of $NT")
		OnePeriodUpdate!(t, da, dt, IncomeTransition, NaJ, VP, M)
	end
end
#------------------------------------------------------------------------------
# Change parameters
r = 0.025
T = 40
ρ = 0.03
γ = 2

# Create model objects
MC = MarkovChain()
M = Model(MC; ρ = ρ, r = r, γ = γ)
VP = ValueAndPolicy(M)
BackwardInduction!(VP, M)
#OnePeriodUpdate!(t, da, dt, IncomeTransitionVP, M)
