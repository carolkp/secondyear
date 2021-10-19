# Carolina Kowalski Piazza
# ECO 528 - Macroeconomic Perspectives on Inequality
# Assignment 2: Solve the Life-Cycle model
#

cd(dirname(@__FILE__))

using UnPack, LinearAlgebra, SparseArrays
using Plots


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
 
    function Model(MC::MarkovChain; ρ = 0.03, r = 0.025, γ = 2, T = 40, Na = 50, NT = 240)
	amin = 0
	amax = 10
	a = range(amin, stop = amax, length = Na)
	timegrid = range(0, stop = T, length = NT)
	return new(r, ρ, γ, T, Na, NT, a, timegrid, MC)
    end
end


mutable struct ValueAndPolicy
	V::Array{Float64,3} # Value function
	consumption_pol::Array{Float64, 3} # consumption policy
	savings_pol::Array{Int64,3} # savings policy function
	wealth_distribution::Array{Float64, 3}
	IncomeTransition::Array{Float64,2}

	function ValueAndPolicy(M::Model)
		@unpack r, ρ, γ, T, Na, NT, a, timegrid, MC = M
		dt = step(timegrid)
		x_t = x.(timegrid)
		V = zeros(NT, Na, MC.states) # Vtij = V(t, a[i], z[j])a
		consumption_pol = zeros(NT, Na, MC.states)
		wealth_distribution = zeros(NT, Na, MC.states)
		for t in 1:NT # initial guess is consuming whole income + return on savings
			for i in 1:Na
				if t < NT
					c =  r * a[i] .+ x_t[t] .* MC.state_values[:]
				else
					c = a[i] .+ x_t[t] .* MC.state_values[:]
				end
				if t > 1
					wealth_distribution[t, Na, :] = repeat([1/(Na*MC.states)], outer = MC.states)
				end
				consumption_pol[t, i, :] = c
				V[t, i, :] = utility.(c; γ)
			end
		end
		wealth_distribution[1, 1, :] =  FindStationaryDistribution(MC.transition')
		savings_pol = round.(Int, ones((NT, Na, MC.states))) # savings policy fuction (in indices)
		IncomeTransition = sparse(kron( (MC.transition - I(MC.states)), I(Na))) # Kronecker product
		return new(V, consumption_pol, savings_pol, wealth_distribution, IncomeTransition)

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

function FindStationaryDistribution(Π)
	λ = eigvals(Π)
	idx = Int(findfirst(x -> abs(x - 1) < 1e-6, λ))
	v = eigvecs(Π)
	stationary = v[:, idx] ./ sum(v[:, idx])
	return(stationary)

end

function GetA(oldV, da, NaJ, x_t, M::Model)
	@unpack r, ρ, γ, T, Na, NT, a, timegrid, MC = M

	V_forward = similar(oldV)
	V_backward = similar(oldV)
	c_central = similar(oldV)
	s_central = zeros(size(c_central))
	s_forward = similar(c_central)
	s_backward = similar(c_central)
	
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
	
	return(A, current_utility, current_consumption)
end

function OnePeriodUpdate!(t, da, dt, NaJ, VP::ValueAndPolicy, M::Model, δ)
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

	while iter < maxiter && ε > tolerance
		
		A, current_utility, current_consumption = GetA(oldV, da, NaJ, x_t, M)
		TransitionMatrix = A + VP.IncomeTransition

		b = current_utility .+ (1 / dt) .* nextperiodV
		B  = ( (ρ + δ_t + 1 / dt) .* sparse(I(NaJ)) - TransitionMatrix)
		newV = reshape(B \ b, Na, MC.states)
		
		ε = maximum(abs.(newV - oldV))
		oldV = newV
		if iter % 2 == 0
			veps = round(ε, digits = 6)
			println("Iteration: $iter; Error $veps")
		end
		iter +=1
		VP.consumption_pol[t, :, :] = current_consumption
	end
	if iter == maxiter && ε > tolerance
		println("Failed to converge :(")
	else
		VP.V[t, :, :] = oldV
		println("Converged for period $t !")
	end
end

function BackwardInduction!(VP::ValueAndPolicy, M::Model, δ::Function)
	@unpack r, ρ, γ, T, Na, NT, a, timegrid, MC = M
	
	da = step(a)
	dt = step(timegrid)
	NaJ = MC.states * Na
	
	for t in (NT-1):-1:1
		println("\nPeriod $t out of $NT")
		OnePeriodUpdate!(t, da, dt, NaJ, VP, M, δ)
	end
end

function KFE!(VP::ValueAndPolicy, M::Model)
	@unpack r, ρ, γ, T, Na, NT, a, timegrid, MC = M
	# Use expession for wealth distribution from last slide
	# (1 / dt) g(t+1, ) - g(t, ) = (A' - lambda(t)I)g(t, )
	# where g is the wealth distribution, A is A + IncomeTransition and lambda is the death rate (δ here)
	# We set stationary distribution for z at period 1 and everyone with 0 assets
	da = step(a)
	NaJ = MC.states * Na
	g = zeros(NaJ,1)
	dt = step(timegrid)

	for t in 2:NT
		v = VP.V[t, :, :]
		x_t = x(timegrid[t])
		δ_t = δ(t)
		A, _, __ =  GetA(v, da, NaJ, x_t, M::Model)
		TransitionMatrix = A + VP.IncomeTransition
		previousdistribution = reshape(VP.wealth_distribution[t - 1, :, :], NaJ, 1)
		
		# iter = 0
		# ε = 1.0
		# maxiter = 1000
		# olddist = reshape(VP.wealth_distribution[t, :, :], NaJ, 1) 
		# while ε > 10e-3 || iter < maxiter
		# 	newdist = (I - TransitionMatrix') \ olddist	
		# 	#nextdistribution = ((1 - δ_t) .* I + dt .* TransitionMatrix') * previousdistribution # with this, total weath decreases over time so need to renormalize
		# 	iter += 1
		# 	ε = maximum(abs.(newdist - olddist))
		# 	print("ε = $ε")
		# 	newdist = olddist
		# end
		nextdistribution = ( I + dt .* TransitionMatrix') * previousdistribution
		#nextdistribution = nextdistribution ./ nextdistribution'*ones(NaJ, 1) # normalize to sum to 1 because total wealth is shrinking when people die
		VP.wealth_distribution[t, :, :] = reshape(nextdistribution, Na, MC.states)
	end
end

function FindConsumptionSummaryStats(VP::ValueAndPolicy, M::Model)
	@unpack Na, NT, MC = M
	logconsumption_mean = zeros(NT)
	logconsumption_variance = zeros(NT)
	for t in 1:M.NT
		consumption = log.(reshape(VP.consumption_pol[t, :, :], (Na * MC.states,) ))
		probabs = reshape(VP.wealth_distribution[t, :, :], (Na * MC.states,))
		logconsumption_mean[t] = dot(consumption, probabs)
		logconsumption_variance[t] = dot( (consumption .- logconsumption_mean[t]).^2 , probabs )
	end
	return(logconsumption_mean, logconsumption_variance)
end
#------------------------------------------------------------------------------
# Change parameters
r = 0.025
T = 40
ρ = 0.03
γ = 2

# Item a) Assess results a little bit

 NT = 120
 MC = MarkovChain()
 M = Model(MC; NT = NT)
 VP = ValueAndPolicy(M)
 BackwardInduction!(VP, M, δ)
 
 plot(M.timegrid, VP.V[:, 1:5:50, 2], legend = false, title = "Value function for z2 at various savings levels", lw = 1.5, xlabel = "time", ylabel = "Value", titlefont=font(12))
 png("figures/a-valuefunctionNT120.png")

NT = 400
MC = MarkovChain()
M = Model(MC; NT = NT)
VP = ValueAndPolicy(M)
BackwardInduction!(VP, M, δ)

plot(VP.V[:, 1:5:50, 2], legend = false, title = "Value function for z2 at various savings levels", lw = 1.5, xlabel = "time", ylabel = "Value", titlefont=font(12))
png("figures/a-valuefunctionNT400.png")

# Now plot consumption decision for middle income
# item b)
# Stay with base model with NT = 240
MC = MarkovChain()
M = Model(MC; NT = 240)
BackwardInduction!(VP, M, δ)

ages = [5, 25, 40]
ids = Int.(ones(length(ages)))
for (i, age) in enumerate(ages)
	ids[i] = Int(findfirst(x -> x >= age, M.timegrid))
end

plot(M.a[1:25], VP.consumption_pol[ids, 1:25, 2]', legend = :topleft, labels = ["Age 5" "Age 25" "Age 40"], title = "Consumption for middle income individuals", lw = 1.5, xlabel = "savings a", ylabel = "consumption", titlefont = font(12) )
png("figures/b-consumptionrule_middleincome.png")

# item c)
# No death rate
δ0(t) = 0

VP2 = ValueAndPolicy(M)
BackwardInduction!(VP2, M ,δ0)

plot(M.a[1:25], [VP.consumption_pol[ids[2], 1:25, 2] VP2.consumption_pol[ids[2], 1:25, 2]], title = "Consumption rule with and without death rate for middle income", lw = 1.5, xlabel = "savings a", ylabel = "consumption", titlefont = font(12), legend = :topleft, labels = ["baseline death rate" "No death rate"])
png("figures/c-consumptionrule_withoutdeath.png")

# item e)
# Find distribution using KFE
KFE!(VP, M)

# plot mean a for each level of income over life cycle
mean_assets1 = [dot(M.a, VP.wealth_distribution[t, :, 1]) for t in 1:M.NT]
mean_assets2 = [dot(M.a, VP.wealth_distribution[t, :, 2]) for t in 1:M.NT]
mean_assets3 = [dot(M.a, VP.wealth_distribution[t, :, 3]) for t in 1:M.NT]


plot(M.timegrid, mean_assets1, title = "Mean of asset holdings", label="low", legendtitle = "income",  lw = 1.5, xlabel = "age", ylabel = "mean savings", titlefont = font(12))
plot!(M.timegrid, mean_assets2, lw = 1.5, label = "middle")
plot!(M.timegrid, mean_assets3, lw = 1.5, label = "high")
png("figures/e-avgassetbyincome.png")

# plot distribution of assets for middle income at ages 5, 25, 40

data = VP.wealth_distribution[ids, 1:15, 2]
for i in 1:3
	data[i, :] = data[i, :] ./ sum(data[i, :])
end

plot(M.a[1:15], data', legendtitle = "Age", labels = ["5" "25" "40"], lw = 1.5, title = "Wealth distribution for middle income")
png("figures/e-stationarydist_middleincome.png")

# item d)
# We can just use the stationary distribution now


logconsumption_mean_base, logconsumption_variance = FindConsumptionSummaryStats(VP, M)
# Now use constant income process
MC2 = MarkovChain(; state_values = [1, 1, 1]) 
M3 = Model(MC2)
VP3 = ValueAndPolicy(M3)
BackwardInduction!(VP3, M3, δ)
KFE!(VP3, M3)

logconsumption_mean_norisk, _ = FindConsumptionSummaryStats(VP3, M3)

# plot(M.timegrid, [logconsumption_mean_base, logconsumption_mean_norisk], lw = 1.5, labels = ["Baseline mean"  "Constant z mean"], xlabel = "age", titlefont = font(12), title = "Mean and Variance of Consumption")

plt = plot(title="Log Consumption Mean and Variance", legend=:bottomright, xlabel="Age", titlefont = font(12), right_margin = 20Plots.mm);
plot!(plt, M.timegrid, [logconsumption_mean_base, logconsumption_mean_norisk, NaN.*(1:M.NT)], labels = ["Baseline mean" "Constant z mean" "Baseline Variance"], lw = 1.5, ylabel = "Mean of log(c)");
plot!(twinx(plt), M.timegrid, logconsumption_variance, legend = false, color = "green", lw = 1.5, ylabel = "Variance of log(c)")
png("figures/d-meanandvarconsumption.png")

