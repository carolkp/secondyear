#---------------------------------------------------------
# ECO 522: Advanced Macro 2
# Carolina Piazza and Christine Blandhol
# Firm Dynamics and Aggregate Productivity
# Luttmer (2007) with taxes

#---------------------------------------------------------

function calculate_labor(productivity, wage)
	return theta * exp(productivity) * wage^(theta/(theta-1))
end

function calculate_output(productivity, wage)
	return exp(productivity) * wage^(theta/(theta - 1))
end

function calculate_profit(productivity, wage)
	return (1 - theta) * exp(productivity) * wage^(theta/(theta - 1)) - (1 + tax_fixed) * wage * cost_fixed
end

function calculate_mass_firms(exit_threshold, entry_threshold, entrants)
	return entrants * (exit_threshold - entry_threshold)/ mu
end

function calculate_entrants(entry_threshold, exit_threshold, wage)
	numerator = labor_supply * mu
	denominator = cost_entry + mu + (exit_threshold - entry_threshold) * cost_fixed + theta * delta * (wage^(1/(theta - 1))) * (exp(entry_threshold) - exp(exit_threshold)) / (1 - delta)

	return numerator / denominator
end

function value_function(productivity, exit_threshold)
	aux1 = (1 + tax_fixed) * wage * cost_fixed / (interest_rate * (1 + rho))
	aux2 = rho * exp(productivity - exit_threshold) - 1 - rho - exp(- rho * (productivity - exit_threshold))
	return aux1 * aux2
end

function find_exit_threshold()
end

function calculate_aggregate_output
end


#---------------------------------------------------------
## Parameters

# Model
mu = -1
sigma = 1
labor_supply = 1
cost_fixed = 1
cost_entry = 1
theta = 0.5
interest_rate = 0.1
entry_threshold = 1
delta = - 2 * mu / sigma^2
rho = (1/sigma^2) * (mu + sqrt(mu^2 + 2 * interest_rate * sigma^2))

# Numerical
gridsize =  100


#---------------------------------------------------------
# First item
#  plot the exit threshold, entry, the mass of firms, the wage, and total output 

tax_entry = 0
tax_fixed_grid = range(- 0.2, stop = 0.2, length = gridsize)



mass_firms = zeros(gridsize)
entrants = zeros(gridsize)

for n in 1:gridsize
	tax_fixed = tax_fixed_grid[n]

end

