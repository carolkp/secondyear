#---------------------------------------------------------
# ECO 522: Advanced Macro 2
# Carolina Piazza and Christine Blandhol
# Firm Dynamics and Aggregate Productivity
# Luttmer (2007) with taxes

#---------------------------------------------------------


using Roots
using Plots


#---------------------------------------------------------

# free entry gives exit threshold (value that zeroes the function)
function free_entry_condition(exit_threshold, tax_entry, tax_fixed)
	lhs = (1 + tax_entry) * cost_entry
	constant = ((1 + tax_fixed) * cost_fixed / (interest_rate * (1 + rho)) )
	rhs = constant * ( exp( rho * (exit_threshold - entry_threshold)) + rho * exp(entry_threshold - exit_threshold) - 1 - rho )
	return lhs - rhs
end


# boundary condition gives wages (as function of exit_threshold)
function boundary_condition(exit_threshold, tax_fixed)
	numerator = exp(exit_threshold) * (1 - theta) * interest_rate * (1 + rho)
	denominator = (interest_rate - mu - (sigma^2)/2) * (1 + tax_fixed) * cost_fixed
	return (numerator / denominator)^(1 - theta)
end

function calculate_entrants(exit_threshold, wage)
	numerator = labor_supply * mu
	denominator = cost_entry + mu + (exit_threshold - entry_threshold) * cost_fixed + theta * delta * (wage^(1/(theta - 1))) * (exp(entry_threshold) - exp(exit_threshold)) / (1 - delta)

	return numerator / denominator
end


function calculate_mass_firms(exit_threshold, entrants)
	return entrants * (exit_threshold - entry_threshold)/ mu
end


function calculate_aggregate_output(exit_threshold, wage, entrants)
	numerator = delta * entrants * (wage^(theta/(theta - 1))) * (exp(entry_threshold) - exp(exit_threshold))
	denominator = mu * (1 - delta)
	return numerator / denominator
end



#---------------------------------------------------------
## Parameters

# Model
mu = -1 # drift of Brownian 
sigma = 1 # volatility of Brownian
labor_supply = 1 # L
cost_fixed = 1 # c^f
cost_entry = 1 # c^e
theta = 0.5 
interest_rate = 0.1 # r
entry_threshold = 1 # a
delta = - 2 * mu / sigma^2
rho = (1/sigma^2) * (mu + sqrt(mu^2 + 2 * interest_rate * sigma^2))

# Numerical
gridsize =  100


#---------------------------------------------------------
# First item
#  plot the exit threshold, entry, the mass of firms, the wage, and total output 

tax_entry = 0
tax_fixed_grid = range(- 0.2, stop = 0.2, length = gridsize)


exit_threshold_vec = zeros(gridsize)
mass_firms_vec = zeros(gridsize)
entry_vec = zeros(gridsize)
wage_vec = zeros(gridsize)
output_vec = zeros(gridsize)



for n in 1:gridsize
	tax_fixed = tax_fixed_grid[n]
	println("tau_f:", tax_fixed)

	f = (x -> free_entry_condition(x, tax_entry, tax_fixed))

	b = find_zero(f, 0)
	println("exit: ", b)
	
	w = boundary_condition(b, tax_fixed)
	println("wage: ", w)
	
	E = calculate_entrants(b, w)
	println("E: ", E)

	exit_threshold_vec[n] = b
	wage_vec[n] = w
	entry_vec[n] = E
	mass_firms_vec[n] = calculate_mass_firms(b, E)
	output_vec[n] = calculate_aggregate_output(b, w, E)
end

plot(tax_fixed_grid, exit_threshold_vec, title = "Exit Threshold", lw = 1.5, xlabel = "\\tau^f", legend = false)
png("pset1-figures/a-threshold")

plot(tax_fixed_grid, mass_firms_vec, title = "Mass of Firms", lw = 1.5, xlabel = "\\tau^f", legend = false)
png("pset1-figures/a-firm_mass")

plot(tax_fixed_grid, entry_vec, title = "Entry", lw = 1.5, xlabel = "\\tau^f", legend = false)
png("pset1-figures/a-entry")

plot(tax_fixed_grid, wage_vec, title = "Wage", lw = 1.5, xlabel = "\\tau^f", legend = false)
png("pset1-figures/a-wage")

plot(tax_fixed_grid, output_vec, title = "Output", lw = 1.5, xlabel = "\\tau^f", legend = false)
png("pset1-figures/a-output")


#---------------------------------------------------------
# Second item
#  plot the exit threshold, entry, the mass of firms, the wage, and total output 

tax_entry_grid = range(- 0.2, stop = 0.2, length = gridsize)
tax_fixed = 0

exit_threshold_vec = zeros(gridsize)
mass_firms_vec = zeros(gridsize)
entry_vec = zeros(gridsize)
wage_vec = zeros(gridsize)
output_vec = zeros(gridsize)

for n in 1:gridsize
	tax_entry = tax_entry_grid[n]
	f = (x -> free_entry_condition(x, tax_entry, tax_fixed))
	b = find_zero(f, 0)
	w = boundary_condition(b, tax_fixed)
	E = calculate_entrants(b, w)

	exit_threshold_vec[n] = b
	wage_vec[n] = w
	entry_vec[n] = E
	mass_firms_vec[n] = calculate_mass_firms(b, E)
	output_vec[n] = calculate_aggregate_output(b, w, E)
end

plot(tax_entry_grid, exit_threshold_vec, title = "Exit Threshold", lw = 1.5, xlabel = "\\tau^e", legend = false)
png("pset1-figures/b-threshold")

plot(tax_entry_grid, mass_firms_vec, title = "Mass of Firms", lw = 1.5, xlabel = "\\tau^e", legend = false)
png("pset1-figures/b-firm_mass")

plot(tax_entry_grid, entry_vec, title = "Entry", lw = 1.5, xlabel = "\\tau^e", legend = false)
png("pset1-figures/b-entry")

plot(tax_entry_grid, wage_vec, title = "Wage", lw = 1.5, xlabel = "\\tau^e", legend = false)
png("pset1-figures/b-wage")

plot(tax_entry_grid, output_vec, title = "Output", lw = 1.5, xlabel = "\\tau^e", legend = false)
png("pset1-figures/b-output")
