#---------------------------------------------------------
# ECO 522: Advanced Macro 2
# Carolina Piazza and Christine Blandhol
# Firm Dynamics and Aggregate Productivity
# Luttmer (2007) with taxes

#---------------------------------------------------------




#---------------------------------------------------------
## Parameters

# Model
mu = -1
sigma = 1
L = 1
cost_fixed = 1
cost_entry = 1
theta = 0.5
r = 0.1
a = 1

# Numerical
gridsize =  100


#---------------------------------------------------------
# First item

tau_entry = 0
tau_fixed = range(-0.2, stop = 0.2, length = gridsize)

