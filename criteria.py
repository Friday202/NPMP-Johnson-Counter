from models import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from criteria_helpers import fitness_function, fitness_function_calc, get_ideal_response, run_counter

# Set bounds for the parameters
# Original parameter values
original_params = [34.73, 49.36, 32.73, 49.54, 1.93, 0.69, 10.44, 4.35]

# Set bounds close to original values
bounds_factor = 0.62  # Adjust this factor to determine how close the bounds are to the original values
bounds = [(param - bounds_factor * param, param + bounds_factor * param) for param in original_params]
# bounds = [(0.1, 50), (0.1, 50), (0.1, 50), (0.1, 50), (0.001, 50), (0.001, 50), (0.01, 250), (1, 5)]

# Run optimization
result = minimize(fitness_function, x0=np.random.rand(8), bounds=bounds)

# Get the optimized parameters
optimized_params = result.x

# Run the model with the optimized parameters
Q1_opt, Q2_opt, Q3_opt, clk_opt, T_opt = run_counter(*optimized_params)

# Plot the results
plt.plot(T_opt, Q1_opt, label='q1_opt')
plt.plot(T_opt, Q2_opt, label='q2_opt')
plt.plot(T_opt, Q3_opt, label='q3_opt')
plt.plot(T_opt, clk_opt, '--', linewidth=2, label="CLK_opt", color='black', alpha=0.25)
plt.legend()
plt.show()

print("Optimized parameters:", optimized_params)

# run model with original parameters
Q1_orig, Q2_orig, Q3_orig, clk_orig, T_orig = run_counter(*original_params)

# Plot the results
plt.plot(T_orig, Q1_orig, label='q1_orig')
plt.plot(T_orig, Q2_orig, label='q2_orig')
plt.plot(T_orig, Q3_orig, label='q3_orig')
plt.plot(T_opt, clk_opt, '--', linewidth=2, label="CLK_opt", color='black', alpha=0.25)
plt.legend()
plt.show()