import numpy as np
from deap import base, creator, tools, algorithms
from functools import partial
from scipy.integrate import odeint
from scipy.fft import fft
import matplotlib.pyplot as plt
from params import param_ranges
from models import counter_model_3, modulated_clock
from criteria_helpers import run_counter, fitness_function_calc, mean_squared_error, get_ideal_response

def generate_random_params():
    # Original parameter values
    original_params = [34.73, 49.36, 32.73, 49.54, 1.93, 0.69, 10.44, 4.35]
    param_names = ['alpha1', 'alpha2', 'alpha3', 'alpha4', 'delta1', 'delta2', 'Kd', 'n']
    # Set bounds close to original values
    bounds_factor = 0.62
    bounds = [(param - bounds_factor * param, param + bounds_factor * param) for param in original_params]
    params = {param_names[i]: np.random.uniform(lower, upper) for i, (lower, upper) in enumerate(bounds)}

    return params

from sklearn.metrics import mean_squared_error
def fitness_function_calc(normalized_Q1, ideal_response_q1, normalized_Q2, ideal_response_q2, normalized_Q3, ideal_response_q3):
    # Calculate Mean Squared Error for all three outputs, if one signal has significantly higher error, it will be penalized
    mse_q1 = mean_squared_error(normalized_Q1, ideal_response_q1)
    mse_q2 = mean_squared_error(normalized_Q2, ideal_response_q2)
    mse_q3 = mean_squared_error(normalized_Q3, ideal_response_q3)

    # Calculate the fitness
    fitness = mse_q1**2 + mse_q2**2 + mse_q3**2
    return -fitness


def objective_function(individual, inhibition_rate=200, set_number=2):
    # Extract parameters from individual
    alpha1 = individual["alpha1"]
    alpha2 = individual["alpha2"]
    alpha3 = individual["alpha3"]
    alpha4 = individual["alpha4"]
    delta1 = individual["delta1"]
    delta2 = individual["delta2"]
    Kd = individual["Kd"]
    n = individual["n"]
    # Run the counter model to get the simulated signals
    Q1_sim, Q2_sim, Q3_sim, clk_sim, T_sim = run_counter(alpha1, alpha2, alpha3, alpha4, delta1, delta2, Kd, n)
    # Calculate fitness
    fitness = fitness_function_calc(Q1_sim, get_ideal_response([0, 1], T_sim, clk_sim, 1, 0.01),
                                    Q2_sim, get_ideal_response([1, 1, 0, 0], T_sim, clk_sim, 1, 0.01),
                                    Q3_sim, get_ideal_response([0, 0, 1, 1, 1, 1, 0, 0], T_sim, clk_sim, 1, 0.01))

    return fitness,

def custom_cxBlend(ind1, ind2, alpha=0.5):
    """Blends two individuals with a given alpha."""
    for key in ind1.keys():
        ind1[key], ind2[key] = (1. - alpha) * ind1[key] + alpha * ind2[key], (1. - alpha) * ind2[key] + alpha * ind1[key]
    return ind1, ind2
import random
def custom_mutGaussian(individual, mu, sigma, indpb):
    """Applies Gaussian mutation to each parameter in the individual."""
    for key in individual.keys():
        if random.random() < indpb:
            individual[key] += random.gauss(mu, sigma)
    return individual,

def optimize_parameters(model_function, objective_function, generations, population_size):
    print("Starting Optimization")

    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    creator.create("Individual", dict, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("params", generate_random_params)

    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.params)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", custom_cxBlend, alpha=0.5)
    toolbox.register("mutate", custom_mutGaussian, mu=0, sigma=1.5, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("evaluate", objective_function)

    population = toolbox.population(n=population_size)

    print("Initial Population:")
    for ind in population:
        print(ind)

    for gen in range(generations):
        print(f"\nGeneration {gen + 1}/{generations}")
        
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.8, mutpb=0.2)
        fits = toolbox.map(toolbox.evaluate, offspring)

        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        population = toolbox.select(offspring + population, k=len(population))

        # Additional print statements to monitor the population
        best_ind = tools.sortNondominated(population, len(population), first_front_only=True)[0][0]
        print(f"Best Individual in Generation {gen + 1}: {best_ind.fitness.values}")

    print("\nFinal Population:")
    for ind in population:
        print(ind)

    print("\nOptimization Completed")
    
    best_params = tools.sortNondominated(population, len(population), first_front_only=True)[0][0]
    return best_params

t_end = 200
N = 1000
inhibition_rate = 200
set_number = 2
noise_amplitude = 10

# Pass additional arguments to the objective_function using partial
opt_objective_function = partial(objective_function)

best_params = optimize_parameters(counter_model_3, objective_function, generations=10, population_size=50)
print(best_params)

# Run the model with the optimized parameters
Q1_opt, Q2_opt, Q3_opt, clk_opt, T_opt = run_counter(best_params['alpha1'], 
                                                     best_params['alpha2'], 
                                                     best_params['alpha3'], 
                                                     best_params['alpha4'],
                                                     best_params['delta1'], 
                                                     best_params['delta2'],
                                                     best_params['Kd'], 
                                                     best_params['n'])

# Plot the results
plt.plot(T_opt, Q1_opt, label='q1_opt')
plt.plot(T_opt, Q2_opt, label='q2_opt')
plt.plot(T_opt, Q3_opt, label='q3_opt')
plt.plot(T_opt, clk_opt, '--', linewidth=2, label="CLK_opt", color='black', alpha=0.25)
plt.legend()
plt.show()

# # Set bounds for the parameters
# # Original parameter values
original_params = [34.73, 49.36, 32.73, 49.54, 1.93, 0.69, 10.44, 4.35]
# run model with original parameters
Q1_orig, Q2_orig, Q3_orig, clk_orig, T_orig = run_counter(*original_params)

# Plot the results
plt.plot(T_orig, Q1_orig, label='q1_orig')
plt.plot(T_orig, Q2_orig, label='q2_orig')
plt.plot(T_orig, Q3_orig, label='q3_orig')
plt.plot(T_opt, clk_opt, '--', linewidth=2, label="CLK_opt", color='black', alpha=0.25)
plt.legend()
plt.show()