from models import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def run_counter(alpha1 = 34.73, alpha2 = 49.36 , alpha3 = 32.73, alpha4 = 49.54, delta1 = 1.93, delta2 = 0.69, Kd = 10.44, n = 4.35):
    """
        TESTING
    """
    clock_enable = 0  # clock enable - 1 is on, 0 is off
    set_number = 2

    # simulation parameters
    t_end = 200
    N = 1000

    # # model parameters
    # alpha1 = 34.73  # protein_production 0.1 to 50
    # alpha2 = 49.36  # protein_production 0.1 to 50
    # alpha3 = 32.73  # protein_production 0.1 to 50
    # alpha4 = 49.54  # protein_production 0.1 to 50
    # delta1 = 1.93  # protein_degradation 0.001 to 50
    # delta2 = 0.69  # protein_degradation 0.001 to 50
    # Kd = 10.44  # Kd 0.01 to 250
    # n = 4.35  # hill 1 to 5

    inhibition_rate = 200

    params_ff = (alpha1, alpha2, alpha3, alpha4, delta1, delta2, Kd, n)

    # three-bit counter with external clock
    # a1, not_a1, q1, not_q1, a2, not_a2, q2, not_q2, a3, not_a3, q3, not_q3
    # Y0 = np.array([0]*12) # initial state
    Y0 = np.array([0] * 12)  # initial state
    T = np.linspace(0, t_end, N)  # vector of timesteps

    # numerical interation
    Y = odeint(counter_model_3, Y0, T, args=(params_ff, clock_enable * inhibition_rate, set_number))

    Y_reshaped = np.split(Y, Y.shape[1], 1)


    Q1 = Y_reshaped[2]
    not_Q1 = Y_reshaped[3]
    Q2 = Y_reshaped[6]
    not_Q2 = Y_reshaped[7]
    Q3 = Y_reshaped[10]
    not_Q3 = Y_reshaped[11]
    clk = modulated_clock(T, clock_enable * inhibition_rate, Kd, n)

    # plotting the results for current params
    # plt.subplot(1, 2, 1)
    # plt.plot(T, Q1, label='q1')
    # plt.plot(T, Q2, label='q2')
    # plt.plot(T, Q3, label='q3')
    # plt.plot(T, clk, '--', linewidth=2, label="CLK", color='black', alpha=0.25)
    # plt.legend()

    # MY CODE:


    def normalize(np_array):
        """
        Normalizes a numpy array in range 0 to 1.

        :param np_array: np_array to normalize
        :return: normalized np_array
        """
        return (np_array - np_array.min()) / (np_array.max() - np_array.min())


    normalized_clk = normalize(clk)
    normalized_T = normalize(T)
    normalized_Q1 = normalize(Q1)
    normalized_Q2 = normalize(Q2)
    normalized_Q3 = normalize(Q3)

    # plotting normalized graph
    # plt.subplot(1, 2, 2)
    # plt.plot(normalized_T, normalized_Q1, label='q1')
    # plt.plot(normalized_T, normalized_Q2, label='q2')
    # plt.plot(normalized_T, normalized_Q3, label='q3')
    # plt.plot(normalized_T, normalized_clk, '--', linewidth=2, label="CLK", color='black', alpha=0.25)

    # plt.tight_layout()
    # plt.show()

    return normalized_Q1, normalized_Q2, normalized_Q3, normalized_clk, normalized_T

# print(normalized_T)


def get_ideal_response(period, T, clk, mode, threshold):
    """
    Function returns values for an ideal electrical response.

    :param period: user defined values to repeat ie. [0,1] or [0,0,1,1]
    :param T: timestep (x value)
    :param clk: clock (sin) value (y value)
    :param mode: trigger mode for clock 1 - rising edge, 2 - lowering edge, 3 - positive state, 4 - negative state
    :param threshold: the value at which we evaluate the output
    :return: array of ideal values of size T
    """
    # At threshold value of the clock the output should change immediately
    current_value = 0
    ideal_values = []

    prev_clk_value = clk[0]

    if mode == 1:
        # Positive edge
        for timestep in range(len(T)):
            if clk[timestep] > threshold >= prev_clk_value:
                current_value += 1
                if current_value >= len(period):
                    current_value = 0

            ideal_values.append(period[current_value])
            prev_clk_value = clk[timestep]

    elif mode == 2:
        # Negative edge
        for timestep in range(len(T)):
            if clk[timestep] <= threshold < prev_clk_value:
                current_value += 1
                if current_value >= len(period):
                    current_value = 0

            ideal_values.append(current_value)
            prev_clk_value = clk[timestep]

    # These modes can be made with rising and lowering edge

    elif mode == 3:
        for timestep in range(len(T)):
            if clk[timestep] < 0.005:
                if changed == 1:
                    current_value = not current_value
                changed = 0
            else:
                changed = 1

            ideal_values.append(current_value)

    elif mode == 4:
        for timestep in range(len(T)):
            if clk[timestep] > 0.995:
                if changed == 1:
                    current_value = not current_value
                changed = 0
            else:
                changed = 1

            ideal_values.append(current_value)

    else:
        print("Error mode is incorrect")
        return

    return np.array(ideal_values)

# # plotting normalized graph
# plt.plot(normalized_T, normalized_Q1, label='q1')
# plt.plot(normalized_T, ideal_response_q1, label='q1 ideal')
# # plt.plot(normalized_T, normalized_Q2, label='q2')
# # plt.plot(normalized_T, ideal_response_q2, label='q2 ideal')
# # plt.plot(normalized_T, normalized_Q3, label='q3')
# # plt.plot(normalized_T, ideal_response_q3, label='q3 ideal')
# plt.plot(normalized_T, normalized_clk, '--', linewidth=2, label="CLK", color='black', alpha=0.25)
# plt.show()


from sklearn.metrics import mean_squared_error
def fitness_function_calc(normalized_Q1, ideal_response_q1, normalized_Q2, ideal_response_q2, normalized_Q3, ideal_response_q3):
    # Calculate Mean Squared Error for all three outputs, if one signal has significantly higher error, it will be penalized
    mse_q1 = mean_squared_error(normalized_Q1, ideal_response_q1)
    mse_q2 = mean_squared_error(normalized_Q2, ideal_response_q2)
    mse_q3 = mean_squared_error(normalized_Q3, ideal_response_q3)

    # Calculate the fitness
    fitness = mse_q1**2 + mse_q2**2 + mse_q3**2
    return -fitness



import numpy as np
from scipy.optimize import minimize

# Define your fitness function
def fitness_function(params):
    alpha1, alpha2, alpha3, alpha4, delta1, delta2, Kd, n = params
    normalized_Q1, normalized_Q2, normalized_Q3, normalized_clk, normalized_T = run_counter(alpha1, alpha2, alpha3, alpha4, delta1, delta2, Kd, n)
    # Your fitness function here, e.g., the sum of Q3 values
    ideal_response_q1 = get_ideal_response([0, 1], normalized_T, normalized_clk, 1, 0.01)
    ideal_response_q2 = get_ideal_response([1, 1, 0, 0], normalized_T, normalized_clk, 1, 0.01)
    ideal_response_q3 = get_ideal_response([0, 0, 1, 1, 1, 1, 0, 0], normalized_T, normalized_clk, 1, 0.01)
    fitness = fitness_function_calc(normalized_Q1, ideal_response_q1, normalized_Q2, ideal_response_q2, normalized_Q3, ideal_response_q3)
    return -fitness  # Minimize, so we negate the fitness

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




