from models import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt

"""
    TESTING
"""
clock_enable = 0  # clock enable - 1 is on, 0 is off
set_number = 2

# simulation parameters
t_end = 200
N = 1000

# model parameters
alpha1 = 34.73  # protein_production 0.1 to 50
alpha2 = 49.36  # protein_production 0.1 to 50
alpha3 = 32.73  # protein_production 0.1 to 50
alpha4 = 49.54  # protein_production 0.1 to 50
delta1 = 1.93  # protein_degradation 0.001 to 50
delta2 = 0.69  # protein_degradation 0.001 to 50
Kd = 10.44  # Kd 0.01 to 250
n = 4.35  # hill 1 to 5

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
plt.subplot(1, 2, 1)
plt.plot(T, Q1, label='q1')
plt.plot(T, Q2, label='q2')
plt.plot(T, Q3, label='q3')
plt.plot(T, clk, '--', linewidth=2, label="CLK", color='black', alpha=0.25)
plt.legend()

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
plt.subplot(1, 2, 2)
plt.plot(normalized_T, normalized_Q1, label='q1')
# plt.plot(normalized_T, normalized_Q2, label='q2')
# plt.plot(normalized_T, normalized_Q3, label='q3')
plt.plot(normalized_T, normalized_clk, '--', linewidth=2, label="CLK", color='black', alpha=0.25)

plt.tight_layout()
plt.show()

print(normalized_T)


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
        for timestep in range(len(T)):
            if clk[timestep] <= threshold and prev_clk_value > threshold:
                current_value = 1 if current_value == 0 else 0

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


ideal_response_q1 = get_ideal_response([0, 1], normalized_T, normalized_clk, 1, 0.01)

# plotting normalized graph
plt.plot(normalized_T, normalized_Q1, label='q1')
plt.plot(normalized_T, ideal_response_q1, label='q1 ideal')
# plt.plot(normalized_T, normalized_Q2, label='q2')
# plt.plot(normalized_T, normalized_Q3, label='q3')
plt.plot(normalized_T, normalized_clk, '--', linewidth=2, label="CLK", color='black', alpha=0.25)


plt.show()


