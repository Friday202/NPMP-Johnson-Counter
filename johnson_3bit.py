from models import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt
    

"""
    TESTING
"""

SET_NUMBER = 5 #from 0-7
COUNT_BACKWARDS = True
CLOCK_ENABLED = True

# get binary representation of number
def get_binary(number):
    binary_str = bin(number)[2:].zfill(3)
    return [int(i) for i in binary_str]

q_values = get_binary(SET_NUMBER)
# zakaj a-je množimo s 100? Dela tudi z 10, z 1 pa ne. 
INITIAL_STATE = [100*q_values[2], 0, q_values[2], 0, 100*q_values[1], 0, q_values[1], 0, 100*q_values[0], 0, q_values[0], 0]

# simulation parameters
t_end = 200
N = 1000


# model parameters
alpha1 = 34.73 # protein_production
alpha2 = 49.36 # protein_production
alpha3 = 32.73 # protein_production
alpha4 = 49.54 # protein_production
delta1 = 1.93 # protein_degradation
delta2 = 0.69 # protein_degradation
Kd = 10.44 # Kd
n = 4.35 # hill

params_ff = (alpha1, alpha2, alpha3, alpha4, delta1, delta2, Kd, n)

# stevec obogoatim z kontrolno logiko (clock enable, UP/DOWN - DWON = 0 steje navzgor oz navzdol,SET - zunanji) + analiza + opis
# optimizacija z hevristikami - postavimo kristerisjko funkcijo npr. čim večjo razliko med sstnanjem 0 in 1 z genetskimi algoritmi - iskanje optimalne rešitve
# 2. vse rešitve kjer sistem dela dobro - izhajamo iz članka - dobimo območje parametrov kjer nek sistem dela dobro
# programski stevec v moddel procesorja

# three-bit counter with external clock
# a1, not_a1, q1, not_q1, a2, not_a2, q2, not_q2, a3, not_a3, q3, not_q3

# namesto ničel, vzačetka nastavimo številko, od katere naprej želimo šteti
Y0 = np.array(INITIAL_STATE) # initial state
T = np.linspace(0, t_end, N) # vector of timesteps

# numerical interation
# First plot for three_bit_model
Y1 = odeint(three_bit_model, Y0, T, args=(params_ff,))
Y1_reshaped = np.split(Y1, Y1.shape[1], 1)

Q1_1 = Y1_reshaped[2]
Q2_1 = Y1_reshaped[6]
Q3_1 = Y1_reshaped[10]

# Second plot for counter_model_3
# novi external argumenti za naš counter
Y2 = odeint(counter_model_3, Y0, T, args=(params_ff,CLOCK_ENABLED,COUNT_BACKWARDS,))
Y2_reshaped = np.split(Y2, Y2.shape[1], 1)

Q1_2 = Y2_reshaped[2]
Q2_2 = Y2_reshaped[6]
Q3_2 = Y2_reshaped[10]

# Plotting
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(T, Q1_1, label='q1 - three_bit_model')
plt.plot(T, Q2_1, label='q2 - three_bit_model')
plt.plot(T, Q3_1, label='q3 - three_bit_model')
plt.plot(T, get_clock(T), '--', linewidth=2, label="CLK", color='black', alpha=0.25)
plt.legend()
plt.title('Plot for three_bit_model')

plt.subplot(2, 1, 2)
plt.plot(T, Q1_2, label='q1 - counter_model_3')
plt.plot(T, Q2_2, label='q2 - counter_model_3')
plt.plot(T, Q3_2, label='q3 - counter_model_3')
plt.plot(T, get_clock(T), '--', linewidth=2, label="CLK", color='black', alpha=0.25)
plt.legend()
plt.title('Plot for counter_model_3')

plt.tight_layout()
plt.show()
