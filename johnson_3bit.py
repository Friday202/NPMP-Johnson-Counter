from models import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt
    

"""
    TESTING
"""
clock_enable = 0  #clock enable - 1 is on, 0 is off
set_number = 2

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

# 10 generations, 50 pop - genetic optimized
# {'alpha1': 33.17535379460892, 
# 'alpha2': 47.08900464951819,
# 'alpha3': 32.29413516074225,
# 'alpha4': 47.99465829689244,
# 'delta1': 2.4434824725739523,
# 'delta2': 2.3416351524918966,
# 'Kd': 4.791792719700191,
# 'n': 3.420550360325404}

# 50 generations, 80 pop - genetic optimized
# alpha1 = 35.56041696860184
# alpha2 = 45.58394396448937 
# alpha3 = 37.46739964697278 
# alpha4 = 37.70026405657447
# delta1 = 1.9270183139912507
# delta2 = 2.2806780207649853 
# Kd = 8.293142441550676
# n = 3.566363109972069

inhibition_rate = 200

params_ff = (alpha1, alpha2, alpha3, alpha4, delta1, delta2, Kd, n)


# three-bit counter with external clock
# a1, not_a1, q1, not_q1, a2, not_a2, q2, not_q2, a3, not_a3, q3, not_q3
# Y0 = np.array([0]*12) # initial state
Y0 = np.array([0]*12) # initial state
T = np.linspace(0, t_end, N) # vector of timesteps

# numerical interation
Y = odeint(counter_model_3, Y0, T, args=(params_ff,clock_enable*inhibition_rate,set_number))

Y_reshaped = np.split(Y, Y.shape[1], 1)

# plotting the results
Q1 = Y_reshaped[2]
not_Q1 = Y_reshaped[3]
Q2 = Y_reshaped[6]
not_Q2 = Y_reshaped[7]
Q3 = Y_reshaped[10]
not_Q3 = Y_reshaped[11]


plt.plot(T, Q1, label='q1')
plt.plot(T, Q2, label='q2')
plt.plot(T, Q3, label='q3')
#plt.plot(T, not_Q1, label='not q1')
#plt.plot(T, not_Q2, label='not q2')

plt.plot(T, modulated_clock(T, clock_enable*inhibition_rate, Kd, n),  '--', linewidth=2, label="CLK", color='black', alpha=0.25)

plt.legend()
plt.show()
