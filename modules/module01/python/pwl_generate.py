import numpy as np
import matplotlib.pyplot as plt
import csv

from random import seed
from random import random

def generate_sine(tsim, npoints, filename):
    x = []
    y = []
    x = np.linspace(0, tsim, npoints)
    y = np.sin(x)
    
    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(x,y))
    
    return x, y


times, values = generate_sine(10, 1000, '1_rad_sinewave.txt')
    
fig, ax = plt.subplots()
ax.plot(times, values)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Voltage [V]')
ax.grid()
plt.show()

def generate_noise(mu, sigma, tsim, npoints, filename):
    x = []
    y = []
    x = np.linspace(0, tsim, npoints)
    y = np.random.normal(mu, sigma, npoints)
    
    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(x,y))
    
    return x, y


times, values = generate_noise(0, 1, 10, 1001, 'pseudorandom_noise.txt')
print(np.std(values))
print(times)
print(values)
    
fig, ax = plt.subplots()
ax.plot(times, values)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Voltage [V]')
ax.grid()
plt.show()