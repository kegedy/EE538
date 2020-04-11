import os
import sys
import cmath
import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import signal

""" Plot V-I relationship for an ideal diode """
Ioff = 0
Ion = 100
Von = 0.6
# V = np.linspace(0, 1, num=100, endpoint=True) # linear sweep of voltage
# I = np.piecewise(V,[V < Von, V == Von, V > Von], [0, Ion, 0])


fig, ax = plt.subplots()
# ax.plot(V, 1e3*I)
ax.set_ylabel('$I_D$ [mA]')
ax.set_xlabel('$V_{D}$ [V]')
ax.set_title('Current vs Voltage for an Ideal Diode')
ax.grid()
ax.hlines(Ioff, 0, Von, color='tab:blue')
ax.vlines(Von, 0, Ion, color='tab:blue')
textstr = r'$V_{on}=%.1f$' % (Von, )
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize =14,
            verticalalignment='top', bbox=props)
plt.show()
fig.clear() 

""" Plot I-V relationship for a BJT using the Ebers Moll model """
Is = 1e-15
k = 1.38e-23
T = 300
q = 1.6e-19
V_T = k*T/q
V_be = np.linspace(0.1,0.8,num=70)
Ic = Is*(np.exp(V_be/V_T) - 1) 

fig, ax = plt.subplots()
ax.plot(V_be, 1e3*Ic)
ax.set_ylabel('$I_C$ [mA]')
ax.set_xlabel('$V_{BE}$ [V]')
ax.set_title('Current vs Voltage for an BJT')
ax.grid()
plt.show()