import os
import sys
import cmath
import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import signal


""" Common Emitter Amplifier Nonlinearity """
Is = 1e-15
k = 1.38e-23
T = 300
q = 1.6e-19
V_T = k*T/q
R_C = 10e3
f = 1e3
w = 2*np.pi*f
t = np.linspace(0, 3e-3, num=100)
V_BE = 0.7
V_in_1mV = 1e-3*np.sin(w*t)+V_BE
V_in_20mV = 20e-3*np.sin(w*t)+V_BE
Ic = Is*(np.exp(V_in_1mV/V_T) - 1) 
Ic_20mV = Is*(np.exp(V_in_20mV/V_T) - 1) 
I_DC = Is*(np.exp(V_BE/V_T) - 1) 
V_out_1mV = 10 - Ic*R_C
V_out_ideal_1mV = 10 - I_DC*R_C - (V_in_1mV - V_BE)*I_DC/V_T*R_C 
V_out_20mV = 10 - Ic_20mV*R_C
V_out_ideal_20mV = 10 - I_DC*R_C - (V_in_20mV - V_BE)*I_DC/V_T*R_C 

fig, ax = plt.subplots(2)
actual_line_1mV, = ax[0].plot(t*1e3, V_out_1mV-np.mean(V_out_1mV), color='tab:blue')
actual_line_1mV.set_label('Ebers-Moll Model')
ideal_line_1mV, = ax[0].plot(t*1e3,V_out_ideal_1mV-np.mean(V_out_ideal_1mV), color='tab:red')
ideal_line_1mV.set_label('Linear Approximation')

actual_line_20mV, = ax[1].plot(t*1e3, V_out_20mV-np.mean(V_out_1mV), color='tab:blue')
actual_line_20mV.set_label('Ebers-Moll Model')
ideal_line_20mV, = ax[1].plot(t*1e3,V_out_ideal_20mV-np.mean(V_out_ideal_20mV), color='tab:red')
ideal_line_20mV.set_label('Linear Approximation')

ax[0].set_ylabel('$V_{out}: V_{in} = 1mV$')
ax[1].set_ylabel('$V_{out}: V_{in} = 20mV$')
ax[1].set_xlabel('Time [ms]')
#ax.set_xlabel('Time [s]')
#ax.set_ylim(-6, 6)
#ax[0].set_title('Nonlinear Distortion in a Common-Emitter Stage')
ax[0].grid()
ax[1].grid()
ax[0].legend()
ax[0].legend(loc='upper center', ncol=2, fancybox=True, 
           shadow=True, bbox_to_anchor=(0.5,1.2) )
plt.show()

""" Common-emitter gain as a function of temperature """
k = 1.38e-23
q = 1.6e-19
I_bias = 1e-3
T = np.linspace(-40, 105, num=165)
V_T = k*(T+273.15)/q
R_0 = 10e3
R_C = R_0*( 1 + 200/1e6*(T - 25) )
g_m = I_bias/V_T
Av_gm = g_m*R_0 
Av_gmR = g_m*R_C

fig, ax = plt.subplots()
gm_line, = ax.plot(T, Av_gm, color='tab:blue')
gmR_line, = ax.plot(T, Av_gmR, color='tab:red')
gm_line.set_label(r'$A_v = \frac{I_C}{V_T} \cdot R_0$')
gmR_line.set_label(r'$A_v = \frac{I_C}{V_T} \cdot R(T)$')
ax.legend()
ax.set_ylabel('Voltage Gain [V/V]')
ax.set_xlabel('Temperature [C]')
ax.set_title('Common-Emitter Voltage Gain as a Function of Temperature')
ax.grid()
plt.show()

""" Emitter-degenerated gain as a function of temperature """
k = 1.38e-23
q = 1.6e-19
T = np.linspace(-40, 105, num=165)
V_T = k*(T+273.15)/q
R_C0 = 10e3
R_E0 = 1e3
R_C = R_C0*( 1 + 200/1e6*(T - 25) )
R_E = R_E0*( 1 + 200/1e6*(T - 25) )
g_m = I_bias/V_T
Av = g_m*R_C/(1+g_m*R_E)

fig, ax = plt.subplots()
Av_line, = ax.plot(T, Av, color='tab:blue')
ax.set_ylabel('Voltage Gain [V/V]')
ax.set_xlabel('Temperature [C]')
ax.set_title('Emitter-Degenerated Gain as a Function of Temperature')
ax.grid()
plt.show()

""" Differential Signals with noise """
f = 1e3
w = f*2*np.pi
t = np.linspace(0,3e-3,num=300)
mu = 0
sigma = 100e-6

v_plus = 1e-3*np.sin(w*t)
v_minus = -1e-3*np.sin(w*t)
v_noise = np.random.normal(mu, sigma, 300)

vd_plus = v_plus + v_noise
vd_minus = v_minus + v_noise
v_diff = vd_plus - vd_minus

fig, ax = plt.subplots(3)
vplus_line = ax[0].plot(1e3*t, 1e3*vd_plus)
vminus_line = ax[1].plot(1e3*t, 1e3*vd_minus)
vdiff_line = ax[2].plot(1e3*t, 1e3*v_diff)

ax[0].set_ylabel('$v_+$ [mV]')
ax[1].set_ylabel('$v_-$ [mV]')
ax[2].set_ylabel('$v_+ - v_-$ [mV]')
ax[2].set_xlabel('Time [ms]')
plt.show()

""" Plot ID-VGS relationship for an n-channel MOSFET """
K = 5e-3
V_th = 1
V_GS = np.linspace(1,5,num=500)
V_ov = V_GS - V_th
I_D = ( K*V_ov**2 ) 

fig, ax = plt.subplots()
ax.plot(V_ov, 1e3*I_D)
ax.set_ylabel('$I_D$ [mA]')
ax.set_xlabel('$V_{GS} - V_{th}$ [V]')
ax.set_title('Drain Current vs $V_{GS} - V_{th}$ for an n-channel MOSFET')
ax.grid()
plt.show()

""" Plot ID-VDS relationship for an n-channel MOSFET 
    for lambda = 0"""
V_th = 1
V_GS = 1.5
V_DS_lin = np.linspace(0,.5,num=50)
V_DS_sat = [0.5, 2]
I_D_lin = 2*K*( (V_GS - V_th)*V_DS_lin - V_DS_lin**2/2 ) 
I_D_sat = [ 1e3*( K*(V_GS - V_th)**2 ), 1e3*( K*(V_GS - V_th)**2 )]  

fig, ax = plt.subplots()
ax.plot(V_DS_lin, 1e3*I_D_lin, V_DS_sat, I_D_sat, color='tab:blue')
ax.set_ylabel('$I_D$ [mA]')
ax.set_xlabel('$V_{DS}$ [V]')
ax.set_title('Drain Current vs $V_{DS}$ for an n-channel MOSFET')
ax.grid()
plt.show()

""" Plot ID-VDS relationship for an n-channel MOSFET 
    in saturation region"""
V_th = 1
V_GS = 1.5
V_DS_lin = np.linspace(0,.5,num=50)
V_DS_sat = np.linspace(.5,5,num=450)
lam = 0.1
I_D_lin = 2*K*( (V_GS - V_th)*V_DS_lin - V_DS_lin**2/2 ) * (1 + lam*V_DS_lin) 
I_D_sat = ( K*(V_GS - V_th)**2 ) * (1 + lam*V_DS_sat) 

fig, ax = plt.subplots()
ax.plot(V_DS_lin, 1e3*I_D_lin, V_DS_sat, 1e3*I_D_sat, color='tab:blue')
ax.set_ylabel('$I_D$ [mA]')
ax.set_xlabel('$V_{DS}$ [V]')
ax.set_title('Drain Current vs $V_{DS}$ for an n-channel MOSFET')
ax.grid()
plt.show()