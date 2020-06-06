import os
import sys
import cmath
import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import signal
from scipy.integrate import quad
from scipy import integrate
from scipy import signal
from scipy.interpolate import interp1d
from scipy.fftpack import rfft
import cmath
from numpy.random import normal
from itertools import accumulate as acc

def fft_mag(vin, N, T, t):
    fft_sig  = rfft(vin)  
    fft_freqs = np.linspace(0.0, t/(2.0*T), N//2)
    fft_mag = 2.0/N * np.abs(fft_sig[:N//2])
    
    return fft_freqs, fft_mag
    
def generate_noise(mu, sigma, N, T):
    x = []
    y = []
    x = np.linspace(0, N*T, N)
    y = np.random.normal(mu, sigma, N)
       
    return x, y


""" Integrator Magnitude Response """
fs = 10e3
f = np.linspace(0, fs/2, 1000)
h_mag = 1/(2*abs(np.sin(np.pi*f/fs)))

fig, ax = plt.subplots()
ax.plot(f/fs, h_mag)
ax.grid(which='both',axis='both')
ax.set_xlabel(r'Normalized Frequency [$f/f_s$]')
ax.set_ylabel(r'Magnitude Response $|v_{out}/v_{in}|$')

plt.show()  

""" Differentiator Magnitude Response """
fs = 10e3
f = np.linspace(0, fs/2, 1000)
h_mag = 2*abs(np.sin(np.pi*f/fs))

fig, ax = plt.subplots()
ax.plot(f/fs, h_mag)
ax.grid(which='both',axis='both')
ax.set_xlabel(r'Normalized Frequency [$f/f_s$]')
ax.set_ylabel(r'Magnitude Response $|v_{out}/v_{in}|$')

plt.show()  

""" First-Order Noise Shaping """
fs = 10e3
Vref = 3.3
nbits = 1
f = np.linspace(0, fs/2, 1000)

ntf = 4*(np.sin(np.pi*f/fs))**2

fig, ax = plt.subplots()
ax.plot(f/fs, ntf, color='tab:blue', label='with noise shaping')
ax.plot(f/fs, np.ones(np.size(f)), color='tab:red', label='without')
ax.grid(which='both', axis='both')
ax.set_xlabel(r'Normalized Frequency [$f/f_s$]')
ax.set_ylabel(r'$|NTF|^2$')
ax.legend()
ax.legend(loc='upper center', ncol=2, fancybox=True, 
           shadow=True, bbox_to_anchor=(0.5,1.1) )
plt.show()

""" RMS Quantization Noise vs OSR """
en_qe_2 = (Vref/2)**2/12/fs*np.ones(np.size(f))
vn_rms = np.sqrt(integrate.cumtrapz(2*en_qe_2, f, initial=0))
vn_rms_ns = np.sqrt(integrate.cumtrapz(2*en_qe_2*ntf, f, initial=0))
vn_rms_tot = Vref/np.sqrt(12)/2

fig, ax = plt.subplots()
ax.semilogx(fs/f/2, vn_rms/vn_rms_tot, color='tab:blue', label='oversampling only')
ax.semilogx(fs/f/2, vn_rms_ns/vn_rms_tot, color='tab:red', label='with noise shaping')
ax.set_xlabel(r'Oversampling Ratio (OSR)')
ax.set_ylabel(r'Normalized RMS Quantization Noise')
ax.grid(which='both', axis='both')
ax.legend()
ax.legend(loc='upper center', ncol=2, fancybox=True, 
           shadow=True, bbox_to_anchor=(0.5,1.1) )
           
plt.show()

""" Effective Resolution vs OSR """
vfs_rms = Vref/2/np.sqrt(2)


fig, ax = plt.subplots()
ax.semilogx(fs/f/2, (20*np.log10(vfs_rms/vn_rms)-1.76)/6.02 - 1, color='tab:blue', label='oversampling only')
ax.semilogx(fs/f/2, (20*np.log10(vfs_rms/vn_rms_ns)-1.76)/6.02 - 1, color='tab:red', label='oversampling with noise shaping')
ax.set_xlabel(r'Oversampling Ratio (OSR)')
ax.set_ylabel(r'Resolution Improvement [bits]')
ax.grid(which='both', axis='both')
ax.legend()
ax.legend(loc='upper center', ncol=2, fancybox=True, 
           shadow=True, bbox_to_anchor=(0.5,1.1) )
           
plt.show()

""" Higher Order Modulators """
ntf_2 = ntf**2
ntf_3 = ntf**3
vn_rms_2 = np.sqrt(integrate.cumtrapz(2*en_qe_2*ntf_2, f, initial=0))
vn_rms_3 = np.sqrt(integrate.cumtrapz(2*en_qe_2*ntf_3, f, initial=0))

fig, ax = plt.subplots()
ax.semilogx(fs/f/2, (20*np.log10(vfs_rms/vn_rms)-1.76)/6.02 - 1, color='tab:blue', label='oversampling only')
ax.semilogx(fs/f/2, (20*np.log10(vfs_rms/vn_rms_ns)-1.76)/6.02 - 1, color='tab:red', label=r'$1^{st}$-order noise shaping')
ax.semilogx(fs/f/2, (20*np.log10(vfs_rms/vn_rms_2)-1.76)/6.02 - 1, color='tab:green', label=r'$2^{nd}$-order noise shaping')
ax.semilogx(fs/f/2, (20*np.log10(vfs_rms/vn_rms_3)-1.76)/6.02 - 1, color='tab:orange', label=r'$3^{rd}$-order noise shaping')
ax.set_xlabel(r'Oversampling Ratio (OSR)')
ax.set_ylabel(r'Resolution Improvement [bits]')
ax.grid(which='both', axis='both')
ax.legend()
ax.legend(loc='upper center', ncol=2, fancybox=True, 
           shadow=True, bbox_to_anchor=(0.5,1.15) )
           
plt.show()

