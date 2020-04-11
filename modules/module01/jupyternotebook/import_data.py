import csv
import numpy as np
import matplotlib.pyplot as plt

    
def read_ltspice_tran(file_name):
    with open(file_name, 'r') as data:
        x = []
        y = []
        next(data) # skip header line
        for line in data:
            p = line.split()
            x.append(float(p[0]))
            y.append(float(p[1]))

    return x, y

times, voltages = read_ltspice_tran('noise_result.txt')
print(np.std(voltages))
print(np.shape(times))
# print(times)
# print(voltages)

fig, ax = plt.subplots()
ax.plot(times, voltages)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Voltage [V]')
ax.grid()
plt.show()

def read_ltspice_ac(file_name):
    with open(file_name, 'r') as data:
        x = []
        y = []
        z = []
        next(data) # skip header line
        for line in data:
            p = line.split()
            x.append(float(p[0]))
            complex = p[1].split(",")
            y.append(float(complex[0]))
            z.append(float(complex[1]))

    return x, y, z

freqs, reals, imags = read_ltspice_ac('RC_LP_ac.txt')
mags = np.sqrt(np.asarray(reals)**2 + np.asarray(imags)**2)

fig, ax = plt.subplots()
ax.semilogx(freqs, 20*np.log10(mags))
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('Magnitude [dB]')
ax.grid()
plt.show()