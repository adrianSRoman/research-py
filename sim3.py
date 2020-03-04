import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import linregress

fs = 2000
T = 1/fs
dur = 25
time = np.arange(0,dur,T)

spr = 1/0.450 # Hz
faster = 1/0.250 # Hz
fast = 1/0.350 # Hz
slow = 1/0.600 # Hz
slower = 1/0.800 # Hz

a = 1
b = -1
l1 = 2*2.7
l2 = 1.0/1024
nlearn = 8
z = (0.0+1j)*np.ones(time.shape)

mean_slope = 0
for i, f_s in enumerate([spr, faster, fast, slow, slower]):

	x = np.exp(1j*2*np.pi*time*f_s)
	x[int(nlearn*fs/f_s):] = 0
	f = spr*np.ones(time.shape)

	for n, t in enumerate(time[:-1]):
		z[n+1] = z[n] + (T*f[n])*(z[n]*(a + 1j*2*np.pi + b*(np.power(np.abs(z[n]),2))) + x[n])
		f[n+1] = f[n] + T*(-np.abs(f[n]-f_s)*l1*(np.real(x[n]))*np.sin(np.angle(z[n])) - (f[n])*np.abs(spr-f[n])*l2*(f[n]-spr)/spr)
	
	print('###############################')
	print('stimulus freq (Hz): ',f_s)
	print('learned IOI (ms): ', 1000/f[int(nlearn*fs/f_s)])
	#plt.plot(time,np.real(z))
	#plt.plot(time,np.real(x))
	#plt.plot(time,1/f)
	#plt.show()
	#plt.grid()
	#plt.clf
	peaks, _ = find_peaks(np.real(z[int((nlearn+1.5)*fs/f_s):]))
	peaks = 1000*peaks/fs # converting to miliseconds
	slope, _, _, _, _ = linregress(range(len(np.diff(peaks))), np.diff(peaks))
	print('Slope: ', slope - mean_slope)
	#plt.plot(np.diff(peaks))
	#plt.show()
	#plt.clf
	if i == 0:
		mean_slope = slope
		
