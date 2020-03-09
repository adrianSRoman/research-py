import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import linregress

fs = 2000
T = 1/fs
dur = 25
time = np.arange(0,dur,T)

faster = 1/0.250 # Hz
fast = 1/0.350 # Hz
slow = 1/0.600 # Hz
slower = 1/0.800 # Hz

a = 1
b = -1
l1 = 5
l2 = 0.0025
nlearn = 8
z = (0.0+1j)*np.ones(time.shape)

mean_slope = 0
spr_cv = 0
slopes = []
cvs = []

sprs = np.array([[310, 195, 230, 500, 580],[350, 200, 260, 540, 625],[355, 275, 320, 415, 830],
                [359, 244, 299, 449, 699],[382, 257, 307, 492, 707],[390, 200, 240, 600, 915],
                [390, 265, 315, 470, 655],[415, 275, 290, 490, 725],[418, 238, 298, 638, 908],
                [430, 225, 310, 510, 770],[435, 310, 375, 455, 485],[438, 213, 313, 578, 768],
                [439, 229, 314, 599, 724],[439, 199, 249, 664, 864],[443, 308, 363, 503, 813],
                [445, 285, 375, 535, 655],[455, 355, 407, 525, 601],[457, 245, 345, 582, 947],
                [462, 252, 362, 588, 664],[470, 245, 368, 735, 1045],[475, 255, 400, 585, 705],
                [525, 257, 362, 800, 1305],[572, 362, 447, 642, 697],[662, 222, 342, 982, 1382]])

for spr in enumerate(sprs)

    
    
    for i, f_s in enumerate([spr, faster, fast, slow, slower]):

        x = np.exp(1j*2*np.pi*time*f_s)
        x[int(nlearn*fs/f_s):] = 0
        f = (spr+0.01*np.random.randn())*np.ones(time.shape)

        for n, t in enumerate(time[:-1]):
            z[n+1] = z[n] + T*f[n]*(z[n]*(a + 1j*2*np.pi + b*(np.power(np.abs(z[n]),2))) + x[n])
            f[n+1] = f[n] + T*(-np.power(np.abs(f[n]-f_s),2)*l1*(np.real(x[n]))*np.sin(np.angle(z[n])) - (np.power(np.abs(spr-f[n]),2))*l2*(f[n]-spr)/spr)

        print('###############################')
        print('stimulus freq (Hz): ',f_s)
        print('learned IOI (ms): ', 1000/f[int(nlearn*fs/f_s)])
        #plt.plot(time,np.real(z))
        #plt.plot(time,np.real(x))
        #plt.plot(time,1/f)
        #plt.show()
        #plt.close()
        peaks, _ = find_peaks(np.real(z[int((nlearn+1.5)*fs/f_s):]))
        peaks = 1000*peaks/fs # converting to miliseconds
        slope, _, _, _, _ = linregress(range(len(np.diff(peaks))), np.diff(peaks))
        cv = np.std(np.diff(peaks))/np.mean(np.diff(peaks))
        print('Slope: ', slope - mean_slope)
        print('CV: ', cv)
        #plt.plot(np.diff(peaks))
        #plt.show()
        #plt.close()
        if i == 0:
            mean_slope = slope
            spr_cv = cv
        else:
            slopes.append(slope-mean_slope)
            cvs.append(cv)

cvs.insert(2,spr_cv)
plt.subplot(1,2,1)
plt.bar(range(5),cvs)
plt.subplot(1,2,2)
plt.bar(range(4),slopes)
plt.show()

