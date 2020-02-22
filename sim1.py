import numpy as np 
import matplotlib.pyplot as plt

from math import pi

# Simulation parameters
fs        = 240
T         = 1/fs
dur       = 250
t         = np.linspace(0, dur, dur*fs)
ntime     = t.size
halfsamps = np.floor(ntime/2);


# z - parameters
a = 1
b = -1

z = 0.5 * np.exp(1j * 2 * pi) * np.ones(t.size) # oscillator

#%%%%%%%%%%%%%%%%%%%% Group Musicians %%%%%%%%%%%%%%%%%%%%%%%
# Mean Group SPR - (404ms)
musicians = np.array([250, 260, 300, 310, 325, 340, 345, 350, 380, 400, 410, 430, 440, 450, 460, 465, 475, 480, 600, 650])

#%%%%%%%%%%%%%%%%%% Group Non-Musicians %%%%%%%%%%%%%%%%%%%%%%
# Mean Group SPR - (306ms)
# musicians = np.array([200 250 255 260 265 275 275 285 300 305 310 315 315 315 320 340 345 360 380 450])

#%%%%%%%%%%%%%%%%%% HEBBIAN LEARNING PARAMETERS %%%%%%%%%%%%%%
lambda_1 = 2.7;  #9.6316;      # learning parameter
lambda_2 = 0.76; #2.2368;      # flexibility parameter

# zeros - params: shape
mean_indiv = np.zeros((musicians.size, 4)) 
locs_z = pks_z = locs_F = pks_F  = []        # Locations @ local maxima
for ispr in range(0, musicians.size):
    
    # generate period lengths ms 30% faster 15% ... to a given SPR
    f_s = np.linspace(0.7, 1.30, 5) * musicians[ispr]  # Metronome's period length in milliseconds
    f_s = np.delete(f_s, 2, axis=0) # remove central element
    mean_asyn_freqs = np.zeros(f_s.shape)
    
    # iterate over stimulus frequencies
    for i in range(0, f_s.size):
        f = np.zeros(t.shape);                       # Adaptive frequency (Hebbian)
        f[0] = 1000/(musicians[ispr]);               # Get musician's SPR
        F = np.exp(1j * 2 * pi * t * (1000/f_s[i]))  # Stimulus "Metronome"
        locs_z = pks_z = locs_F = pks_F  = []        # Locations @ local maxima
        
        # Forward Euler integration
        for j in range(1, ntime):
            z[j] = z[j-1] + T*f[j-1]*(z[j-1]*(a + 1j*2*pi + b*(abs(z[j-1])**2)) + F[j-1])
            f[j] = f[j-1] + T*(1/(2*pi))*f[j-1]*(-lambda_1*(F[j-1].real)*np.sin(np.angle(z[j-1])) - lambda_2*(f[j-1]-f[0])/f[0])
            
            # Find local maxima and location - (zero crossing)
            if (z[j].imag >= 0.0) and (z[j-1].imag <= 0.0):
                locs_z  = np.append(locs_z, j)
                pks_z   = np.append(pks_z, z[j].real)
            if (F[j].imag >= 0.0) and (F[j-1].imag <= 0.0):
                locs_F  = np.append(locs_F, j)
                pks_F   = np.append(pks_F, z[j].real)
                
        np.insert(locs_F, 0, 1)
        
        # which z peak is closest to the midpoint of the simulation?
        halfsamps_locsz_diff = np.absolute(halfsamps - locs_z)
        mid_nzpeak_index = np.argmin(halfsamps_locsz_diff) # get index of minimum @ half samp
        mid_nzpeak = locs_z[mid_nzpeak_index]

        # eliminate the first half of the simulation for z
        locs_z = locs_z[mid_nzpeak_index:]

        # which F peak is closest to mid_nzpeak?
        mid_nzpeak_locs_F_diff = np.absolute(locs_F - mid_nzpeak)
        mid_F_peaks_index = np.argmin(mid_nzpeak_locs_F_diff)

        # which z peak is penultimate?
        pen_nzpeak = locs_z[-2]
        # which F peak is closest to the penultimate z peak?
        pen_nzpeak_locs_F_diff = np.absolute(locs_F - pen_nzpeak)
        pen_F_peaks_index = np.argmin(pen_nzpeak_locs_F_diff)

        # compute the mean asynchrony
        mean_asynchrony = locs_z[0:-2] - locs_F[mid_F_peaks_index:pen_F_peaks_index]
        mean_asyn_freqs[i] = 1000 * mean_asynchrony.mean(0)/fs
        
    mean_indiv[ispr,:] = mean_asyn_freqs

mean_asynchronies = mean_indiv.mean(0);
plt.bar(np.arange(len(mean_asynchronies)), mean_asynchronies)


 
