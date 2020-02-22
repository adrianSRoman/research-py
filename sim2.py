import numpy as np 
import matplotlib.pyplot as plt

from math import pi

# Simulation parameters
fs        = 10000
T         = 1/fs
dur       = 52
t         = np.linspace(0, dur, dur*fs)
ntime     = t.size
halfsamps = np.floor(ntime/2);


# z - parameters
a = 1
b = -1

z = 0.5 * np.exp(1j * 2 * pi) * np.ones(t.size) # oscillator1
y = 0.5 * np.exp(1j * 2 * pi) * np.ones(t.size) # oscillator2

f_1 = np.zeros(t.shape);  # Adaptive frequency osc1
f_2 = np.zeros(t.shape);  # Adaptive frequency osc2

#%%%%%%%%%%%%% Group Mismatch - SPR diff > 110 ms %%%%%%%%%%%%%%%%%%%%%%%%
freqs_miss1 = np.array([180, 210, 215, 220, 280, 310, 330, 350, 340, 420])
freqs_miss2 = np.array([310, 340, 330, 380, 410, 460, 462, 470, 500, 520])

#%%%%%%%%%%%%% Group Match - SPR diff < 10 ms %%%%%%%%%%%%%%%%%%%%%%%%%%%%
freqs_match1 = np.array([269, 280, 350, 343, 373, 376, 398, 420, 433, 451])
freqs_match2 = np.array([278, 289, 359, 352, 280, 384, 407, 429, 440, 460])

#%%%%%%%%%%%%%%%%%% HEBBIAN LEARNING PARAMETERS %%%%%%%%%%%%%%
lambda_1 = 2.7       # learning parameter
lambda_2 = 0.76      # flexibility parameter

f_s = 1000/400                       # Metronome Frequency 
F = np.exp(1j * 2 * pi * t * (f_s))  # Stimulus "Metronome"

mean_SPR_miss_pairs  = np.zeros((freqs_miss1.size, 4)) 
mean_SPR_match_pairs = np.zeros((freqs_miss1.size, 4))

tr_time = 0.400 * 4        # training time (s)
tr_samps = tr_time * 1000  # training samples

locs_z = pks_z = locs_F = pks_F = locs_y = pks_y = [] # Locations @ local maxima

# for all frequencies in group miss/match
for i in range(0, freqs_miss1.size):
    
    f_1[0] = 1000/freqs_miss1[i]
    f_2[0] = 1000/freqs_miss2[i] 
        
    # Forward Euler integration
    for j in range(1, ntime):
        if t[j-1] <= tr_time:
            z[j] = z[j-1] + T*f_2[j-1]*(z[j-1]*(a + 1j*2*pi + b*(abs(z[j-1])**2)) + F[j-1])
            f_2[j] = f_2[j-1] + T*(1/(2*pi))*f_2[j-1]*(-lambda_1*(F[j-1].real)*np.sin(np.angle(z[j-1])) - lambda_2*(f_2[j-1]-f_2[0])/f_2[0])

            y[j] = y[j-1] + T*f_1[j-1]*(y[j-1]*(a + 1j*2*pi + b*(abs(y[j-1])**2)) + F[j-1])
            f_1[j] = f_1[j-1] + T*(1/(2*pi))*f_1[j-1]*(-lambda_1*(F[j-1].real)*np.sin(np.angle(y[j-1])) - lambda_2*(f_1[j-1]-f_1[0])/f_1[0])
        else:
            z[j] = z[j-1] + T*f_2[j-1]*(z[j-1]*(a + 1j*2*pi + b*(abs(z[j-1])**2)) + y[j-1])
            f_2[j] = f_2[j-1] + T*(1/(2*pi))*f_2[j-1]*(-lambda_1*(y[j-1].real)*np.sin(np.angle(z[j-1])) - lambda_2*(f_2[j-1]-f_2[0])/f_2[0])

            y[j] = y[j-1] + T*f_1[j-1]*(y[j-1]*(a + 1j*2*pi + b*(abs(y[j-1])**2)) + z[j-1])
            f_1[j] = f_1[j-1] + T*(1/(2*pi))*f_1[j-1]*(-lambda_1*(z[j-1].real)*np.sin(np.angle(y[j-1])) - lambda_2*(f_1[j-1]-f_1[0])/f_1[0])

        # Find local maxima and location - (zero crossing)
        if (z[j].imag >= 0.0) and (z[j-1].imag <= 0.0):
            locs_z  = np.append(locs_z, j)
            pks_z   = np.append(pks_z, z[j].real)
        if (y[j].imag >= 0.0) and (y[j-1].imag <= 0.0):
            locs_y  = np.append(locs_y, j)
            pks_y   = np.append(pks_y, y[j].real)

    f_1trsamps  = np.take(f_1, tr_samps)
    f_2trsamps  = np.take(f_2, tr_samps)

    # Finding leader
    find_leader = []
    find_leader = np.append(find_leader, np.absolute(f_s - f_1trsamps.real))
    find_leader = np.append(find_leader, np.absolute(f_s - f_2trsamps.real))
    # Get leader
    leader = np.minimum(np.array(np.absolute(f_s - f_2trsamps.real)), np.array(np.absolute(f_s - f_1trsamps.real)))

    # Find which oscillator is more similar to stimulus
    which_min = np.where(find_leader == leader)

    if which_min[0][0] == 0:
        locs_lead = locs_z
        locs_follow = locs_y
    else:
        locs_lead = locs_y
        locs_follow = locs_z      

    new_followlocs = np.zeros(len(locs_lead));

    for iloc in range(0, len(locs_lead)):
        locsy_diff = np.absolute(locs_follow - locs_lead[iloc])
        nypeak_index = np.argmin(locsy_diff) # get index of minimum
        new_followlocs[iloc] = locs_follow[nypeak_index]
                          
    # find the index after training  
    tr_samps_locsz_diff = np.absolute(tr_samps - locs_lead)
    nzpeak_index = np.argmin(tr_samps_locsz_diff)
    mid_nzpeak = locs_lead[nzpeak_index]
    # eliminate the training part of the simulation for z
    locs_lead = locs_lead[nzpeak_index:]
    # eliminate the training part of the simulation for y
    new_followlocs = new_followlocs[nzpeak_index:]               
    # calculate number of peaks divisible by 4
    mod_four1 = np.mod(locs_lead.size, 4)
    # eliminate extra peaks
    locs_lead = locs_lead[0:(locs_lead.size)-mod_four1]
    new_followlocs = new_followlocs[0:(new_followlocs.size)-mod_four1]

    # Recover Variable names for computation
    if which_min[0][0] == 0:
        locs_z = locs_lead
        locs_y = new_followlocs
    else:
        locs_y = locs_lead
        locs_z = new_followlocs

    # Mean Asynchrony - break locations vector into 4
    # Note: 4 = cycles/repetitions of synchronization
    z_locsFourCycles = locs_z.reshape((-1, 4), order='F') # quemamada "F"
    y_locsFourCycles = locs_y.reshape((-1, 4), order='F')
    
    z_locsFourCycles = z_locsFourCycles.astype(int)
    
    y_locsFourCycles = y_locsFourCycles.astype(int)

    # Take mean of asynchronies over the four repetitions.
    mean_SPR_miss_pairs[i,:] = np.mean(np.absolute(np.take(t, z_locsFourCycles) - np.take(t, y_locsFourCycles)), 0)
    locs_z = pks_z = locs_F = pks_F = locs_y = pks_y = [] # Refresh ocations @ local maxima
                  

f_1 = np.zeros(t.shape);  # Adaptive frequency osc1
f_2 = np.zeros(t.shape);  # Adaptive frequency osc2

# for all frequencies in group miss/match
for i in range(0, freqs_miss1.size):
    
    f_1[0] = 1000/freqs_match1[i]
    f_2[0] = 1000/freqs_match2[i] 
        
    # Forward Euler integration
    for j in range(1, ntime):
        if t[j-1] <= tr_time:
            z[j] = z[j-1] + T*f_2[j-1]*(z[j-1]*(a + 1j*2*pi + b*(abs(z[j-1])**2)) + F[j-1])
            f_2[j] = f_2[j-1] + T*(1/(2*pi))*f_2[j-1]*(-lambda_1*(F[j-1].real)*np.sin(np.angle(z[j-1])) - lambda_2*(f_2[j-1]-f_2[0])/f_2[0])

            y[j] = y[j-1] + T*f_1[j-1]*(y[j-1]*(a + 1j*2*pi + b*(abs(y[j-1])**2)) + F[j-1])
            f_1[j] = f_1[j-1] + T*(1/(2*pi))*f_1[j-1]*(-lambda_1*(F[j-1].real)*np.sin(np.angle(y[j-1])) - lambda_2*(f_1[j-1]-f_1[0])/f_1[0])
        else:
            z[j] = z[j-1] + T*f_2[j-1]*(z[j-1]*(a + 1j*2*pi + b*(abs(z[j-1])**2)) + y[j-1])
            f_2[j] = f_2[j-1] + T*(1/(2*pi))*f_2[j-1]*(-lambda_1*(y[j-1].real)*np.sin(np.angle(z[j-1])) - lambda_2*(f_2[j-1]-f_2[0])/f_2[0])

            y[j] = y[j-1] + T*f_1[j-1]*(y[j-1]*(a + 1j*2*pi + b*(abs(y[j-1])**2)) + z[j-1])
            f_1[j] = f_1[j-1] + T*(1/(2*pi))*f_1[j-1]*(-lambda_1*(z[j-1].real)*np.sin(np.angle(y[j-1])) - lambda_2*(f_1[j-1]-f_1[0])/f_1[0])

        # Find local maxima and location - (zero crossing)
        if (z[j].imag >= 0.0) and (z[j-1].imag <= 0.0):
            locs_z  = np.append(locs_z, j)
            pks_z   = np.append(pks_z, z[j].real)
        if (y[j].imag >= 0.0) and (y[j-1].imag <= 0.0):
            locs_y  = np.append(locs_y, j)
            pks_y   = np.append(pks_y, y[j].real)

    f_1trsamps  = np.take(f_1, tr_samps)
    f_2trsamps  = np.take(f_2, tr_samps)

    # Finding leader
    find_leader = []
    find_leader = np.append(find_leader, np.absolute(f_s - f_1trsamps.real))
    find_leader = np.append(find_leader, np.absolute(f_s - f_2trsamps.real))
    # Get leader
    leader = np.minimum(np.array(np.absolute(f_s - f_2trsamps.real)), np.array(np.absolute(f_s - f_1trsamps.real)))

    # Find which oscillator is more similar to stimulus
    which_min = np.where(find_leader == leader)

    if which_min[0][0] == 0:
        locs_lead = locs_z
        locs_follow = locs_y
    else:
        locs_lead = locs_y
        locs_follow = locs_z      

    new_followlocs = np.zeros(len(locs_lead));

    for iloc in range(0, len(locs_lead)):
        locsy_diff = np.absolute(locs_follow - locs_lead[iloc])
        nypeak_index = np.argmin(locsy_diff) # get index of minimum
        new_followlocs[iloc] = locs_follow[nypeak_index]
                          
    # find the index after training  
    tr_samps_locsz_diff = np.absolute(tr_samps - locs_lead)
    nzpeak_index = np.argmin(tr_samps_locsz_diff)
    mid_nzpeak = locs_lead[nzpeak_index]
    # eliminate the training part of the simulation for z
    locs_lead = locs_lead[nzpeak_index:]
    # eliminate the training part of the simulation for y
    new_followlocs = new_followlocs[nzpeak_index:]               
    # calculate number of peaks divisible by 4
    mod_four1 = np.mod(locs_lead.size, 4)
    # eliminate extra peaks
    locs_lead = locs_lead[0:(locs_lead.size)-mod_four1]
    new_followlocs = new_followlocs[0:(new_followlocs.size)-mod_four1]

    # Recover Variable names for computation
    if which_min[0][0] == 0:
        locs_z = locs_lead
        locs_y = new_followlocs
    else:
        locs_y = locs_lead
        locs_z = new_followlocs

    # Mean Asynchrony - break locations vector into 4
    # Note: 4 = cycles/repetitions of synchronization
    z_locsFourCycles = locs_z.reshape((-1, 4), order='F') # quemamada "F"
    y_locsFourCycles = locs_y.reshape((-1, 4), order='F')
    
    z_locsFourCycles = z_locsFourCycles.astype(int)
    
    y_locsFourCycles = y_locsFourCycles.astype(int)

    # Take mean of asynchronies over the four repetitions.
    mean_SPR_match_pairs[i,:] = np.mean(np.absolute(np.take(t, z_locsFourCycles) - np.take(t, y_locsFourCycles)), 0)
    locs_z = pks_z = locs_F = pks_F = locs_y = pks_y = [] # Refresh ocations @ local maxima


# Create coupled bar plots
fig, ax = plt.subplots()
ind = np.arange(len(np.mean(mean_SPR_match_pairs, 0))) # x locations for the groups
width = 0.25  # width of the bars
p1 = ax.bar(ind, 1000 * np.mean(mean_SPR_match_pairs, 0), width, bottom=0)
p2 = ax.bar(ind + width, 1000 * np.mean(mean_SPR_miss_pairs, 0), width, bottom=0)

ax.set_title('Mean Absolute Asynchrony (ms)')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('1', '2', '3', '4'))

ax.legend((p1[0], p2[0]), ('Match', 'Missmatch'))
ax.autoscale_view()

plt.show()
 
