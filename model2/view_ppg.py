import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
path = r'C:\Users\Intern\Desktop\bp-new\new-data\20.npz'
data = np.load(path)
ppg = data['ppg']
fs = 200.00
t = np.arange(len(ppg))/fs
SBP = data['sbp']
DBP = data['dbp']

min_dist = 0.4*fs
peaks,_ = find_peaks(ppg,distance=min_dist)
valleys,_ = find_peaks(-ppg,distance=min_dist)
# Detect dicrotic notches

# Detect dicrotic notches
notch = []

for i in range(len(peaks)-1):
    peak = peaks[i]
    next_peak = peaks[i+1]

    # Define search window: from peak to halfway to next peak (or full interval)
    segment_start = peak
    segment_end = next_peak

    # Extract this segment of ABP
    segment = ppg[segment_start:segment_end]

    # Find a local minimum (notch) in this segment
    local_min_idx, _ = find_peaks(-segment)  # peaks of negative signal = minima
    if len(local_min_idx) > 0:
        # Choose the first minimum after the peak (could refine with heuristics)
        notch_idx = segment_start + local_min_idx[0]
        notch.append(notch_idx)


ppg_peak_amps   = ppg[peaks]
ppg_trough_amps = ppg[valleys]
ppg_dn_amps = ppg[notch]
print(notch)


# compute means
mean_peak   = np.mean(ppg_peak_amps)
mean_trough = np.mean(ppg_trough_amps)
mean_dn = np.mean(ppg_dn_amps)

# “absolute amplitude” as you defined it
absolute_amplitude = mean_peak - mean_trough
absolute_amplitude2 = mean_dn - mean_trough
alx = (absolute_amplitude-absolute_amplitude2)/absolute_amplitude
print("ALX = ",alx)
print(SBP)
print(DBP)

plt.figure()
plt.plot(ppg,label='PPG')
plt.plot(peaks,ppg[peaks],'ro')
plt.plot(valleys,ppg[valleys],'go')
plt.plot(notch,ppg[notch],'bo')
#plt.hlines(mean_peak,   xmin=t[0], xmax=t[-1], colors='r', linestyles='--', label="Mean Peak")
#plt.hlines(mean_trough, xmin=t[0], xmax=t[-1], colors='b', linestyles='--', label="Mean Trough")
#plt.hlines(mean_dn, xmin=t[0], xmax=t[-1], colors='b', linestyles='--', label="Mean DN")
#plt.plot(t[notch],ppg[notch],'bo',label='DN')
plt.title('PPG with Dicrotic Notches')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()





