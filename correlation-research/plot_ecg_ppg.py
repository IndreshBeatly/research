import matplotlib.pyplot as plt
import numpy as np 
from scipy.signal import find_peaks

path = r'C:\Users\Intern\Desktop\bp-new\p095343\win1.npz'
data = np.load(path)
ppg = data['PPG_Record_F']
ecg = data['ECG_Record_F']
fs = 125.0
time = np.arange(len(ppg))/fs 
min_dist_ecg = int(0.4 * fs)  # min 0.6 s apart
peaks_ecg, _ = find_peaks(ecg, distance=min_dist_ecg)
min_dist_ppg = int(0.4* fs)  # min 0.6 s apart
peaks_ppg, _ = find_peaks(ppg, distance=min_dist_ppg)

# plot ppg and ecg 
fig,(ax1,ax2) = plt.subplots(2,1,sharex = False,figsize=(12,8))
# ecg subplot
ax1.plot(time,ecg,label='ECG')
ax1.scatter(time[peaks_ecg], ecg[peaks_ecg],
            c='red', marker='o', label='R-peaks')
ax1.set_ylabel('ECG amplitude')
ax1.set_title('ECG')
ax1.legend(loc='upper right')
ax1.grid(True)
# ppg subplot
ax1.plot(time,ppg,label='PPG')
ax1.scatter(time[peaks_ppg], ppg[peaks_ppg],
            c='red', marker='o', label='R-peaks')
ax1.set_ylabel('PPG amplitude')
ax1.set_title('PPG')
ax1.legend(loc='upper right')
ax1.grid(True)

plt.tight_layout()
plt.show()
