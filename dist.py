import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 1) Load & clean exactly as before
df = pd.read_csv(
    r'C:\Users\Intern\Desktop\bp-new\data_138_100.csv',
    skipinitialspace=True
)
# 2) Peak detection
record_duration_s = 30.0
fs_ecg = 500.0
fs_ppg = 50.0

# ECG
ecg = df['y0000'].values
ecg = ecg[:15000]
t_ecg = np.arange(len(ecg)) / fs_ecg
min_dist_ecg = int(0.1 * fs_ecg)  # min 0.6 s apart
peaks_ecg, _ = find_peaks(ecg, distance=min_dist_ecg)

# PPG
ppg = df['y0001'].values
ppg = ppg[:1500]
t_ppg = np.arange(len(ppg)) / fs_ppg
min_dist_ppg = int(0.6 * fs_ppg)
peaks_ppg, _ = find_peaks(ppg, distance=min_dist_ppg)

# 3) Plot with independent x-axes
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(12, 8))

# ECG subplot
ax1.plot(t_ecg, ecg, label='ECG (500 Hz)')
ax1.scatter(t_ecg[peaks_ecg], ecg[peaks_ecg],
            c='red', marker='o', label='R-peaks')
ax1.set_xlim(0, record_duration_s)
ax1.set_ylabel('ECG amplitude')
ax1.set_title('ECG — fs = 500 Hz')
ax1.legend(loc='upper right')
ax1.grid(True)

# PPG subplot
ax2.plot(t_ppg, ppg, label='PPG (50 Hz)')
ax2.scatter(t_ppg[peaks_ppg], ppg[peaks_ppg],
            c='red', marker='o', label='PPG peaks')
ax2.set_xlim(0, record_duration_s)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('PPG amplitude')
ax2.set_title('PPG — fs = 50 Hz')
ax2.legend(loc='upper right')
ax2.grid(True)

plt.tight_layout()
plt.show()
