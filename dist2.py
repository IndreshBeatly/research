import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks,butter,filtfilt

# 1) Load & clean exactly as before
df = pd.read_csv(
    r'C:\Users\Intern\Desktop\bp-new\data\data_122_86.csv',
    skipinitialspace=True
)
# 2) Peak detection
record_duration_s = 10.0
fs_ecg = 500.0
fs_ppg = 50.0

# ECG
ecg = df['y0000'].values
print(ecg)
ecg = ecg[:5000]

# bandpass filter 
def butter_bandpass(lowcut,highcut,fs,order=5):
    nyq = 0.5*fs
    low = max(lowcut/nyq,0.001)
    high = min(highcut/nyq,0.99)
    b,a = butter(order,[low,high],btype='band')
    return b,a
# apply bandpass filter 
lowcut_ecg = 0.5
highcut_ecg = 40.0
b,a = butter_bandpass(lowcut_ecg,highcut_ecg,fs_ecg)
ecg_filtered = filtfilt(b,a,ecg)

t_ecg = np.arange(len(ecg_filtered)) / fs_ecg
min_dist_ecg = int(0.4 * fs_ecg)  # min 0.6 s apart
peaks_ecg, _ = find_peaks(ecg_filtered, distance=min_dist_ecg)

# PPG
ppg = df['y0001'].values
ppg = ppg[:1500]
lowcut_ppg = 0.5
highcut_ppg = 8.0
b,a = butter_bandpass(lowcut_ppg,highcut_ppg,fs_ppg)
ppg_filtered = filtfilt(b,a,ppg)

t_ppg = np.arange(len(ppg_filtered)) / fs_ppg
min_dist_ppg = int(0.6 * fs_ppg)
peaks_ppg, _ = find_peaks(ppg_filtered, distance=min_dist_ppg)

ppg_peak_time = []
for x in peaks_ppg:
    a = x/fs_ppg
    ppg_peak_time.append(a)

print(ppg_peak_time)

ecg_peak_time = []
for x in peaks_ecg:
    a = x/fs_ecg
    ecg_peak_time.append(a)

print(ecg_peak_time)

l = min(len(ecg_peak_time),len(ppg_peak_time))
pat = []
for i in range (l):
    arrival_time = ecg_peak_time[i]-ppg_peak_time[i]
    pat.append(arrival_time)
print(pat)
a_time = np.mean(pat)
if(a_time<0):a_time = -a_time
print("Mean A Time : ",a_time)

# 3) Plot with independent x-axes
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(12, 8))

# ECG subplot
ax1.plot(t_ecg, ecg_filtered, label='ECG (500 Hz)')
ax1.scatter(t_ecg[peaks_ecg], ecg_filtered[peaks_ecg],
            c='red', marker='o', label='R-peaks')
ax1.set_xlim(0, record_duration_s)
ax1.set_ylabel('ECG amplitude')
ax1.set_title('ECG — fs = 500 Hz')
ax1.legend(loc='upper right')
ax1.grid(True)

# PPG subplot
ax2.plot(t_ppg, ppg_filtered, label='PPG (50 Hz)')
ax2.scatter(t_ppg[peaks_ppg], ppg_filtered[peaks_ppg],
            c='red', marker='o', label='PPG peaks')
ax2.set_xlim(0, record_duration_s)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('PPG amplitude')
ax2.set_title('PPG — fs = 50 Hz')
ax2.legend(loc='upper right')
ax2.grid(True)

plt.tight_layout()
plt.show()
