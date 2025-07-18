import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

df = pd.read_csv(r'C:\Users\Intern\Desktop\bp-new\dhilip_116_82.csv')
print(df.head())
signal = df['y0000'].values
ppg = df['y0001'].values

signal = signal
ppg = ppg

n = len(signal)
print(n)
t = 60.00
fs = n/t
print(fs)
min_dist = int(0.6*fs)
peaks,_ = find_peaks(signal,distance=min_dist)
ppgpeaks,_ = find_peaks(ppg,distance=min_dist)
print("Detected R peak indices : ",peaks)
print("Detected PPG peaks ",ppgpeaks)
times = np.arange(n)/fs
plt.figure()
plt.plot(times,signal)
plt.scatter(times[peaks],signal[peaks],marker='x')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('ECG Signal with Detected R-peaks')
plt.show()

plt.figure()
plt.plot(times,ppg)
plt.scatter(times[ppgpeaks],ppg[ppgpeaks],marker='x')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('PPG Signal with Detected peaks')
plt.show()
