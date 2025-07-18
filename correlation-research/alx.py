#detecting landmarks in the ppg wave
# at the top of your file
import numpy as np
import pywt
from scipy.signal import find_peaks, savgol_filter,argrelextrema
import matplotlib.pyplot as plt
# load a single window for feature extraction 
path = r"C:\Users\Intern\Desktop\bp-new\p060441\win3.npz"
data = np.load(path)
ppg = data["ppg"]
fs = 125
sbp = data["SegSBP"]
dbp = data["SegDBP"]

print(sbp , dbp , fs )

def detect_landmarks(ppg:np.ndarray,fs:float):
    #1. locate peaks
    peaks,_ = find_peaks(ppg,distance=int(0.4*fs),prominence=0.02)
    hr = len(peaks)

    #2. locate feet - minima before each peak
    foots = []
    for i,p in enumerate(peaks):
        start = 0 if i==0 else peaks[i-1]
        segment = ppg[start:p]
        foot = start + np.argmin(segment)
        foots.append(foot)

    #3. using peaks and feet index lets build the landmarks for all peaks except the last one
    # because for it , there is no next foot in our list so we cant compute Tend adn will get index out of bound error
    landmarks = []
    for i in range(len(peaks)-1):

        Tstart = foots[i]
        Tp     = peaks[i]
        Tend   = foots[i+1]

        # rising half amplitude 
        amp0,amp1 = ppg[Tstart],ppg[Tp]
        half_r = amp0 + 0.5*(amp1-amp0)
        idx_r = np.where(ppg[Tstart:Tp]>=half_r)[0]
        Thalf_r = Tstart + (idx_r[0] if len(idx_r) else (Tp-Tstart)//2)

        # falling half amplitude
        amp2 = ppg[Tend]
        half_f = amp1 + 0.5*(amp2-amp1)
        seg_f = ppg[Tp:Tend]
        idx_f = np.where(seg_f<=half_f)[0]
        Thalf_f = Tp + (idx_f[0] if len(idx_f) else (Tend-Tp)//2)

        # dicrotic notch detection
        valley_seg = ppg[Tp:Tend]
        dt = 1.0/fs

        # 1) compute & smooth SDPPG
        vpg   = np.gradient(valley_seg, dt)
        sd2   = np.gradient(vpg, dt)
        # smooth out tiny wiggles
        sd2_s = savgol_filter(sd2, window_length=7, polyorder=3)

        # 2) find SDPPG minima (a-b-c-d-e pattern) in 25 to 85% region
        lo1,hi1 = int(0.25*len(sd2_s)),int(0.85*len(sd2_s))
        core2   = sd2_s[lo1:hi1]
        mins = argrelextrema(core2,np.less)[0]
        if len(mins) >= 2:
            # the 2nd trough is the “d‐wave” (true notch)
            d_idx = mins[1]
        elif len(mins) == 1:
            d_idx = mins[0]
        else:
            d_idx = None

        if d_idx is not None:
            # 3) targeted CWT refine around that d_idx
            start_w = max(0, d_idx-20)
            win     = valley_seg[start_w : d_idx+20]
            scales, coeffs = np.arange(3,8), None
            coeffs, _ = pywt.cwt(win, scales, 'mexh', sampling_period=1/fs)
            power     = np.sum(np.abs(coeffs), axis=0)
            if power.size:
                refine = np.argmax(power)
                d_idx  = (start_w) + refine
            Tdn = Tp + lo1 + d_idx
        else:
            # 4) fallback #1: first local valley in core
            inv_core = -valley_seg[lo1:hi1]
            wpk, _   = find_peaks(inv_core,
                                  distance=int(0.05*fs),
                                  prominence=0.01)
            if len(wpk) > 0:
                d_idx = wpk[0]
            else:
                # 5) fallback #2: deepest point in core
                rel_min = np.argmin(valley_seg[lo1:hi1])
                d_idx   = rel_min
            Tdn = Tp + lo1 + d_idx
        # e) final fallback: if still no notch, use Thalf_f
        if Tdn is None:
            Tdn = Thalf_f



        mean_peaks = np.mean(Tp)
        mean_valley = np.mean(Tend)
        mean_dn = np.mean(Tdn)
        amp1 = mean_peaks-mean_valley
        amp2 = mean_dn-mean_valley
        alx = (amp1-amp2)/amp1
        # append all the landmarks to the array and return 
        landmarks.append((alx,hr))
        alx_mean = np.mean(alx)
    print(alx_mean,hr)
    #return landmarks

lm = detect_landmarks(ppg,fs)
