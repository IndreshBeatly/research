import numpy as np
import pywt
import os
from scipy.signal import find_peaks, savgol_filter, argrelextrema
from tqdm import tqdm   # <-- tqdm for progress bar


def detect_landmarks(ppg: np.ndarray, fs: float, sbp: float, dbp: float):
    peaks, _ = find_peaks(ppg, distance=int(0.4 * fs), prominence=0.02)
    hr = len(peaks)

    foots = []
    for i, p in enumerate(peaks):
        start = 0 if i == 0 else peaks[i - 1]
        segment = ppg[start:p]
        foot = start + np.argmin(segment)
        foots.append(foot)

    alx_list = []

    for i in range(len(peaks) - 1):
        Tstart = foots[i]
        Tp = peaks[i]
        Tend = foots[i + 1]

        amp0, amp1 = ppg[Tstart], ppg[Tp]
        half_r = amp0 + 0.5 * (amp1 - amp0)
        idx_r = np.where(ppg[Tstart:Tp] >= half_r)[0]
        Thalf_r = Tstart + (idx_r[0] if len(idx_r) else (Tp - Tstart) // 2)

        amp2 = ppg[Tend]
        half_f = amp1 + 0.5 * (amp2 - amp1)
        seg_f = ppg[Tp:Tend]
        idx_f = np.where(seg_f <= half_f)[0]
        Thalf_f = Tp + (idx_f[0] if len(idx_f) else (Tend - Tp) // 2)

        valley_seg = ppg[Tp:Tend]
        dt = 1.0 / fs
        vpg = np.gradient(valley_seg, dt)
        sd2 = np.gradient(vpg, dt)
        sd2_s = savgol_filter(sd2, window_length=7, polyorder=3)
        lo1, hi1 = int(0.25 * len(sd2_s)), int(0.85 * len(sd2_s))
        core2 = sd2_s[lo1:hi1]
        mins = argrelextrema(core2, np.less)[0]

        if len(mins) >= 2:
            d_idx = mins[1]
        elif len(mins) == 1:
            d_idx = mins[0]
        else:
            d_idx = None

        if d_idx is not None:
            start_w = max(0, d_idx - 20)
            win = valley_seg[start_w: d_idx + 20]
            scales = np.arange(3, 8)
            coeffs, _ = pywt.cwt(win, scales, 'mexh', sampling_period=1 / fs)
            power = np.sum(np.abs(coeffs), axis=0)
            if power.size:
                refine = np.argmax(power)
                d_idx = (start_w) + refine
            Tdn = Tp + lo1 + d_idx
        else:
            inv_core = -valley_seg[lo1:hi1]
            wpk, _ = find_peaks(inv_core, distance=int(0.05 * fs), prominence=0.01)
            if len(wpk) > 0:
                d_idx = wpk[0]
            else:
                rel_min = np.argmin(valley_seg[lo1:hi1])
                d_idx = rel_min
            Tdn = Tp + lo1 + d_idx

        if Tdn is None:
            Tdn = Thalf_f

        amp1 = ppg[Tp] - ppg[Tend]
        amp2 = ppg[Tdn] - ppg[Tend]
        alx = (amp1 - amp2) / amp1 if amp1 != 0 else 0
        alx_list.append(alx)

    mean_alx = np.mean(alx_list) if alx_list else None
    return mean_alx, hr, sbp, dbp


# ---------- Main Batch Processing -----------
root_folder = r"/home/beatly-digital/Documents/indresh/indresh/ppg/ppg2bpv3/processed"

alx_list = []
hr_list = []
sbp_list = []
dbp_list = []

# --- First count how many files you are going to process for tqdm ---
npz_files = []
for dirpath, dirnames, filenames in os.walk(root_folder):
    if not os.path.basename(dirpath).startswith('p0'):
        continue
    for file in filenames:
        if file.endswith('.npz'):
            npz_files.append(os.path.join(dirpath, file))

# --- tqdm progress bar here ---
for file_path in tqdm(npz_files, desc="Processing files"):
    data = np.load(file_path)

    ppg = data["ppg"]
    fs = 125
    sbp = data["SegSBP"]
    dbp = data["SegDBP"]

    alx, hr, sbp_val, dbp_val = detect_landmarks(ppg, fs, sbp, dbp)

    # Handle None ALX safely
    alx = 0.0 if alx is None else alx

    alx_list.append(alx)
    hr_list.append(hr)
    sbp_list.append(sbp_val)
    dbp_list.append(dbp_val)

# Save everything into a .npz
save_path = os.path.join(root_folder, "aggregated_metrics.npz")
np.savez(save_path, alx=np.array(alx_list), hr=np.array(hr_list), sbp=np.array(sbp_list), dbp=np.array(dbp_list))

print(f"\nSaved aggregated metrics to: {save_path}")
