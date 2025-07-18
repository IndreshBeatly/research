import os 
import numpy as np 
import pandas as pd 

ppg_root      = r'C:\Users\Intern\Desktop\bp-new\output_data'
meta_path     = r'C:\Users\Intern\Desktop\bp-new\Subjects Information.xlsx'
output_folder = r'C:\Users\Intern\Desktop\bp-new\new-data'

# ─── Read metadata, using read_excel for xlsx ──────────────────────────────
if meta_path.lower().endswith(('.xls', '.xlsx')):
    meta = pd.read_excel(meta_path, dtype={"ID": str})
else:
    # fallback for CSV
    meta = pd.read_csv(meta_path, dtype={"ID": str}, encoding="cp1252")

# index by ID for fast lookup
meta = meta.set_index("ID", drop=False)

os.makedirs(output_folder, exist_ok=True)

for fname in os.listdir(ppg_root):
    if not fname.lower().endswith(".csv"):
        continue

    patient_id = os.path.splitext(fname)[0]
    if patient_id not in meta.index:
        print(f"ID {patient_id!r} not found in metadata; skipping.")
        continue

    csv_path = os.path.join(ppg_root, fname)
    df = pd.read_csv(csv_path, header=0)

    # ensure at least 4 columns
    if df.shape[1] < 4:
        print(f"⚠️  {fname} only has {df.shape[1]} cols; skipping.")
        continue

    # take last 2000 of the 4th column
    ppg = df.iloc[:, 3].values[-2000:]

    row = meta.loc[patient_id]
    sbp = float(row["SBP(mmHg)"])
    dbp = float(row["DBP(mmHg)"])

    out_path = os.path.join(output_folder, f"{patient_id}.npz")
    np.savez(out_path, ppg=ppg, sbp=sbp, dbp=dbp)
    print(f"✓ Wrote {out_path}  (ppg.shape={ppg.shape}, sbp={sbp}, dbp={dbp})")
