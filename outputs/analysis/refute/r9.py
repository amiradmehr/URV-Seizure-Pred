import pandas as pd, numpy as np, glob, os, re
ROOT="/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred"
def edf_dur(p):
    with open(p,'rb') as f:
        h=f.read(256)
    nrec=int(h[236:244]); dur=float(h[244:252])
    return nrec*dur
sz=pd.read_csv(f'{ROOT}/data/interim/manifests/seizure_manifest.csv',dtype={'subject':str})
sz['rid']=sz.seizure_id.str.replace(r'_seizure-\d+$','',regex=True)
m=pd.read_csv(f'{ROOT}/data/interim/manifests/processed_shard_manifest.csv')
kept=set(m.recording_id)
drop=sorted(set(sz.rid)-kept)
rows=[]
for rid in drop:
    sub=rid.split('_')[0]; ses=rid.split('_')[1]
    p=f"{ROOT}/data/raw/{sub}/{ses}/eeg/{rid}_eeg.edf"
    rows.append((rid, edf_dur(p) if os.path.exists(p) else np.nan, os.path.exists(p)))
d=pd.DataFrame(rows,columns=['rid','dur_s','exists'])
print("dropped seizure-recordings:",len(d),"edf present:",d.exists.sum())
print(d.dur_s.describe())
print("dropped recs with duration >= 3300 s (would yield >=1 candidate decision):",(d.dur_s>=3300).sum())
print(d[d.dur_s>=3300].sort_values('dur_s',ascending=False).head(12).to_string(index=False))
# all raw recordings vs kept
alledf=glob.glob(f"{ROOT}/data/raw/sub-*/ses-*/eeg/*_eeg.edf")
allrid={os.path.basename(p).replace('_eeg.edf','') for p in alledf}
print("\ntotal raw recordings:",len(allrid),"kept:",len(kept),"dropped total:",len(allrid-kept))
missdur=[edf_dur(p) for p in alledf if os.path.basename(p).replace('_eeg.edf','') not in kept]
missdur=np.array(missdur)
print("dropped-overall durations: median %.0f s ; >=3300s: %d of %d"%(np.median(missdur),(missdur>=3300).sum(),len(missdur)))
keptdur=np.array([edf_dur(p) for p in alledf if os.path.basename(p).replace('_eeg.edf','') in kept])
print("kept durations: min %.0f s median %.0f s"%(keptdur.min(),np.median(keptdur)))
