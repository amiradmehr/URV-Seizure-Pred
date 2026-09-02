import numpy as np, pandas as pd
from pathlib import Path
ROOT=Path("/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred"); FD=ROOT/"data/interim/chunk_features"
W=pd.read_csv(ROOT/"outputs/analysis/refute6/windows.csv")
bad=W[W.deadfrac>0.99]
print("recordings with >=99%% dead windows:", sorted(bad.rid.unique()))
dec=pd.read_csv(ROOT/"data/interim/manifests/decision_manifest.csv",
                usecols=["recording_id","label","split","decision_end_sample"])
sub=dec[dec.recording_id.isin(bad.rid.unique())]
print("\nIn the FULL manifest, those recordings hold:")
print(sub.groupby("recording_id").label.agg(["size","sum"]).rename(columns={"size":"n_decisions","sum":"n_positive"}))
print("\ntotal positives available in those recordings: %d"%sub.label.sum())
# how many positive decisions exist at all, per recording, corpus-wide
posrec=set(dec[dec.label==1].recording_id.unique())
print("recordings holding >=1 positive decision, corpus-wide: %d / %d"%(len(posrec),dec.recording_id.nunique()))
print("of the 8 dead recordings, how many hold >=1 positive: %d"%len(set(bad.rid.unique())&posrec))
# distribution of "dead" values
vals=[]
for rid in bad.rid.unique():
    b=np.load(FD/f"{rid}_features.npy",mmap_mode="r"); av=np.load(FD/f"{rid}_availability.npy")
    for col in (5,14,23):
        if not av[col]: continue
        x=np.asarray(b[:,col]); vals.append(x[x<-20])
vals=np.concatenate(vals)
print("\ndead-chunk log_total_power values n=%d: min=%.4f max=%.4f, frac exactly at floor(%.4f)=%.4f"%(
    vals.size,vals.min(),vals.max(),np.log(1e-12),np.mean(np.isclose(vals,np.log(1e-12),atol=1e-3))))
