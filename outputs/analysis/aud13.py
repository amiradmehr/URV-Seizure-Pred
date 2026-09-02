import numpy as np, pandas as pd, mne, time, sys
from scipy.signal import welch
from scipy import stats
mne.set_log_level("ERROR")
S=pd.read_pickle("outputs/analysis/S.pkl")
S["recording_id"]=("sub-"+S.subject+"_ses-"+S.session.astype(str).str.zfill(2)+"_task-"+S.task+"_run-"+S.run.astype(str).str.zfill(2))
elig=S[S.eligible_for_prediction].copy()
N=int(sys.argv[1])
pick=elig.sample(n=min(N,len(elig)),random_state=11)
BANDS=[("in_0.5_40",0.5,40),("hi_40_48",40,48),("hi_52_70",52,70),("hi_70_90",70,90),("hi_90_125",90,125)]
def bandpow(x,sf):
    f,P=welch(x,fs=sf,nperseg=1280,noverlap=0,axis=-1)
    return {n:P[...,(f>=lo)&(f<hi)].sum(axis=-1) for n,lo,hi in BANDS}
rows=[];t0=time.time()
for _,r in pick.iterrows():
    p=f"data/raw/sub-{r.subject}/ses-{str(r.session).zfill(2)}/eeg/{r.recording_id}_eeg.edf"
    try: raw=mne.io.read_raw_edf(p,preload=False)
    except Exception: continue
    sf=raw.info["sfreq"]; T=float(r.onset_seconds)
    d={}
    ok=True
    for name,t in [("last10",T-600),("mid",T-1800),("early",T-3000)]:
        a=int(t*sf); b=a+int(600*sf)
        if a<0 or b>raw.n_times: ok=False;break
        x=raw.get_data(start=a,stop=b); x=x-x.mean(axis=1,keepdims=True)
        for k,v in bandpow(x,sf).items(): d[f"{name}::{k}"]=float(np.mean(v))
    if not ok: continue
    d.update(dict(seizure_id=r.seizure_id,subject=r.subject,vigilance=r.vigilance))
    rows.append(d)
R=pd.DataFrame(rows); R.to_pickle("outputs/analysis/HF2.pkl")
print("elapsed %.0fs  n=%d  subjects=%d"%(time.time()-t0,len(R),R.subject.nunique()))
print(f"\nPAIRED within the clean pre-seizure hour: log(last10min / [45-35min before onset])")
print(f"{'band':12s} {'mean_log':>9s} {'t':>7s} {'p':>8s}  {'n>0':>6s}")
for n,_,_ in BANDS:
    a=np.log(R[f"last10::{n}"]/R[f"early::{n}"]).replace([np.inf,-np.inf],np.nan).dropna()
    t,pv=stats.ttest_1samp(a,0)
    print(f"{n:12s} {a.mean():9.3f} {t:7.2f} {pv:8.4f}  {int((a>0).sum()):3d}/{len(a)}")
print(f"\nRATIO of high-band to in-band, last10 vs early (log ratio-of-ratios):")
for n,_,_ in BANDS[1:]:
    a=np.log((R[f"last10::{n}"]/R["last10::in_0.5_40"])/(R[f"early::{n}"]/R["early::in_0.5_40"])).replace([np.inf,-np.inf],np.nan).dropna()
    t,pv=stats.ttest_1samp(a,0)
    print(f"{n:12s} {a.mean():9.4f} {t:7.2f} {pv:8.4f}")
print("\nmedian fraction of 0.5-125Hz power above 40 Hz (last10):",
      float(np.nanmedian(1-R["last10::in_0.5_40"]/R[[f"last10::{n}" for n,_,_ in BANDS]].sum(axis=1))))
print("\nby vigilance (n):",R.vigilance.value_counts().to_dict())
for n,_,_ in BANDS:
    g=R.groupby("vigilance")[f"last10::{n}"].apply(lambda x: np.log(x).median())
    print(f"  {n:12s}", {k:round(v,2) for k,v in g.items()})
