import numpy as np, pandas as pd, mne, warnings
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
ROOT=Path("/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred")
sz=pd.read_csv(ROOT/"data/interim/manifests/seizure_manifest.csv")
sz=sz[sz.eligible_for_prediction & sz.vigilance.isin(["asleep","awake"])].copy()
rng=np.random.default_rng(11)
sz=sz.sample(frac=1.0,random_state=11).groupby("subject").head(2)   # <=2 per subject
sz=sz.head(120)
print("seizures sampled: %d, subjects %d, asleep %d awake %d"%(len(sz),sz.subject.nunique(),
      (sz.vigilance=="asleep").sum(),(sz.vigilance=="awake").sum()))
BANDS={"in_0.5_40":(0.5,40),"hi_40_48":(40,48),"hi_52_70":(52,70),"hi_70_90":(70,90),"hi_90_125":(90,125)}
def bandpow(x,fs):
    n=x.shape[-1]; w=np.hanning(n); X=np.fft.rfft(x*w,axis=-1)
    P=(np.abs(X)**2)/(np.sum(w**2)*fs); P[...,1:-1]*=2
    f=np.fft.rfftfreq(n,1/fs); return f,P
rows=[]
for _,r in sz.iterrows():
    p=ROOT/"data/raw"/f"sub-{str(r.subject).zfill(3)}"/f"ses-{str(r.session).zfill(2)}"/"eeg"/ \
      f"sub-{str(r.subject).zfill(3)}_ses-{str(r.session).zfill(2)}_task-{r.task}_run-{str(r.run).zfill(2)}_eeg.edf"
    if not p.exists(): continue
    try: raw=mne.io.read_raw_edf(p,preload=False,verbose="error")
    except Exception: continue
    fs=raw.info["sfreq"]; on=float(r.onset_seconds)
    def seg(a,b):
        s=int((on-a)*fs); e=int((on-b)*fs)
        if s<0 or e>raw.n_times: return None
        return raw.get_data(start=s,stop=e)
    last=seg(600,0); base=seg(2700,2100)      # last 10 min ; 45-35 min before
    if last is None or base is None: continue
    d={}
    for nm,x in (("last",last),("base",base)):
        nchunk=x.shape[1]//int(5*fs)
        xx=x[:,:nchunk*int(5*fs)].reshape(x.shape[0],nchunk,int(5*fs))
        f,P=bandpow(xx,fs)
        for b,(lo,hi) in BANDS.items():
            m=(f>=lo)&(f<hi)
            bp=P[...,m].sum(-1)               # (ch, nchunk)
            d[f"{nm}_{b}"]=np.median(np.log(bp.mean(0)+1e-12))   # mean over ch, median over chunks
    d.update(seizure_id=r.seizure_id,subject=r.subject,vig=r.vigilance,fs=fs)
    # power fraction above 40 Hz in the last 10 min
    d["frac_above40"]=float(np.exp(np.logaddexp.reduce([d[f"last_{b}"] for b in ["hi_40_48","hi_52_70","hi_70_90","hi_90_125"]]))/
                            np.exp(np.logaddexp.reduce([d[f"last_{b}"] for b in BANDS])))
    rows.append(d)
D=pd.DataFrame(rows); D.to_csv(ROOT/"outputs/analysis/refute6/claim6.csv",index=False)
print("n usable=%d subjects=%d  sfreq set=%s"%(len(D),D.subject.nunique(),sorted(D.fs.unique())))
print("median frac of 0.5-125Hz power above 40Hz (last 10 min) = %.4f  [IQR %.4f-%.4f]"%(
    D.frac_above40.median(),D.frac_above40.quantile(.25),D.frac_above40.quantile(.75)))
print("\npaired log(last10 / 45-35min) per band:")
for b in BANDS:
    d=D[f"last_{b}"]-D[f"base_{b}"]
    t,p=stats.ttest_1samp(d,0); print("  %-11s mean=%+.3f  t=%+.2f p=%.3f"%(b,d.mean(),t,p))
print("\nratio-of-ratios vs in-band:")
for b in list(BANDS)[1:]:
    d=(D[f"last_{b}"]-D[f"base_{b}"])-(D["last_in_0.5_40"]-D["base_in_0.5_40"])
    t,p=stats.ttest_1samp(d,0); print("  %-11s mean=%+.4f t=%+.2f p=%.3f"%(b,d.mean(),t,p))
y=(D.vig=="asleep").astype(int).values
def ci(y,s,n=3000,seed=0):
    rng=np.random.default_rng(seed); ip=np.where(y==1)[0]; ineg=np.where(y==0)[0]; o=[]
    for _ in range(n):
        a=rng.choice(ip,len(ip),True); b=rng.choice(ineg,len(ineg),True)
        o.append(roc_auc_score(np.r_[np.ones(len(a)),np.zeros(len(b))],np.r_[s[a],s[b]]))
    return np.percentile(o,[2.5,97.5]),np.array(o)
print("\nsingle-feature asleep-vs-awake AUC (n=%d, asleep=%d) -- NOT patient-held-out:"%(len(y),y.sum()))
boots={}
for b in BANDS:
    s=D[f"last_{b}"].values; a=roc_auc_score(y,s)
    if a<0.5: a=roc_auc_score(y,-s); s=-s
    (lo,hi),bs=ci(y,s); boots[b]=bs
    print("  %-11s AUC=%.3f  95%%CI=[%.3f,%.3f]"%(b,a,lo,hi))
d=boots["hi_40_48"]-boots["in_0.5_40"]
print("  paired bootstrap diff (40-48Hz minus 0.5-40Hz): mean=%+.3f 95%%CI=[%+.3f,%+.3f]  P(diff<=0)=%.3f"%(
    d.mean(),np.percentile(d,2.5),np.percentile(d,97.5),(d<=0).mean()))
