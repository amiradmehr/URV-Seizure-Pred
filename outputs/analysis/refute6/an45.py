import numpy as np, pandas as pd, sys, json
from pathlib import Path
sys.path.insert(0,"/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred/src")
from seizure_prediction.features import normalize_window
from scipy import stats
ROOT=Path("/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred"); FD=ROOT/"data/interim/chunk_features"
CH=1280; NC=540
TP=[5,14,23]   # log_total_power per channel

# ---------- CLAIM 5a: dead-chunk prevalence over sampled recordings
shard=pd.read_csv(ROOT/"data/interim/manifests/processed_shard_manifest.csv")
rids=sorted({p.name.replace("_features.npy","") for p in FD.glob("*_features.npy")})
rng=np.random.default_rng(7)
samp=list(rng.choice(rids,size=400,replace=False))
tot=0; dead=0; recs_with=0; allv=[]
for rid in samp:
    b=np.load(FD/f"{rid}_features.npy",mmap_mode="r"); av=np.load(FD/f"{rid}_availability.npy")
    v=[]
    for c,col in enumerate(TP):
        if not av[col]: continue
        x=np.asarray(b[:,col]); v.append(x)
    if not v: continue
    v=np.concatenate(v); tot+=v.size; d=(v<-20).sum(); dead+=d
    if d>0: recs_with+=1
    if len(allv)<40: allv.append(v[::37])
allv=np.concatenate(allv)
print("CLAIM5a present-channel chunks scanned=%d  frac log_total_power<-20 = %.4f%%  recordings_with_any=%d/400 (%.1f%%)"%(
    tot, 100*dead/tot, recs_with, 100*recs_with/400))
print("  percentiles of log_total_power (subsample n=%d): p1=%.2f p99=%.2f min=%.2f"%(
    allv.size,np.percentile(allv,1),np.percentile(allv,99),allv.min()))
print("  log(1e-12)=%.4f"%np.log(1e-12))

# ---------- decision windows: claims 4 and 5b
dec=pd.read_csv(ROOT/"data/interim/manifests/decision_manifest.csv",
   usecols=["recording_id","subject","decision_end_sample","label","split"])
pos=dec[dec.label==1].sample(n=2000,random_state=3)
neg=dec[dec.label==0].sample(n=6000,random_state=3)
S=pd.concat([pos,neg])
res=[]
banks={}
for rid,g in S.groupby("recording_id"):
    f=FD/f"{rid}_features.npy"
    if not f.exists(): continue
    b=np.load(f,mmap_mode="r"); av=np.load(FD/f"{rid}_availability.npy")
    for _,r in g.iterrows():
        stop=int(r.decision_end_sample)//CH; start=stop-NC
        if start<0 or stop>b.shape[0]: continue
        w=np.asarray(b[start:stop],dtype=np.float32)
        n=normalize_window(w,av)
        q=np.percentile(w,[25,75],axis=0); iqr=(q[1]-q[0])[av]
        pres=[c for c in TP if av[c]]
        deadfrac=max(((w[:,c]<-20).mean()) for c in pres) if pres else np.nan
        res.append(dict(label=int(r.label), subject=r.subject, rid=rid,
                        maxabs=float(np.abs(n).max()), min_iqr=float(iqr.min()),
                        deadfrac=deadfrac))
R=pd.DataFrame(res); R.to_csv(ROOT/"outputs/analysis/refute6/windows.csv",index=False)
print("\nCLAIM4 n=%d windows (pos=%d neg=%d)"%(len(R),(R.label==1).sum(),(R.label==0).sum()))
print("  max|normalized| : median=%.2f p99=%.2f max=%.3e"%(R.maxabs.median(),R.maxabs.quantile(.99),R.maxabs.max()))
print("  present-channel IQR<1e-6 : %d/%d (%.3f%%)"%((R.min_iqr<1e-6).sum(),len(R),100*(R.min_iqr<1e-6).mean()))
print("  max|val|>100 : %d (%.3f%%)   >1000 : %d"%((R.maxabs>100).sum(),100*(R.maxabs>100).mean(),(R.maxabs>1000).sum()))
print("  IQR<1e-6 by label: pos=%.3f%% neg=%.3f%%"%(100*R[R.label==1].min_iqr.lt(1e-6).mean(),100*R[R.label==0].min_iqr.lt(1e-6).mean()))
print("  min_iqr percentiles: p0.1=%.3g p1=%.3g p50=%.3g"%(R.min_iqr.quantile(.001),R.min_iqr.quantile(.01),R.min_iqr.median()))

print("\nCLAIM5b dead-chunk windows")
for th,lab in [(0.0,">0%"),(0.99,">=99%")]:
    p=(R[R.label==1].deadfrac>th).mean()*100; n=(R[R.label==0].deadfrac>th).mean()*100
    a=(R[R.label==1].deadfrac>th).sum(); c=(R[R.label==0].deadfrac>th).sum()
    npos=(R.label==1).sum(); nneg=(R.label==0).sum()
    odd,pv=stats.fisher_exact([[a,npos-a],[c,nneg-c]])
    print("  %-6s pos=%.3f%% (%d/%d)  neg=%.3f%% (%d/%d)  fisher p=%.4f"%(lab,p,a,npos,n,c,nneg,pv))
# subject-level: are dead windows concentrated in few subjects?
d=R[R.deadfrac>0.99]
print("  windows >=99%% dead come from %d subjects / %d recordings; total %d"%(d.subject.nunique(),d.rid.nunique(),len(d)))
print("  max AUC advantage obtainable from a >=99%%-dead indicator = %.5f"%(0.5*((R[R.label==0].deadfrac>0.99).mean()-(R[R.label==1].deadfrac>0.99).mean())))
