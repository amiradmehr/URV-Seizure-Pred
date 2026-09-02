import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
ROOT=Path("/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred"); FD=ROOT/"data/interim/chunk_features"
CH=1280; NC=540; TP=[5,14,23]
dec=pd.read_csv(ROOT/"data/interim/manifests/decision_manifest.csv",
   usecols=["recording_id","subject","decision_end_sample","label"])
posrec=set(dec[dec.label==1].recording_id.unique())
d2=dec[dec.recording_id.isin(posrec)]
print("restricted to the %d recordings that hold >=1 positive: %d decisions, %d positive"%(
    len(posrec),len(d2),d2.label.sum()))
pos=d2[d2.label==1].sample(n=2000,random_state=5)
neg=d2[d2.label==0].sample(n=6000,random_state=5)
S=pd.concat([pos,neg]); out=[]
for rid,g in S.groupby("recording_id"):
    f=FD/f"{rid}_features.npy"
    if not f.exists(): continue
    b=np.load(f,mmap_mode="r"); av=np.load(FD/f"{rid}_availability.npy")
    pres=[c for c in TP if av[c]]
    if not pres: continue
    for _,r in g.iterrows():
        stop=int(r.decision_end_sample)//CH; start=stop-NC
        if start<0 or stop>b.shape[0]: continue
        w=np.asarray(b[start:stop],dtype=np.float32)
        out.append(dict(label=int(r.label),rid=rid,deadfrac=max((w[:,c]<-20).mean() for c in pres)))
D=pd.DataFrame(out)
print("n=%d (pos %d / neg %d)"%(len(D),(D.label==1).sum(),(D.label==0).sum()))
for th,lab in [(0.0,">0%"),(0.5,">50%"),(0.99,">=99%")]:
    a=(D[D.label==1].deadfrac>th).sum(); c=(D[D.label==0].deadfrac>th).sum()
    npos=(D.label==1).sum(); nneg=(D.label==0).sum()
    _,pv=stats.fisher_exact([[a,npos-a],[c,nneg-c]])
    print("  %-6s pos=%.3f%% (%d/%d)  neg=%.3f%% (%d/%d)  fisher p=%.4f"%(lab,100*a/npos,a,npos,100*c/nneg,c,nneg,pv))
