import numpy as np, pandas as pd, pickle, sys
from pathlib import Path
sys.path.insert(0,"/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred/src")
from seizure_prediction.features import normalize_window
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
R=Path("/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred/outputs/analysis/refute6")
C=pickle.load(open(R/"cache.pkl","rb"))
pre=C["pre"]; meta=C["meta"]
AV=np.stack([m["av"] for m in meta]); S=np.array([m["subject"] for m in meta])
vig=np.array([str(m["vigilance"]) for m in meta])
print("vigilance counts:", pd.Series(vig).value_counts().to_dict())
keep=np.isin(vig,["asleep","awake"])
W=pre[keep]; A=AV[keep]; g=S[keep]; y=(vig[keep]=="asleep").astype(int)
print("n=%d asleep=%d awake=%d subjects=%d"%(len(y),y.sum(),(1-y).sum(),len(set(g))))
df=pd.DataFrame(dict(s=g,y=y)); mix=df.groupby("s").y.nunique()
print("subjects with BOTH asleep and awake seizures: %d/%d ; seizures in mixed subjects: %d/%d"%(
    (mix==2).sum(),len(mix), df[df.s.isin(mix[mix==2].index)].shape[0], len(df)))

def normW(W,A): return np.stack([normalize_window(W[i],A[i]) for i in range(len(W))])
Wn=normW(W,A)
# gain-invariant relative features: band power minus that channel's total power (per chunk)
def relative(w):        # (nc,27) -> same shape, band cols replaced by ratio to total
    o=w.copy()
    for c in range(3):
        b=c*9; tot=w[:,b+5]
        for k in range(5): o[:,b+k]=w[:,b+k]-tot
        o[:,b+8]=w[:,b+8]-0.5*tot     # line length is ~amplitude -> log LL - 0.5 log P
        o[:,b+5]=0.0                  # drop absolute level entirely
    return o
Wr=np.stack([relative(W[i])*A[i] for i in range(len(W))])

def cv(X,y,g):
    oof=np.zeros(len(y),float); aucs=[]
    for tr,te in GroupKFold(n_splits=5).split(X,y,groups=g):
        clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,C=1.0))
        clf.fit(X[tr],y[tr]); p=clf.predict_proba(X[te])[:,1]; oof[te]=p
        if len(np.unique(y[te]))>1: aucs.append(roc_auc_score(y[te],p))
    return roc_auc_score(y,oof), float(np.mean(aucs)), oof
def clus_ci(y,s,g,n=1500,seed=0):
    rng=np.random.default_rng(seed); subs=np.unique(g); out=[]
    for _ in range(n):
        pick=rng.choice(subs,len(subs),True)
        idx=np.concatenate([np.where(g==p)[0] for p in pick])
        if len(np.unique(y[idx]))<2: continue
        out.append(roc_auc_score(y[idx],s[idx]))
    return np.percentile(out,[2.5,97.5])

pools={"mean":lambda w:w.mean(0),"max":lambda w:w.max(0),
       "mean+std":lambda w:np.r_[w.mean(0),w.std(0)],
       "q10q50q90":lambda w:np.r_[np.percentile(w,10,axis=0),np.percentile(w,50,axis=0),np.percentile(w,90,axis=0)]}
for label,BANK in [("RAW (no normalize_window)",W),("NORMALIZED (features.normalize_window)",Wn),
                   ("GAIN-INVARIANT band ratios, no window centring",Wr)]:
    print("\n--- %s"%label)
    for pn,fn in pools.items():
        X=np.stack([fn(w) for w in BANK])
        a,mf,oof=cv(X,y,g); lo,hi=clus_ci(y,oof,g)
        print("   %-10s pooledOOF=%.4f  meanfold=%.4f  subj-clustered95CI=[%.3f,%.3f]"%(pn,a,mf,lo,hi))
