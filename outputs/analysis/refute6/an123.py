import numpy as np, pandas as pd, pickle, sys, json
from pathlib import Path
sys.path.insert(0,"/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred/src")
from seizure_prediction.features import normalize_window
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy import stats
R=Path("/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred/outputs/analysis/refute6")
C=pickle.load(open(R/"cache.pkl","rb"))
ict,pre,inter,meta,imeta = C["ict"],C["pre"],C["inter"],C["meta"],C["imeta"]
AVp=np.stack([m["av"] for m in meta]); AVi=np.stack([m["av"] for m in imeta])
Sp=np.array([m["subject"] for m in meta]); Si=np.array([m["subject"] for m in imeta])
dur=np.array([m["dur"] for m in meta],float)
print("seizure duration s: median %.1f mean %.1f p75 %.1f p90 %.1f  frac>=300s %.3f"%(
    np.median(dur),dur.mean(),np.percentile(dur,75),np.percentile(dur,90),(dur>=300).mean()))
print("ictal chunks in 540 (dur/5): median %.1f -> dilution %.0fx"%(np.median(dur)/5, 540/(np.median(dur)/5)))

def norm(W,AV):
    return np.stack([normalize_window(W[i],AV[i]) for i in range(len(W))])

POOL={
 "mean":     lambda w: w.mean(0),
 "max":      lambda w: w.max(0),
 "p95":      lambda w: np.percentile(w,95,axis=0),
 "top10mean":lambda w: np.sort(w,axis=0)[-54:].mean(0),
 "last60":   lambda w: w[-60:].mean(0),
 "last120":  lambda w: w[-120:].mean(0),
 "last12":   lambda w: w[-12:].mean(0),
 "last60+contrast": lambda w: np.concatenate([w[-60:].mean(0), w[-60:].mean(0)-w[:-60].mean(0)]),
 "last12+contrast": lambda w: np.concatenate([w[-12:].mean(0), w[-12:].mean(0)-w[:-12].mean(0)]),
}
def pooled(W,name): return np.stack([POOL[name](w) for w in W])

def cv_auc(X,y,g,seed=0):
    """returns pooled-OOF auc and mean-of-fold auc"""
    oof=np.zeros(len(y)); folds=np.zeros(len(y),int); aucs=[]
    gkf=GroupKFold(n_splits=5)
    for k,(tr,te) in enumerate(gkf.split(X,y,groups=g)):
        clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,C=1.0))
        clf.fit(X[tr],y[tr]); p=clf.predict_proba(X[te])[:,1]; oof[te]=p; folds[te]=k
        if len(np.unique(y[te]))>1: aucs.append(roc_auc_score(y[te],p))
    return roc_auc_score(y,oof), float(np.mean(aucs)), float(np.std(aucs)), oof

def auc_ci(y,s,n=2000,seed=0):
    rng=np.random.default_rng(seed); y=np.asarray(y); s=np.asarray(s)
    ip=np.where(y==1)[0]; ineg=np.where(y==0)[0]; out=[]
    for _ in range(n):
        a=rng.choice(ip,len(ip),True); b=rng.choice(ineg,len(ineg),True)
        out.append(roc_auc_score(np.r_[np.ones(len(a)),np.zeros(len(b))], np.r_[s[a],s[b]]))
    return np.percentile(out,[2.5,97.5])

for tag,POS,AVP in [("ICTAL(claim1)",ict,AVp),("PREICTAL(claim2)",pre,AVp)]:
    Wn=np.concatenate([norm(POS,AVP),norm(inter,AVi)])
    y=np.r_[np.ones(len(POS)),np.zeros(len(inter))]
    g=np.r_[Sp,Si]
    print("\n=== %s  n_pos=%d n_neg=%d n_subj=%d"%(tag,len(POS),len(inter),len(set(g))))
    for name in POOL:
        X=pooled(Wn,name)
        a,mf,sf,oof=cv_auc(X,y,g)
        lo,hi=auc_ci(y,oof)
        print("  %-18s pooledOOF_AUC=%.4f [%.3f,%.3f]  meanfoldAUC=%.4f+-%.4f"%(name,a,lo,hi,mf,sf))
