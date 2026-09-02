import numpy as np, pandas as pd, pickle, sys, json
sys.path.insert(0,"/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred/src")
from seizure_prediction.features import normalize_window
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy import stats
ROOT="/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred"
C=pickle.load(open(ROOT+"/outputs/analysis/ref7/cache.pkl","rb"))
pre,i300,ioff,inter=C["pre"],C["i300"],C["ioff"],C["inter"]
meta,imeta=C["meta"],C["imeta"]
FN=json.load(open(ROOT+"/data/interim/chunk_features/feature_names.json"))
AVp=np.stack([m["av"] for m in meta]); AVi=np.stack([m["av"] for m in imeta])
Sp=np.array([m["subject"] for m in meta]); Si=np.array([m["subject"] for m in imeta])
dur=np.array([m["dur"] for m in meta],float)
print("A. seizure duration (eligible, n=%d): median=%.1fs mean=%.1f p90=%.1f frac>=300s=%.4f"%(
  len(dur),np.median(dur),dur.mean(),np.percentile(dur,90),(dur>=300).mean()))
print("   ictal chunks inside the 540-chunk window: median=%.1f (=%.0fx dilution, not 9x)"%(
  np.median(dur)/5, 540/max(np.median(dur)/5,1e-9)))
print("   frac of seizures where the LAST 12 chunks of the onset+300s window are still ictal: %.4f"%((dur>=240).mean()))
def norm(W,AV): return np.stack([normalize_window(W[i],AV[i]) for i in range(len(W))])
POOL={"mean":lambda w:w.mean(0),"max":lambda w:w.max(0),"p95":lambda w:np.percentile(w,95,axis=0),
 "top10mean":lambda w:np.sort(w,axis=0)[-54:].mean(0),
 "last60":lambda w:w[-60:].mean(0),"last120":lambda w:w[-120:].mean(0),"last12":lambda w:w[-12:].mean(0),
 "last60+contrast":lambda w:np.concatenate([w[-60:].mean(0),w[-60:].mean(0)-w[:-60].mean(0)])}
def cv(X,y,g,seed=0):
    oof=np.zeros(len(y)); aucs=[]
    for tr,te in GroupKFold(n_splits=5).split(X,y,groups=g):
        clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,C=1.0))
        clf.fit(X[tr],y[tr]); p=clf.predict_proba(X[te])[:,1]; oof[te]=p
        if len(np.unique(y[te]))>1: aucs.append(roc_auc_score(y[te],p))
    return roc_auc_score(y,oof),float(np.mean(aucs)),float(np.std(aucs)),oof
def ci(y,s,n=1500,seed=0):
    rng=np.random.default_rng(seed); ip=np.where(y==1)[0]; ineg=np.where(y==0)[0]; o=[]
    for _ in range(n):
        a=rng.choice(ip,len(ip),True); b=rng.choice(ineg,len(ineg),True)
        o.append(roc_auc_score(np.r_[np.ones(len(a)),np.zeros(len(b))],np.r_[s[a],s[b]]))
    return np.percentile(o,[2.5,97.5])
Ni=norm(inter,AVi); Ri=inter
for tag,P in [("ICTAL onset+300s (analyst claim1)",i300),("ICTAL ends at seizure OFFSET (true pos-control)",ioff),("PREICTAL onset-60s (claim2)",pre)]:
    for mode in ["normalized","raw"]:
        Wp = norm(P,AVp) if mode=="normalized" else P
        Wi = Ni if mode=="normalized" else Ri
        W=np.concatenate([Wp,Wi]); y=np.r_[np.ones(len(Wp)),np.zeros(len(Wi))]; g=np.r_[Sp,Si]
        line=[]
        for nm,f in POOL.items():
            X=np.stack([f(w) for w in W])
            a,mf,sf,oof=cv(X,y,g); lo,hi=ci(y,oof)
            line.append("%s=%.4f[%.3f,%.3f]"%(nm,a,lo,hi))
        print("\n%s  [%s] n_pos=%d n_neg=%d"%(tag,mode,len(Wp),len(Wi)))
        print("   "+"  ".join(line))
