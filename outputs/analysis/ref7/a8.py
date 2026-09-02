import numpy as np, pickle, sys
sys.path.insert(0,"/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred/outputs/analysis/ref7")
from pin import normalize_window_orig as NW
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
ROOT="/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred"
C=pickle.load(open(ROOT+"/outputs/analysis/ref7/cache.pkl","rb"))
i300,inter,meta,imeta=C["i300"],C["inter"],C["meta"],C["imeta"]
AVp=np.stack([m["av"] for m in meta]); AVi=np.stack([m["av"] for m in imeta])
Sp=np.array([m["subject"] for m in meta]); Si=np.array([m["subject"] for m in imeta])
dur=np.array([m["dur"] for m in meta],float)
Np=np.stack([NW(i300[i],AVp[i]) for i in range(len(i300))])
Ni=np.stack([NW(inter[i],AVi[i]) for i in range(len(inter))])
y=np.r_[np.ones(len(Np)),np.zeros(len(Ni))]; g=np.r_[Sp,Si]; W=np.concatenate([Np,Ni])
def cv(X):
    oof=np.zeros(len(y))
    for tr,te in GroupKFold(n_splits=5).split(X,y,groups=g):
        c=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000)); c.fit(X[tr],y[tr])
        oof[te]=c.predict_proba(X[te])[:,1]
    return roc_auc_score(y,oof)
print("Which minute after onset carries the 'ictal' signal the analyst attributes to last-60 pooling?")
print(" (window ends at onset+300s; slice -60:-48 = min 0-1 post onset, ... -12: = min 4-5)")
for a,b,lab in [(-60,-48,"min 0-1 (contains the ictal, median dur 45 s)"),
                (-48,-36,"min 1-2 (postictal for 82% of seizures)"),
                (-36,-24,"min 2-3 (postictal)"),(-24,-12,"min 3-4 (postictal)"),
                (-12,None,"min 4-5 (postictal for 97% of seizures)")]:
    X=np.stack([w[a:b].mean(0) for w in W]); print("   %-46s AUC=%.4f"%(lab,cv(X)))
print(" fraction of eligible seizures still ictal at +1 min: %.3f ; at +2 min: %.3f ; at +4 min: %.3f"%(
   (dur>=60).mean(),(dur>=120).mean(),(dur>=240).mean()))
