import numpy as np, pickle, sys
sys.path.insert(0,"/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred/outputs/analysis/ref7")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
ROOT="/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred"
C=pickle.load(open(ROOT+"/outputs/analysis/ref7/cache.pkl","rb"))
pre,inter,meta,imeta=C["pre"],C["inter"],C["meta"],C["imeta"]
Sp=np.array([m["subject"] for m in meta]); Si=np.array([m["subject"] for m in imeta])
y=np.r_[np.ones(len(pre)),np.zeros(len(inter))]; g=np.r_[Sp,Si]
def oof(X):
    o=np.zeros(len(y))
    for tr,te in GroupKFold(n_splits=5).split(X,y,groups=g):
        c=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000)); c.fit(X[tr],y[tr])
        o[te]=c.predict_proba(X[te])[:,1]
    return o
def clusterboot(o,n=3000,seed=0):
    rng=np.random.default_rng(seed); subs=np.unique(g); out=[]
    for _ in range(n):
        pick=rng.choice(subs,len(subs),True)
        idx=np.concatenate([np.where(g==s)[0] for s in pick])
        if len(np.unique(y[idx]))<2: continue
        out.append(roc_auc_score(y[idx],o[idx]))
    return np.percentile(out,[2.5,97.5])
for nm,f in [("raw mean-pool",lambda w:w.mean(0)),
             ("raw [last60 mean, last60-rest]",lambda w:np.concatenate([w[-60:].mean(0),w[-60:].mean(0)-w[:-60].mean(0)]))]:
    W=np.concatenate([pre,inter]); X=np.stack([f(w) for w in W])
    o=oof(X); a=roc_auc_score(y,o); lo,hi=clusterboot(o)
    print("PREICTAL vs far-interictal, %-32s AUC=%.4f  subject-clustered 95%% CI [%.3f,%.3f]"%(nm,a,lo,hi))
