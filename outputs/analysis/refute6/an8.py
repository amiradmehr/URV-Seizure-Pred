import numpy as np, pandas as pd, pickle, sys
from pathlib import Path
sys.path.insert(0,"/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred/src")
from seizure_prediction.features import normalize_window
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
ROOT=Path("/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred"); FD=ROOT/"data/interim/chunk_features"
R=ROOT/"outputs/analysis/refute6"; C=pickle.load(open(R/"cache.pkl","rb"))
inter,imeta,meta=C["inter"],C["imeta"],C["meta"]
AVi=np.stack([m["av"] for m in imeta]); Si=np.array([m["subject"] for m in imeta])
CH=1280;NC=540
def grab(b,e):
    s=int(e)//CH; st=s-NC
    return None if st<0 or s>b.shape[0] else np.asarray(b[st:s],dtype=np.float32)
sets={"end=onset+60 (ictal only)":60.0,"end=onset+300 (analyst control)":300.0,
      "end=onset+900 (last 60 chunks = PURE postictal, 10-15min after onset)":900.0}
banks={}
for m in meta:
    if m["recording_id"] not in banks:
        banks[m["recording_id"]]=np.load(FD/f"{m['recording_id']}_features.npy",mmap_mode="r")
def nw(W,A): return np.stack([normalize_window(W[i],A[i]) for i in range(len(W))])
Ni=nw(inter,AVi)
for lab,off in sets.items():
    Ws=[];As=[];Ss=[]
    for m in meta:
        w=grab(banks[m["recording_id"]],(m["onset"]+off)*256.0)
        if w is None: continue
        Ws.append(w);As.append(m["av"]);Ss.append(m["subject"])
    Ws=nw(np.stack(Ws),np.stack(As))
    X=np.concatenate([Ws[:,-60:].mean(1),Ni[:,-60:].mean(1)])
    y=np.r_[np.ones(len(Ws)),np.zeros(len(Ni))]; g=np.r_[np.array(Ss),Si]
    oof=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(X,y,groups=g):
        c=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)); c.fit(X[tr],y[tr])
        oof[te]=c.predict_proba(X[te])[:,1]
    print("last-60-chunk mean pooling, %-68s n=%d AUC=%.4f"%(lab,len(Ws),roc_auc_score(y,oof)))
