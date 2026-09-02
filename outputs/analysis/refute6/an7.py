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
pre,inter,meta,imeta=C["pre"],C["inter"],C["meta"],C["imeta"]
AVp=np.stack([m["av"] for m in meta]); AVi=np.stack([m["av"] for m in imeta])
Sp=np.array([m["subject"] for m in meta]); Si=np.array([m["subject"] for m in imeta])
vig=np.array([str(m["vigilance"]) for m in meta])
def nw(W,A): return np.stack([normalize_window(W[i],A[i]) for i in range(len(W))])

# ---- CLAIM 2 : subject-clustered CI + power
X=np.concatenate([nw(pre,AVp),nw(inter,AVi)]).mean(1)
y=np.r_[np.ones(len(pre)),np.zeros(len(inter))]; g=np.r_[Sp,Si]
oof=np.zeros(len(y))
for tr,te in GroupKFold(5).split(X,y,groups=g):
    m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)); m.fit(X[tr],y[tr])
    oof[te]=m.predict_proba(X[te])[:,1]
rng=np.random.default_rng(0); subs=np.unique(g); b=[]
for _ in range(2000):
    p=rng.choice(subs,len(subs),True); idx=np.concatenate([np.where(g==q)[0] for q in p])
    if len(np.unique(y[idx]))>1: b.append(roc_auc_score(y[idx],oof[idx]))
print("CLAIM2 mean-pool AUC=%.4f  subject-clustered 95%%CI=[%.3f,%.3f]"%(roc_auc_score(y,oof),
      np.percentile(b,2.5),np.percentile(b,97.5)))
# permutation null (labels permuted at subject level is not possible; permute within all)
perm=[]
for _ in range(400):
    yp=rng.permutation(y); perm.append(roc_auc_score(yp,oof))
print("  label-permutation null AUC sd=%.4f -> a true effect below ~%.3f AUC is undetectable here"%(
      np.std(perm),0.5+2.8*np.std(perm)))

# ---- CLAIM 3 : within-mixed-subject vigilance control
keep=np.isin(vig,["asleep","awake"]); W=pre[keep]; A=AVp[keep]; gg=Sp[keep]; yy=(vig[keep]=="asleep").astype(int)
mix=pd.DataFrame(dict(s=gg,y=yy)).groupby("s").y.nunique(); mixed=set(mix[mix==2].index)
sel=np.isin(gg,list(mixed))
for lab,BANK in [("RAW",W),("NORMALIZED",nw(W,A))]:
    Xm=BANK.mean(1)
    oof=np.zeros(len(yy))
    for tr,te in GroupKFold(5).split(Xm,yy,groups=gg):
        m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000)); m.fit(Xm[tr],yy[tr])
        oof[te]=m.predict_proba(Xm[te])[:,1]
    # AUC computed only within subjects that contain both classes (removes between-patient confound)
    num=den=0.0
    for s in mixed:
        i=np.where(gg==s)[0]; a=oof[i][yy[i]==1]; c=oof[i][yy[i]==0]
        for u in a:
            for v in c:
                den+=1; num += 1.0 if u>v else (0.5 if u==v else 0.0)
    print("CLAIM3 %-11s all-seizure AUC=%.4f | WITHIN-subject AUC (29 mixed subjects, %d pairs)=%.4f"%(
        lab,roc_auc_score(yy,oof),int(den),num/den))
