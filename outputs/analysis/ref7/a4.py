import numpy as np, pandas as pd, pickle, sys, json
sys.path.insert(0,"/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred/outputs/analysis/ref7")
from pin import normalize_window_orig as NW
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy import stats
ROOT="/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred"
C=pickle.load(open(ROOT+"/outputs/analysis/ref7/cache.pkl","rb"))
pre,i300,inter=C["pre"],C["i300"],C["inter"]; meta,imeta=C["meta"],C["imeta"]
AVp=np.stack([m["av"] for m in meta]); AVi=np.stack([m["av"] for m in imeta])
Sp=np.array([m["subject"] for m in meta]); Si=np.array([m["subject"] for m in imeta])
vig=np.array([str(m["vigilance"]) for m in meta])
def norm(W,AV): return np.stack([NW(W[i],AV[i]) for i in range(len(W))])
def cv(X,y,g):
    oof=np.zeros(float(len(y)).__int__())
    for tr,te in GroupKFold(n_splits=5).split(X,y,groups=g):
        clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,C=1.0))
        clf.fit(X[tr],y[tr]); oof[te]=clf.predict_proba(X[te])[:,1]
    return roc_auc_score(y,oof),oof
def ci(y,s,n=1500,seed=0):
    rng=np.random.default_rng(seed); ip=np.where(y==1)[0]; ineg=np.where(y==0)[0]; o=[]
    for _ in range(n):
        a=rng.choice(ip,len(ip),True); b=rng.choice(ineg,len(ineg),True)
        o.append(roc_auc_score(np.r_[np.ones(len(a)),np.zeros(len(b))],np.r_[s[a],s[b]]))
    return np.percentile(o,[2.5,97.5])
POOL={"mean":lambda w:w.mean(0),"max":lambda w:w.max(0),"p95":lambda w:np.percentile(w,95,axis=0),
 "last60":lambda w:w[-60:].mean(0),"last12":lambda w:w[-12:].mean(0)}
print("== PINNED (pre-fix normalize_window) re-check of claims 1 & 2")
Ni=norm(inter,AVi)
for tag,P in [("ictal onset+300",i300),("preictal onset-60",pre)]:
    Wp=norm(P,AVp); W=np.concatenate([Wp,Ni]); y=np.r_[np.ones(len(Wp)),np.zeros(len(Ni))]; g=np.r_[Sp,Si]
    out=[]
    for nm,f in POOL.items():
        X=np.stack([f(w) for w in W]); a,oof=cv(X,y,g); lo,hi=ci(y,oof)
        out.append("%s=%.4f[%.3f,%.3f]"%(nm,a,lo,hi))
    print("  %-18s %s"%(tag,"  ".join(out)))

print("\n== CLAIM 3: vigilance from the 45-min PRE-ICTAL window")
keep=np.isin(vig,["asleep","awake"])
yv=(vig[keep]=="asleep").astype(int); gv=Sp[keep]
print("  n=%d (asleep=%d awake=%d) subjects=%d"%(keep.sum(),yv.sum(),(1-yv).sum(),len(set(gv))))
Praw=pre[keep]; Pn=norm(Praw,AVp[keep])
rows={}
for nm,f in [("mean",lambda w:w.mean(0)),("max",lambda w:w.max(0)),("median",lambda w:np.median(w,axis=0))]:
    for mode,W in [("raw",Praw),("normalized",Pn)]:
        X=np.stack([f(w) for w in W]); a,oof=cv(X,yv,gv); lo,hi=ci(yv,oof)
        rows[(nm,mode)]=(a,lo,hi,oof)
        print("   pool=%-7s %-11s AUC=%.4f [%.3f,%.3f]"%(nm,mode,a,lo,hi))
# is the raw advantage a between-patient confound? within-subject AUC
def within_subject_auc(oof,y,g):
    num=den=0; used=0
    for s in np.unique(g):
        m=g==s
        if len(np.unique(y[m]))<2: continue
        used+=1
        a=oof[m][y[m]==1]; b=oof[m][y[m]==0]
        num+=roc_auc_score(y[m],oof[m])*len(a)*len(b); den+=len(a)*len(b)
    return num/den, used
for nm in ["mean","max"]:
    for mode in ["raw","normalized"]:
        a,lo,hi,oof=rows[(nm,mode)]
        w,us=within_subject_auc(oof,yv,gv)
        print("   pool=%-7s %-11s WITHIN-SUBJECT AUC=%.4f (subjects with both classes: %d)"%(nm,mode,w,us))
# how patient-confounded is vigilance at all?
df=pd.DataFrame(dict(s=gv,y=yv))
pure=df.groupby("s").y.mean()
print("   subjects whose eligible seizures are ALL asleep or ALL awake: %d/%d (%.0f%%)"%(
   int(((pure==0)|(pure==1)).sum()),len(pure),100*((pure==0)|(pure==1)).mean()))
# does vigilance predict the LABEL? sweep vigilance effect check
sw=pd.read_csv(ROOT+"/outputs/sweep/results.csv")
d=sw.sens_asleep-sw.sens_awake
t,p=stats.ttest_rel(sw.sens_asleep,sw.sens_awake)
print("\n== the sweep's vigilance effect (all 42 configs used normalize=True, IQR floor 1e-6)")
print("   sens_asleep=%.4f sens_awake=%.4f paired t=%.2f p=%.5f  -> effect PRESENT despite normalisation"%(
   sw.sens_asleep.mean(),sw.sens_awake.mean(),t,p))
