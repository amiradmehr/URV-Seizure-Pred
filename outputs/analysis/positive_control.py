import json, os, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

FD='data/interim/chunk_features'
NAMES=json.load(open(f'{FD}/feature_names.json'))
CH=5.0; GUARD=1800.0; RATIO=5; RNG=np.random.default_rng(0)

s=pd.read_csv('outputs/analysis/_sz_with_rid.csv')
s=s[s.has_feat].copy()
rows=[]
for rid,g in s.groupby('recording_id'):
    F=np.load(f'{FD}/{rid}_features.npy')
    A=np.load(f'{FD}/{rid}_availability.npy')
    n=len(F)
    if n<400: continue
    iv=[(float(r.onset_seconds),float(r.onset_seconds)+float(r.duration_seconds)) for r in g.itertuples()]
    ict=set(); pre=set()
    for (o,e) in iv:
        a=int(np.ceil(o/CH)); b=int(np.floor(e/CH))
        cc=list(range(a,b)) or [int(o//CH)]
        ict.update(c for c in cc if 0<=c<n)
        pa=int(np.ceil(max(0.0,o-600.0)/CH)); pb=int(np.floor(o/CH))
        pre.update(c for c in range(pa,pb) if 0<=c<n)
    pre-=ict
    # control: >GUARD from every seizure interval
    ok=np.ones(n,bool)
    t=np.arange(n)*CH+CH/2
    for (o,e) in iv:
        ok &= ~((t>o-GUARD)&(t<e+GUARD))
    cand=np.flatnonzero(ok)
    if len(ict)==0 or len(cand)<10: continue
    k=min(len(cand), RATIO*len(ict))
    ctl=RNG.choice(cand,size=k,replace=False)
    sub=int(g.subject.iloc[0])
    for c,lab in [(sorted(ict),'ictal'),(sorted(pre),'preictal'),(sorted(ctl),'control')]:
        if not len(c): continue
        c=np.asarray(c)
        rows.append(pd.DataFrame(F[c],columns=NAMES).assign(
            recording_id=rid,subject=sub,cls=lab,chunk=c,
            avail_L=A[0],avail_R=A[9],avail_X=A[18]))
D=pd.concat(rows,ignore_index=True)
D.to_pickle('outputs/analysis/detection_chunks.pkl')
print('rows',len(D),'recordings',D.recording_id.nunique(),'subjects',D.subject.nunique())
print(D.cls.value_counts())
print('nonfinite feature cells:',int((~np.isfinite(D[NAMES].to_numpy())).sum()))
print('avail counts L/R/X:',D.avail_L.mean(),D.avail_R.mean(),D.avail_X.mean())
