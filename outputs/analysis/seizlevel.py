import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
NAMES=json.load(open('data/interim/chunk_features/feature_names.json'))
D=pd.read_pickle('outputs/analysis/detection_chunks.pkl')
C=D.copy(); C[NAMES]=D.groupby('recording_id')[NAMES].transform(lambda x:x-x.median())
d=C[C.cls.isin(['ictal','control'])].copy()
X=d[NAMES].to_numpy(np.float64); y=(d.cls=='ictal').to_numpy(int); g=d.subject.to_numpy()
p=np.zeros(len(y))
for tr,te in GroupKFold(n_splits=5).split(X,y,g):
    m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)); m.fit(X[tr],y[tr]); p[te]=m.predict_proba(X[te])[:,1]
d['prob']=p
print('chunk-level OOF AUC %.4f'%roc_auc_score(y,p))
d.to_csv('outputs/analysis/oof_ictal_centred.csv',index=False)
# map ictal chunks back to individual seizures
sz=pd.read_csv('outputs/analysis/_sz_with_rid.csv'); sz=sz[sz.has_feat]
ic=d[d.cls=='ictal']; ct=d[d.cls=='control']
rows=[]
for r in sz.itertuples():
    a=int(np.ceil(r.onset_seconds/5)); b=int(np.floor((r.onset_seconds+r.duration_seconds)/5))
    cc=set(range(a,b)) or {int(r.onset_seconds//5)}
    m=ic[(ic.recording_id==r.recording_id)&(ic.chunk.isin(cc))]
    if len(m): rows.append(dict(seizure_id=r.seizure_id,subject=r.subject,eligible=r.eligible_for_prediction,
                                vigilance=r.vigilance,dur=r.duration_seconds,maxprob=m.prob.max(),n=len(m)))
S=pd.DataFrame(rows); S.to_csv('outputs/analysis/seizure_level_detection.csv',index=False)
print('seizures scored:',len(S))
for fa in [0.5,1.0,2.0,6.0]:
    thr=np.quantile(ct.prob,1-fa*5/3600.0)   # control chunks are 5 s; fa alarms/hour
    print(f'  FA={fa:>4} /h  thr={thr:.4f}  seizure-level sensitivity ALL={100*(S.maxprob>thr).mean():5.1f}%  '
          f'ELIGIBLE-only={100*(S[S.eligible].maxprob>thr).mean():5.1f}%  (n_elig={S.eligible.sum()})')
