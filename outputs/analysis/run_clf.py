import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

NAMES=json.load(open('data/interim/chunk_features/feature_names.json'))
D=pd.read_pickle('outputs/analysis/detection_chunks.pkl')
rng=np.random.default_rng(1)

# per-recording median centring (the amplitude-confound removal used in the pipeline)
C=D.copy()
C[NAMES]=D.groupby('recording_id')[NAMES].transform(lambda x: x-x.median())

def oof(df,pos,src,centred):
    d=src[src.cls.isin([pos,'control'])]
    X=d[NAMES].to_numpy(np.float64); y=(d.cls==pos).to_numpy(int); g=d.subject.to_numpy()
    out={}
    for nm,mk in [('logreg',lambda: make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,C=1.0))),
                  ('hgb',   lambda: HistGradientBoostingClassifier(max_iter=200,random_state=0))]:
        p=np.zeros(len(y))
        for tr,te in GroupKFold(n_splits=5).split(X,y,g):
            m=mk(); m.fit(X[tr],y[tr]); p[te]=m.predict_proba(X[te])[:,1]
        prev=y.mean()
        out[nm]=dict(auc=roc_auc_score(y,p),ap=average_precision_score(y,p),
                     chance=prev,ap_over_chance=average_precision_score(y,p)/prev,
                     n_pos=int(y.sum()),n=len(y))
    return out

# subsample preictal to 5x ictal per recording so the two arms are matched in size
keep=[]
for rid,g in D.groupby('recording_id'):
    ni=(g.cls=='ictal').sum(); pi=g.index[g.cls=='preictal'].to_numpy()
    keep.append(rng.choice(pi,size=min(len(pi),5*ni),replace=False))
sub_idx=np.concatenate(keep)
Dm=pd.concat([D[D.cls!='preictal'],D.loc[sub_idx]])
Cm=pd.concat([C[C.cls!='preictal'],C.loc[sub_idx]])

res={}
res['ictal_raw']=oof(None,'ictal',D,False)
res['ictal_centred']=oof(None,'ictal',C,True)
res['preictal_raw']=oof(None,'preictal',Dm,False)
res['preictal_centred']=oof(None,'preictal',Cm,True)
print(json.dumps(res,indent=2))
json.dump(res,open('outputs/analysis/clf_results.json','w'),indent=2)
