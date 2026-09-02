import json, sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
NAMES=json.load(open('data/interim/chunk_features/feature_names.json'))
D=pd.read_pickle('outputs/analysis/detection_chunks.pkl')
rng=np.random.default_rng(1)
C=D.copy(); C[NAMES]=D.groupby('recording_id')[NAMES].transform(lambda x:x-x.median())
keep=[]
for rid,g in D.groupby('recording_id'):
    ni=(g.cls=='ictal').sum(); pi=g.index[g.cls=='preictal'].to_numpy()
    keep.append(rng.choice(pi,size=min(len(pi),5*ni),replace=False))
sub=np.concatenate(keep)
Dm=pd.concat([D[D.cls!='preictal'],D.loc[sub]]); Cm=pd.concat([C[C.cls!='preictal'],C.loc[sub]])
def oof(src,pos,tag):
    d=src[src.cls.isin([pos,'control'])]
    X=d[NAMES].to_numpy(np.float64); y=(d.cls==pos).to_numpy(int); g=d.subject.to_numpy()
    for nm,mk in [('logreg',lambda: make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000))),
                  ('hgb',lambda: HistGradientBoostingClassifier(max_iter=100,random_state=0))]:
        p=np.zeros(len(y))
        for tr,te in GroupKFold(n_splits=5).split(X,y,g):
            m=mk(); m.fit(X[tr],y[tr]); p[te]=m.predict_proba(X[te])[:,1]
        prev=y.mean(); ap=average_precision_score(y,p)
        print(f"{tag:18s} {nm:7s} AUC={roc_auc_score(y,p):.4f} AP={ap:.4f} chance={prev:.4f} AP/chance={ap/prev:.3f} n={len(y)} npos={int(y.sum())}",flush=True)
        if nm=='logreg' and pos=='ictal':
            d.assign(prob=p)[['recording_id','subject','cls','chunk','prob']].to_csv(f'outputs/analysis/oof_{tag}.csv',index=False)
oof(D,'ictal','ictal_raw'); oof(C,'ictal','ictal_centred')
oof(Dm,'preictal','preictal_raw'); oof(Cm,'preictal','preictal_centred')
