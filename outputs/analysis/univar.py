import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
NAMES=json.load(open('data/interim/chunk_features/feature_names.json'))
D=pd.read_pickle('outputs/analysis/detection_chunks.pkl')
avmap={'BTE_LEFT':'avail_L','BTE_RIGHT':'avail_R','CROSS_HEAD':'avail_X'}
C=D.copy(); C[NAMES]=D.groupby('recording_id')[NAMES].transform(lambda x:x-x.median())
def tab(src,pos):
    d=src[src.cls.isin([pos,'control'])]
    r=[]
    for f in NAMES:
        ch=f.split('::')[0]
        dd=d[d[avmap[ch]]]
        y=(dd.cls==pos).to_numpy(int); x=dd[f].to_numpy()
        a,b=x[y==1],x[y==0]
        sp=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
        r.append(dict(feature=f,n_pos=len(a),n_neg=len(b),
                      d=(a.mean()-b.mean())/sp if sp>0 else 0.0,
                      auc=roc_auc_score(y,x)))
    return pd.DataFrame(r).sort_values('auc',key=lambda s:(s-0.5).abs(),ascending=False)
t1=tab(D,'ictal'); t2=tab(C,'ictal'); t3=tab(C,'preictal')
t1.to_csv('outputs/analysis/univar_ictal_raw.csv',index=False)
t2.to_csv('outputs/analysis/univar_ictal_centred.csv',index=False)
t3.to_csv('outputs/analysis/univar_preictal_centred.csv',index=False)
pd.set_option('display.width',200)
print('=== ICTAL vs CONTROL, raw ==='); print(t1.to_string(index=False))
print('\n=== ICTAL vs CONTROL, per-recording median-centred ==='); print(t2.head(10).to_string(index=False))
print('\n=== PREICTAL vs CONTROL, centred (top 10) ==='); print(t3.head(10).to_string(index=False))
