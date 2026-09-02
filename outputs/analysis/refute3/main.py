import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats

NAMES=json.load(open('data/interim/chunk_features/feature_names.json'))
D=pd.read_pickle('outputs/analysis/detection_chunks.pkl')
C=D.copy(); C[NAMES]=D.groupby('recording_id')[NAMES].transform(lambda x:x-x.median())
avmap={'BTE_LEFT':'avail_L','BTE_RIGHT':'avail_R','CROSS_HEAD':'avail_X'}
POW=[f for f in NAMES if ('log_power' in f or 'log_total' in f)]
BANDF=[f for f in NAMES if 'log_power_' in f]
HJ=[f for f in NAMES if 'hjorth' in f]
LL=[f for f in NAMES if 'line_length' in f]
print('== feature groups: pow',len(POW),'band',len(BANDF),'hj',len(HJ),'ll',len(LL))

def oof(d, cols, model='logreg'):
    X=d[cols].to_numpy(np.float64); y=(d.cls==d.attrs['pos']).to_numpy(int); g=d.subject.to_numpy()
    p=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(X,y,g):
        m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)) if model=='logreg' else HistGradientBoostingClassifier(max_iter=200,random_state=0)
        m.fit(X[tr],y[tr]); p[te]=m.predict_proba(X[te])[:,1]
    prev=y.mean()
    return dict(auc=roc_auc_score(y,p),ap=average_precision_score(y,p),prev=prev,
                ap_ch=average_precision_score(y,p)/prev, apmax_ch=1.0/prev, n=len(y), npos=int(y.sum())), p, y

# ---------- A. ictal vs control: how much is pure amplitude? ----------
di=C[C.cls.isin(['ictal','control'])].copy(); di.attrs['pos']='ictal'
res={}
res['all27'],p_all,y = oof(di,NAMES)
res['hjorth6'],_,_    = oof(di,HJ)
res['linelen3'],_,_   = oof(di,LL)
# single amplitude summary: mean log total power over AVAILABLE channels
tot=[f for f in NAMES if 'log_total_power' in f]
av=di[['avail_L','avail_R','avail_X']].to_numpy()
T=di[tot].to_numpy(); amp=(T*av).sum(1)/av.sum(1)
di['amp']=amp
res['amp1'],_,_=oof(di,['amp'])
# spectral SHAPE only: band power minus that channel's total power (removes broadband gain per chunk)
for ch in avmap:
    for b in ['delta','theta','alpha','beta','gamma']:
        di[f'rel_{ch}_{b}']=di[f'{ch}::log_power_{b}']-di[f'{ch}::log_total_power']
REL=[c for c in di.columns if c.startswith('rel_')]
res['shape15'],_,_=oof(di,REL)
res['shape15+hj6'],_,_=oof(di,REL+HJ)
for k,v in res.items(): print(f'ICTAL {k:14s} AUC={v["auc"]:.4f} AP={v["ap"]:.4f} AP/ch={v["ap_ch"]:.3f} (max {v["apmax_ch"]:.2f}) n={v["n"]}')

# correlation among band powers within ictal+control (is it one broadband factor?)
sub=di[di.avail_X][[f'CROSS_HEAD::log_power_{b}' for b in ['delta','theta','alpha','beta','gamma']]]
print('CROSS_HEAD band-power corr matrix (centred):\n', np.round(np.corrcoef(sub.to_numpy().T),3))

# ---------- B. per-recording heterogeneity of the FULL model ----------
di['p_all']=p_all
rows=[]
for rid,g in di.groupby('recording_id'):
    yy=(g.cls=='ictal').astype(int)
    if yy.sum()>=2 and (1-yy).sum()>=5: rows.append((rid,roc_auc_score(yy,g.p_all),int(yy.sum())))
PR=pd.DataFrame(rows,columns=['rid','auc','n_ict'])
print(f'per-recording AUC of FULL model: n={len(PR)} median={PR.auc.median():.3f} mean={PR.auc.mean():.3f} frac>0.7={(PR.auc>0.7).mean():.3f}')
print('n_ict distribution:', PR.n_ict.describe().to_dict())
print('per-rec AUC by n_ict tercile:', PR.groupby(pd.qcut(PR.n_ict,3,duplicates="drop")).auc.median().to_dict())

# ---------- C. claim 4 reproduction: single mean-theta feature ----------
th=[f'{c}::log_power_theta' for c in avmap]
di['theta']=(di[th].to_numpy()*av).sum(1)/av.sum(1)
rows=[]
for rid,g in di.groupby('recording_id'):
    yy=(g.cls=='ictal').astype(int)
    if yy.sum()>=2 and (1-yy).sum()>=5: rows.append((rid,roc_auc_score(yy,g.theta),int(yy.sum()),int((1-yy).sum())))
P4=pd.DataFrame(rows,columns=['rid','auc','n_ict','n_ctl'])
print(f'CLAIM4 repro: n={len(P4)} mean={P4.auc.mean():.4f} std={P4.auc.std():.4f} p25={P4.auc.quantile(.25):.4f} median={P4.auc.median():.4f} p75={P4.auc.quantile(.75):.4f}')
print(f'  frac>0.70={(P4.auc>0.7).mean():.4f} frac>0.5={(P4.auc>0.5).mean():.4f} frac<0.5={(P4.auc<0.5).mean():.4f}')
print('  n_ict quartiles',P4.n_ict.quantile([0,.25,.5,.75,1]).to_dict())
# subject-level (recordings are NOT independent)
sm=di.groupby('recording_id').subject.first()
P4['subject']=P4.rid.map(sm)
print('  distinct subjects among 277 recordings:',P4.subject.nunique(),
      ' top-5 subject recording counts:',P4.subject.value_counts().head(5).to_dict())
print('  subject-median-of-medians:',P4.groupby("subject").auc.median().median().round(4),
      ' mean of subject medians:',P4.groupby("subject").auc.median().mean().round(4))
P4.to_csv('outputs/analysis/refute3/claim4_repro.csv',index=False)
