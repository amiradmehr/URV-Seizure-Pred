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
av=C[['avail_L','avail_R','avail_X']].to_numpy()
th=[f'{c}::log_power_theta' for c in ['BTE_LEFT','BTE_RIGHT','CROSS_HEAD']]
C['theta']=(C[th].to_numpy()*av).sum(1)/av.sum(1)
sz=pd.read_csv('outputs/analysis/_sz_with_rid.csv'); sz=sz[sz.has_feat]

# time-to-nearest-FOLLOWING-onset for preictal chunks + time SINCE previous seizure END
onsets={}; ends={}
for rid,g in sz.groupby('recording_id'):
    onsets[rid]=np.sort(g.onset_seconds.to_numpy())
    ends[rid]=np.sort((g.onset_seconds+g.duration_seconds).to_numpy())
pre=C[C.cls=='preictal'].copy()
t=pre.chunk.to_numpy()*5.0+2.5
tto=np.empty(len(pre)); tsince=np.empty(len(pre))
rids=pre.recording_id.to_numpy()
for i,(rid,tt) in enumerate(zip(rids,t)):
    o=onsets[rid]; nxt=o[o>tt]; tto[i]=nxt.min()-tt if len(nxt) else np.inf
    e=ends[rid]; prv=e[e<=tt]; tsince[i]=tt-prv.max() if len(prv) else np.inf
pre['tto']=tto; pre['tsince']=tsince
ctlmu=C[C.cls=='control'].theta.mean(); ictmu=C[C.cls=='ictal'].theta.mean()
print('control mean theta %.4f  ictal dev %+.4f'%(ctlmu,ictmu-ctlmu))
bins=[(0,30),(30,60),(60,120),(120,180),(180,300),(300,420),(420,600)]
print(f'{"bin":>10} {"n":>6} {"dev":>8} {"se":>7} {"t":>7} {"p":>10}   |  {"dev|no prior sz<20min":>10} {"n":>6} {"p":>9}')
for a,b in bins:
    m=(pre.tto>=a)&(pre.tto<b); x=pre.loc[m,'theta'].to_numpy()-ctlmu
    se=x.std(ddof=1)/np.sqrt(len(x)); tt,pp=stats.ttest_1samp(x,0)
    m2=m&(pre.tsince>1200); x2=pre.loc[m2,'theta'].to_numpy()-ctlmu
    p2=stats.ttest_1samp(x2,0).pvalue if len(x2)>3 else np.nan
    print(f'[{a:>3},{b:>3}) {len(x):>6} {x.mean():+8.4f} {se:7.4f} {tt:7.2f} {pp:10.2e}   |  {x2.mean():+8.4f} {len(x2):>6} {p2:9.2e}')
print('frac of preictal chunks whose recording had a seizure END within 20 min before: %.3f'%(pre.tsince<=1200).mean())

# preictal classifier, matched, + per-recording (leak-free) evaluation
rng=np.random.default_rng(1); keep=[]
for rid,g in D.groupby('recording_id'):
    ni=(g.cls=='ictal').sum(); pi=g.index[g.cls=='preictal'].to_numpy()
    keep.append(rng.choice(pi,size=min(len(pi),5*ni),replace=False))
si=np.concatenate(keep)
Cm=pd.concat([C[C.cls!='preictal'],C.loc[si]])
d=Cm[Cm.cls.isin(['preictal','control'])].copy()
X=d[NAMES].to_numpy(np.float64); y=(d.cls=='preictal').to_numpy(int); g=d.subject.to_numpy()
for nm,mk in [('logreg',lambda:make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000))),
              ('hgb',lambda:HistGradientBoostingClassifier(max_iter=200,random_state=0))]:
    p=np.zeros(len(y))
    for tr,te in GroupKFold(5).split(X,y,g):
        m=mk(); m.fit(X[tr],y[tr]); p[te]=m.predict_proba(X[te])[:,1]
    d['p_'+nm]=p
    prev=y.mean()
    # pooled
    print(f'PREICTAL centred {nm}: pooled AUC={roc_auc_score(y,p):.4f} AP/ch={average_precision_score(y,p)/prev:.3f} (max {1/prev:.2f}) prev={prev:.3f}')
    # leak-free: AUC computed WITHIN each recording, then aggregated
    aa=[]
    for rid,gg in d.groupby('recording_id'):
        yy=(gg.cls=='preictal').astype(int)
        if yy.sum()>=5 and (1-yy).sum()>=5: aa.append(roc_auc_score(yy,gg['p_'+nm]))
    aa=np.array(aa)
    print(f'   within-recording AUC: n={len(aa)} median={np.median(aa):.4f} mean={aa.mean():.4f} '
          f'ttest vs 0.5 p={stats.ttest_1samp(aa,0.5).pvalue:.2e}')
# how variable is the preictal:control ratio across recordings? (what HGB can exploit between recordings)
r=d.groupby('recording_id').cls.apply(lambda s:(s=='preictal').mean())
print('per-recording preictal prevalence: p5=%.3f median=%.3f p95=%.3f  (a between-recording label-rate cue)'%
      (r.quantile(.05),r.median(),r.quantile(.95)))
