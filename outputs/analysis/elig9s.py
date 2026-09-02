import pandas as pd, numpy as np, glob, os
from scipy import stats
R='/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'
m=pd.read_csv(f'{R}/data/interim/manifests/seizure_manifest.csv')
d=pd.read_csv(f'{R}/outputs/analysis/elig_reasons.csv')
m=m.merge(d[['seizure_id','gap_prev_end_min']],on='seizure_id')
m=m[m.eligible_for_prediction]
files=sorted(glob.glob(f"{R}/outputs/cv/*/out_of_fold_predictions.csv"))[:12]
res=[]
for f in files:
    p=pd.read_csv(f,usecols=['label','probability','target_seizure_id','subject'])
    pos=p[p.label==1]
    if pos.empty: continue
    neg=np.sort(p.loc[p.label==0,'probability'].values)
    # within-subject percentile: rank each seizure's max prob among that subject's own negatives
    out=[]
    for subj,gg in p.groupby('subject'):
        nn=np.sort(gg.loc[gg.label==0,'probability'].values)
        if len(nn)<50: continue
        sm=gg[gg.label==1].groupby('target_seizure_id')['probability'].max()
        for sid,v in sm.items():
            out.append((subj,sid,np.searchsorted(nn,v)/len(nn)))
    o=pd.DataFrame(out,columns=['subject','sid','pct_within'])
    o=o.merge(m[['seizure_id','gap_prev_end_min']],left_on='sid',right_on='seizure_id')
    o['has_prior']=o.gap_prev_end_min.notna()
    o['cfg']=os.path.basename(os.path.dirname(f))
    res.append(o)
a=pd.concat(res)
# keep only subjects that have BOTH kinds
both=a.groupby('subject')['has_prior'].nunique()
a2=a[a.subject.isin(both[both==2].index)]
print("subjects with both isolated and prior-seizure eligible seizures:",a2.subject.nunique(),
      " seizures:",a2.sid.nunique())
g=a2.groupby(['cfg','subject','has_prior'])['pct_within'].mean().unstack()
g=g.dropna()
diff=(g[True]-g[False]).groupby(level=0).mean()   # per config, mean within-subject diff
t=stats.ttest_1samp(diff,0)
print("within-subject percentile: has_prior - isolated = %+.4f, paired over %d configs, p=%.3g"%(diff.mean(),len(diff),t.pvalue))
print("  per-config diff >0 in %d/%d configs"%((diff>0).sum(),len(diff)))
