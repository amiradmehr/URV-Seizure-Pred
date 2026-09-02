import pandas as pd, numpy as np
from scipy import stats
d=pd.read_csv('outputs/analysis/verify/recon.csv'); d['subject']=d.subject.astype(str)
g=d.groupby('subject').agg(n=('seizure_id','count'),k=('actual','sum'))
bins=[(1,1),(2,3),(4,7),(8,15),(16,10**6)]
print('burden bin | patients | annotated | retained | retention')
for lo,hi in bins:
    s=g[(g.n>=lo)&(g.n<=hi)]
    print(f'{lo}-{hi}: {len(s)} / {s.n.sum()} / {s.k.sum()} = {100*s.k.sum()/s.n.sum():.1f}%')
g['frac']=g.k/g.n
rho,p=stats.spearmanr(g.n,g.frac); print('spearman rho=%.3f p=%.3g n=%d'%(rho,p,len(g)))
print('patients with 0 eligible:',(g.k==0).sum(),'of',len(g),'holding',g.loc[g.k==0,'n'].sum(),'seizures')
top5=g.sort_values('n',ascending=False).head(5)
print('top5 share annotated %.1f%%  eligible %.1f%%'%(100*top5.n.sum()/g.n.sum(),100*top5.k.sum()/g.k.sum()))
print(top5)
# NULL MODEL: is the burden-retention correlation just arithmetic?
print()
print('--- event_type retention extremes ---')
et=d.groupby('event_type').agg(n=('seizure_id','count'),k=('actual','sum'))
et['ret']=et.k/et.n
print(et.sort_values('ret').head(4)); print(et.sort_values('ret').tail(4))
print('\n--- lateralization retention ---')
la=d.groupby('lat').agg(n=('seizure_id','count'),k=('actual','sum')); la['ret']=la.k/la.n; print(la)
# decision-level: subjects with zero positives
dec=pd.read_csv('data/interim/manifests/decision_manifest.csv',dtype={'subject':str})
print('\ndecisions',len(dec),'positives',int(dec.label.sum()))
print('distinct target seizures in decisions:',dec.loc[dec.label==1,'target_seizure_id'].nunique())
el=set(d.loc[d.actual,'seizure_id'])
cov=set(dec.loc[dec.label==1,'target_seizure_id'].dropna())
print('eligible seizures with 0 positive decisions:',len(el-cov))
print('split counts:'); print(dec.groupby('split').label.agg(['size','sum']))
tr=dec[dec.split=='train']
pos=tr.groupby('subject').label.sum()
print('train subjects:',tr.subject.nunique(),'with zero positive decisions:',(pos==0).sum())
zero=set(g[g.k==0].index)
print('decisions contributed by 0-eligible patients:',len(dec[dec.subject.isin(zero)]))
