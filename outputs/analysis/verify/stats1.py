import pandas as pd, numpy as np
from scipy import stats
rng=np.random.default_rng(0)
d=pd.read_csv('outputs/analysis/verify/recon.csv')
d['subject']=d.subject.astype(str)
print('=== marginal chi-square (analyst style, seizures treated as independent) ===')
def chi(col):
    t=pd.crosstab(d[col],d.actual)
    c2,p,dof,ex=stats.chi2_contingency(t)
    n=t.values.sum(); V=np.sqrt(c2/(n*(min(t.shape)-1)))
    return c2,dof,p,V,(ex<5).mean(),t.shape
for col in ['event_type','loc','vigilance','lat']:
    c2,dof,p,V,frac,shape=chi(col)
    print(f'{col:14s} chi2={c2:8.2f} dof={dof:3d} p={p:.3g} V={V:.3f} shape={shape} frac_expected<5={frac:.2f}')
print()
print('=== within-patient permutation (patient-conditional, preserves each patient\'s eligible count) ===')
subj=d.subject.values; elig=d.actual.values
groups={s:np.where(subj==s)[0] for s in np.unique(subj)}
def stat_chi(col, y):
    t=pd.crosstab(d[col],y)
    if t.shape[1]<2: return 0.0
    return stats.chi2_contingency(t)[0]
NP=2000
for col in ['event_type','loc','vigilance','lat']:
    obs=stat_chi(col,elig)
    null=np.empty(NP)
    for b in range(NP):
        y=elig.copy()
        for s,idx in groups.items():
            y[idx]=rng.permutation(y[idx])
        null[b]=stat_chi(col,y)
    p=(1+ (null>=obs).sum())/(NP+1)
    print(f'{col:14s} obs_chi2={obs:8.2f}  perm p={p:.4f}   null mean={null.mean():.1f}')
print()
print('=== how much does each attribute vary WITHIN patient? ===')
for col in ['event_type','loc','vigilance','lat']:
    g=d.groupby('subject')[col].nunique()
    print(f'{col:14s} patients with >1 level: {(g>1).sum()}/{len(g)}  mean levels/patient={g.mean():.2f}')
