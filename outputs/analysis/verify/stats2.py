import pandas as pd, numpy as np
from scipy import stats
rng=np.random.default_rng(1)
d=pd.read_csv('outputs/analysis/verify/recon.csv'); d['subject']=d.subject.astype(str)
d['n_ann']=d.groupby('subject').seizure_id.transform('count')

print('--- claim2 continuous: onset time & duration ---')
for col in ['onset','dur']:
    a=d.loc[d.actual,col]; b=d.loc[~d.actual,col]
    u,p=stats.mannwhitneyu(a,b); A=u/(len(a)*len(b))
    print(f'{col}: marginal MWU p={p:.3g} A={A:.3f} medians {a.median():.1f} vs {b.median():.1f}')
# within-patient permutation for continuous
subj=d.subject.values; elig=d.actual.values; groups={s:np.where(subj==s)[0] for s in np.unique(subj)}
for col in ['onset','dur']:
    v=d[col].values.astype(float)
    def st(y): 
        a=v[y]; b=v[~y]
        return stats.mannwhitneyu(a,b)[0]/(len(a)*len(b))
    obs=st(elig); null=np.empty(3000)
    for i in range(3000):
        y=elig.copy()
        for s,idx in groups.items(): y[idx]=rng.permutation(y[idx])
        null[i]=st(y)
    p=2*min((null>=obs).mean(),(null<=obs).mean())
    print(f'{col}: within-patient perm A={obs:.3f} null mean A={null.mean():.3f} two-sided p={p:.4f}')

print('\n--- claim3 vigilance ---')
t=pd.crosstab(d.vigilance,d.actual); print(t)
aw=d[d.vigilance=='awake']; asl=d[d.vigilance=='asleep']
tab=[[asl.actual.sum(),(~asl.actual).sum()],[aw.actual.sum(),(~aw.actual).sum()]]
orr,p=stats.fisher_exact(tab); print('asleep vs awake Fisher OR(asleep elig)=%.3f p=%.3g'%(orr,p))
# within-patient (Mantel-Haenszel / conditional permutation) restricted to patients with both
sub=d[d.vigilance.isin(['awake','asleep'])].copy()
ok=sub.groupby('subject').vigilance.nunique(); both=ok[ok>1].index
s2=sub[sub.subject.isin(both)]
print('patients with both awake+asleep seizures:',len(both),'n seizures',len(s2))
# CMH-style: sum of a - E[a] over strata
def cmh(df,y):
    num=0; var=0
    for s,g in df.groupby('subject'):
        a=((g.vigilance=='asleep')&y[g.index]).sum()
        n=len(g); n1=(g.vigilance=='asleep').sum(); m1=y[g.index].sum()
        if n<2: continue
        num+= a-n1*m1/n
        var+= n1*(n-n1)*m1*(n-m1)/(n*n*(n-1)) if n>1 else 0
    return num,var
y=pd.Series(s2.actual.values,index=s2.index)
num,var=cmh(s2,y)
z=num/np.sqrt(var) if var>0 else np.nan
print('CMH (stratified by patient) z=%.3f p=%.4g  excess asleep-eligible=%.2f'%(z,2*stats.norm.sf(abs(z)),num))
# retention within those patients
print(pd.crosstab(s2.vigilance,s2.actual))

print('\n--- claim3 mechanism: clustered / short by vigilance ---')
for col in ['clust','short','nonbg']:
    print(col); print(pd.crosstab(d.vigilance,d[col],normalize='index').round(3))
