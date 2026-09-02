import pandas as pd, numpy as np
from scipy import stats
R='/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'
m=pd.read_csv(f'{R}/data/interim/manifests/seizure_manifest.csv')
d=pd.read_csv(f'{R}/outputs/analysis/elig_reasons.csv')
m=m.merge(d[['seizure_id','clust','short_start','nonbg']],on='seizure_id')
m['elig']=m.eligible_for_prediction.astype(bool)
print("duration by cluster status:")
print(m.groupby('clust')['duration_seconds'].describe()[['count','25%','50%','75%','mean']])
u,p=stats.mannwhitneyu(m.loc[~m.clust,'duration_seconds'],m.loc[m.clust,'duration_seconds'])
print("MWU nonclustered vs clustered duration p=%.3g A=%.3f"%(p,u/((~m.clust).sum()*m.clust.sum())))
# within non-clustered, does duration still differ by eligibility?
s=m[~m.clust]
u,p=stats.mannwhitneyu(s.loc[s.elig,'duration_seconds'],s.loc[~s.elig,'duration_seconds'])
print("within NON-clustered: elig med=%.0f inelig med=%.0f p=%.3g"%(s.loc[s.elig,'duration_seconds'].median(),s.loc[~s.elig,'duration_seconds'].median(),p))
print("\ntop-10 patients by annotated seizure count:")
g=m.groupby('subject').agg(annot=('elig','size'),elig=('elig','sum'),clustered=('clust','sum'))
g['pct']=(100*g.elig/g.annot).round(1)
print(g.sort_values('annot',ascending=False).head(10).to_string())
print("\npatients losing ALL seizures (n=%d), their annotated counts:"%(g.elig==0).sum())
print(sorted(g.loc[g.elig==0,'annot'].tolist()))
print("\nsemiology classes near-eliminated (>=20 annotated, <=30%% retained):")
vc=m.groupby('event_type').agg(n=('elig','size'),e=('elig','sum'))
vc['pct']=(100*vc.e/vc.n).round(1)
print(vc[(vc.n>=20)].sort_values('pct').to_string())
