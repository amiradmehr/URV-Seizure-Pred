import pandas as pd, numpy as np
from scipy import stats
rng=np.random.default_rng(1)
sz=pd.read_csv('data/interim/manifests/seizure_manifest.csv')
sz['elig']=sz.eligible_for_prediction.astype(bool); sz['subject']=sz.subject.astype(str)
sub=sz[sz.vigilance.isin(['awake','asleep'])].copy()
# CMH / stratified permutation on awake-vs-asleep 2x2
obs=pd.crosstab(sub.vigilance,sub.elig)
print("marginal 2x2\n",obs); print("OR=%.3f p=%.3g"%stats.fisher_exact(obs.values))
# Mantel-Haenszel by patient
tabs=[]
for s,g in sub.groupby('subject'):
    t=pd.crosstab(g.vigilance,g.elig).reindex(index=['asleep','awake'],columns=[False,True]).fillna(0).values
    tabs.append(t)
num=sum(t[0,1]*t[1,0]/t.sum() for t in tabs if t.sum()>0)
den=sum(t[0,0]*t[1,1]/t.sum() for t in tabs if t.sum()>0)
print("MH OR (asleep-elig vs awake-elig, patient-stratified) = %.3f"%(num/den if den else np.nan))
# stratified permutation p
def stat(lab):
    a=sub.vigilance.values=='asleep'
    return (lab[a].mean()-lab[~a].mean())
lab=sub.elig.values; idx=[np.where(sub.subject.values==s)[0] for s in sub.subject.unique()]
o=stat(lab); null=[]
for _ in range(5000):
    p=lab.copy()
    for ii in idx: p[ii]=rng.permutation(p[ii])
    null.append(stat(p))
null=np.array(null)
print("obs asleep-awake retention diff=%.4f  within-patient null mean=%.4f sd=%.4f  p=%.4f"%(
    o,null.mean(),null.std(),(1+(np.abs(null-null.mean())>=abs(o-null.mean())).sum())/(1+len(null))))
# concentration: are asleep seizures concentrated in high-burden patients?
b=sz.groupby('subject').size()
sz['burden']=sz.subject.map(b)
print("\nmedian patient burden: asleep sz %.0f, awake sz %.0f"%(
    sz.loc[sz.vigilance=='asleep','burden'].median(), sz.loc[sz.vigilance=='awake','burden'].median()))
print("mean burden asleep %.1f awake %.1f  MWU p=%.3g"%(
    sz.loc[sz.vigilance=='asleep','burden'].mean(), sz.loc[sz.vigilance=='awake','burden'].mean(),
    stats.mannwhitneyu(sz.loc[sz.vigilance=='asleep','burden'],sz.loc[sz.vigilance=='awake','burden'])[1]))
