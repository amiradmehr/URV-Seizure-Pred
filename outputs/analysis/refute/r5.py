import pandas as pd, numpy as np
from scipy import stats
sz=pd.read_csv('data/interim/manifests/seizure_manifest.csv')
sz['elig']=sz.eligible_for_prediction.astype(bool); sz['subject']=sz.subject.astype(str)
g=sz.groupby('subject').agg(n=('elig','size'),k=('elig','sum'))
bins=[(1,1,'1'),(2,3,'2-3'),(4,7,'4-7'),(8,15,'8-15'),(16,10**6,'16+')]
for lo,hi,name in bins:
    s=g[(g.n>=lo)&(g.n<=hi)]
    print(f"{name}: patients {len(s)} annotated {s.n.sum()} retained {s.k.sum()} = {100*s.k.sum()/s.n.sum():.1f}%")
print("total patients",len(g),"annotated",g.n.sum(),"retained",g.k.sum())
rho,p=stats.spearmanr(g.n,g.k/g.n); print("spearman rho=%.3f p=%.3g n=%d"%(rho,p,len(g)))
print("patients with zero eligible:",(g.k==0).sum(), "=%.1f%%"%(100*(g.k==0).mean()),
      "seizures lost:",g.loc[g.k==0,'n'].sum())
top5=g.n.nlargest(5); print("\ntop5 by annotated:\n",g.loc[top5.index])
print("top5 share annotated %.3f  share eligible %.3f"%(top5.sum()/g.n.sum(), g.loc[top5.index,'k'].sum()/g.k.sum()))
print("\nsubject 87:",g.loc['087'].tolist() if '087' in g.index else 'NA',
      " subject 109:",g.loc['109'].tolist() if '109' in g.index else 'NA')
# decisions
d=pd.read_csv('data/interim/manifests/decision_manifest.csv',dtype={'subject':str})
print("\ndecisions",len(d),"pos",int(d.label.sum()))
print("distinct target seizures in positives:",d.loc[d.label==1,'target_seizure_id'].nunique())
tr=d[d.split=='train']
pos_by_s=tr.groupby('subject').label.sum()
print("train subjects",tr.subject.nunique(),"with zero positives",(pos_by_s==0).sum())
zs=pos_by_s[pos_by_s==0].index
print("negative decisions from those zero-positive train subjects:",int(tr[tr.subject.isin(zs)].shape[0]))
# subjects that have annotated seizures but zero eligible, in train
zero_elig=set(g[g.k==0].index)
print("decisions in train from patients with 0 eligible seizures:",int(tr[tr.subject.isin(zero_elig)].shape[0]))
