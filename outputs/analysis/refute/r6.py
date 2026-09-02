import pandas as pd, numpy as np
sz=pd.read_csv('data/interim/manifests/seizure_manifest.csv',dtype={'subject':str})
sz['elig']=sz.eligible_for_prediction.astype(bool)
g=sz.groupby('subject').agg(n=('elig','size'),k=('elig','sum'))
zero=set(g[g.k==0].index); print("zero-elig patients",len(zero),sorted(zero)[:6])
d=pd.read_csv('data/interim/manifests/decision_manifest.csv',dtype={'subject':str})
print("subject sample from decisions:",d.subject.unique()[:5])
tr=d[d.split=='train']
print("train decisions from zero-elig patients:",int(tr.subject.isin(zero).sum()))
print("all-split decisions from zero-elig patients:",int(d.subject.isin(zero).sum()))
pos=tr.groupby('subject').label.sum(); zs=set(pos[pos==0].index)
print("train subjects with 0 positives:",len(zs),"their decisions:",int(tr.subject.isin(zs).sum()))
print("zero-positive train subjects that HAVE annotated seizures:",len(zs & set(g.index)))
print("train subjects with no annotated seizures at all:",len(set(tr.subject.unique())-set(g.index)))
