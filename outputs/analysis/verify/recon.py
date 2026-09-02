import pandas as pd, numpy as np, itertools
SZ='data/interim/manifests/seizure_manifest.csv'
sz=pd.read_csv(SZ)
sz['recording_id']=sz.seizure_id.str.replace(r'_seizure-\d+$','',regex=True)
POST=60.0*60; HIST=60.0*60
def ov(a,b,c,d): return (a<d) and (b>c)
recs={}
for p in sz.events_path.unique():
    ev=pd.read_csv(p,sep='\t')
    r=ev.loc[:,['onset','duration','eventType']].copy()
    r['onset']=pd.to_numeric(r['onset'],errors='coerce')
    r['duration']=pd.to_numeric(r['duration'],errors='coerce')
    r['eventType']=r['eventType'].fillna('').astype(str)
    recs[p]=r.loc[r.onset.notna()&r.duration.notna()&r.duration.ge(0)].reset_index(drop=True)
out=[]
for rid,g in sz.groupby('recording_id'):
    p=g.events_path.iloc[0]; re_=recs[p]
    nonbg=re_[~(re_.eventType.str.strip().str.lower().eq('bckg')|re_.eventType.str.strip().str.lower().str.startswith('sz_'))]
    for _,s in g.iterrows():
        onset=float(s.onset_seconds); start=onset-HIST
        short = start<0
        clust=False
        for _,o in g.iterrows():
            if ov(start,onset,float(o.onset_seconds),float(o.onset_seconds)+float(o.duration_seconds)+POST):
                clust=True; break
        nb=False
        for _,e in nonbg.iterrows():
            if ov(start,onset,float(e.onset),float(e.onset)+float(e.duration)):
                nb=True; break
        out.append(dict(seizure_id=s.seizure_id,recording_id=rid,subject=s.subject,short=short,clust=clust,nonbg=nb,
                        pred_elig=(not short) and (not clust) and (not nb), actual=bool(s.eligible_for_prediction),
                        vigilance=s.vigilance,event_type=s.event_type,onset=onset,dur=s.duration_seconds,
                        lat=s.lateralization,loc=s.localization))
d=pd.DataFrame(out)
d.to_csv('outputs/analysis/verify/recon.csv',index=False)
print('n',len(d))
print('pred elig',d.pred_elig.sum(),'actual elig',d.actual.sum())
print(pd.crosstab(d.pred_elig,d.actual))
print('mismatches:'); print(d[d.pred_elig!=d.actual][['seizure_id','short','clust','nonbg','pred_elig','actual']].to_string())
inel=d[~d.actual]
print('\nnon-exclusive among 566: clust',inel.clust.sum(),'short',inel.short.sum(),'nonbg',inel.nonbg.sum())
# code-order cascade: clustering checked LAST in code; order in code: start<0, n_times, nonfinite, bad, nonbg, ictal
c=0; rem=inel.copy()
for name in ['short','nonbg','clust']:
    k=rem[name].sum(); print('cascade(code order)',name,k); rem=rem[~rem[name]]
print('residual',len(rem))
print('n patients with >=1 eligible', d[d.actual].subject.nunique())
