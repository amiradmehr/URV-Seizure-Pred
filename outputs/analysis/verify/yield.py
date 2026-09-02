import pandas as pd, numpy as np, os
sz=pd.read_csv('data/interim/manifests/seizure_manifest.csv')
sz['recording_id']=sz.seizure_id.str.replace(r'_seizure-\d+$','',regex=True)
sh=pd.read_csv('data/interim/manifests/processed_shard_manifest.csv')
kept=set(sh.recording_id)
recs={}
for p in sz.events_path.unique():
    ev=pd.read_csv(p,sep='\t')
    r=ev.loc[:,['onset','duration','eventType']].copy()
    r['onset']=pd.to_numeric(r['onset'],errors='coerce'); r['duration']=pd.to_numeric(r['duration'],errors='coerce')
    r['eventType']=r['eventType'].fillna('').astype(str)
    r=r.loc[r.onset.notna()&r.duration.notna()&r.duration.ge(0)]
    et=r.eventType.str.strip().str.lower()
    recs[p]=r[~(et.eq('bckg')|et.str.startswith('sz_'))].reset_index(drop=True)

def evaluate(HISTm,POSTm):
    H=HISTm*60.0; P=POSTm*60.0; res=[]
    for rid,g in sz.groupby('recording_id'):
        nb=recs[g.events_path.iloc[0]]
        on=g.onset_seconds.values.astype(float); du=g.duration_seconds.values.astype(float)
        nbo=nb.onset.values.astype(float); nbd=nb.duration.values.astype(float)
        for i,(_,s) in enumerate(g.iterrows()):
            o=on[i]; st=o-H
            if st<0: e=False
            else:
                cl=((st<on+du+P)&(o>on)).any()
                n2=((st<nbo+nbd)&(o>nbo)).any() if len(nbo) else False
                e=(not cl) and (not n2)
            res.append((s.seizure_id,s.subject,rid,e))
    d=pd.DataFrame(res,columns=['sid','subject','rid','elig'])
    d['hasfeat']=d.rid.isin(kept)
    el=d[d.elig]
    return len(el), el.subject.nunique(), int((el.hasfeat).sum()), el[el.hasfeat].subject.nunique()

for h,p in [(60,60),(60,30),(60,10),(55,60),(30,10),(15,60),(15,10),(55,10)]:
    n,np_,nf,npf = evaluate(h,p)
    print(f'hist={h:3d} post={p:3d}  elig={n:4d} pat={np_:3d} | with_features={nf:4d} pat={npf:3d}  (lost_no_features={n-nf})')
