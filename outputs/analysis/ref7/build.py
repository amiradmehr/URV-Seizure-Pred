import numpy as np, pandas as pd, pickle, sys
from pathlib import Path
ROOT = Path("/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred")
FD = ROOT/"data/interim/chunk_features"
sz = pd.read_csv(ROOT/"data/interim/manifests/seizure_manifest.csv")
sz["recording_id"] = ("sub-"+sz.subject.astype(str).str.zfill(3)+"_ses-"+sz.session.astype(str).str.zfill(2)
                      +"_task-"+sz.task.astype(str)+"_run-"+sz.run.astype(str).str.zfill(2))
dec = pd.read_csv(ROOT/"data/interim/manifests/decision_manifest.csv",
    usecols=["recording_id","subject","decision_end_sample","decision_time_seconds","label","target_seizure_id","split"])
print("seizures",len(sz),"eligible",int(sz.eligible_for_prediction.sum()),
      "decisions",len(dec),"pos",int(dec.label.sum()),flush=True)
CH=1280; NC=540
def load(rid):
    f=FD/f"{rid}_features.npy"; a=FD/f"{rid}_availability.npy"
    if not f.exists(): return None,None
    return np.load(f,mmap_mode="r"), np.load(a)
def grab(bank,end_sample):
    stop=int(end_sample)//CH; start=stop-NC
    if start<0 or stop>bank.shape[0]: return None
    return np.asarray(bank[start:stop],dtype=np.float32)
elig=sz[sz.eligible_for_prediction].copy()
pre=[];i300=[];ioff=[];meta=[]
for rid,g in elig.groupby("recording_id"):
    bank,av=load(rid)
    if bank is None: continue
    for _,r in g.iterrows():
        on=float(r.onset_seconds); dur=float(r.duration_seconds)
        wp=grab(bank,(on-60.0)*256.0); w3=grab(bank,(on+300.0)*256.0)
        wo=grab(bank,(on+dur)*256.0)
        if wp is None or w3 is None or wo is None: continue
        pre.append(wp); i300.append(w3); ioff.append(wo)
        meta.append(dict(seizure_id=r.seizure_id,subject=r.subject,recording_id=rid,onset=on,
                         vigilance=r.vigilance,dur=dur,av=av.copy()))
print("n seizures with all 3 windows",len(meta),"subjects",len({m['subject'] for m in meta}),flush=True)
onsets={rid:g.onset_seconds.values for rid,g in sz.groupby("recording_id")}
neg=dec[dec.label==0]
rows=[]
for rid,g in neg.groupby("recording_id"):
    on=onsets.get(rid,None); t=g.decision_time_seconds.values
    keep = (np.abs(t[:,None]-on[None,:]).min(axis=1)>7200.0) if (on is not None and len(on)) else np.ones(len(t),bool)
    gg=g[keep]
    if len(gg)==0: continue
    rows.append(gg.sample(n=min(2,len(gg)),random_state=1))
neg_s=pd.concat(rows).sample(n=1600,random_state=2)
inter=[];imeta=[]
for rid,g in neg_s.groupby("recording_id"):
    bank,av=load(rid)
    if bank is None: continue
    for _,r in g.iterrows():
        w=grab(bank,r.decision_end_sample)
        if w is None: continue
        inter.append(w); imeta.append(dict(subject=r.subject,recording_id=rid,t=r.decision_time_seconds,av=av.copy()))
print("interictal",len(inter),"subjects",len({m['subject'] for m in imeta}),flush=True)
# also: same-recording far-interictal window matched to each seizure (for paired test)
paired={}
for rid,g in neg.groupby("recording_id"):
    on=onsets.get(rid,None); t=g.decision_time_seconds.values
    if on is None or not len(on): continue
    keep=np.abs(t[:,None]-on[None,:]).min(axis=1)>7200.0
    gg=g[keep]
    if len(gg)==0: continue
    paired[rid]=gg.sample(n=min(4,len(gg)),random_state=3).decision_end_sample.values
pm=[];pw=[]
for i,m in enumerate(meta):
    rid=m["recording_id"]
    if rid not in paired: continue
    bank,av=load(rid)
    ws=[grab(bank,e) for e in paired[rid]]
    ws=[w for w in ws if w is not None]
    if not ws: continue
    pm.append(i); pw.append(np.stack(ws))
print("paired seizures",len(pm),"subjects",len({meta[i]['subject'] for i in pm}),flush=True)
pickle.dump(dict(pre=np.stack(pre),i300=np.stack(i300),ioff=np.stack(ioff),inter=np.stack(inter),
                 meta=meta,imeta=imeta,pair_idx=pm,pair_win=pw),
            open(ROOT/"outputs/analysis/ref7/cache.pkl","wb"),protocol=4)
print("saved",flush=True)
