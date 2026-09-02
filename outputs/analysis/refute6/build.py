import numpy as np, pandas as pd, json, pickle, sys
from pathlib import Path
ROOT = Path("/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred")
FD = ROOT/"data/interim/chunk_features"
rng = np.random.default_rng(0)

sz = pd.read_csv(ROOT/"data/interim/manifests/seizure_manifest.csv")
sz["recording_id"] = ("sub-"+sz.subject.astype(str).str.zfill(3)+"_ses-"+sz.session.astype(str).str.zfill(2)
                      +"_task-"+sz.task.astype(str)+"_run-"+sz.run.astype(str).str.zfill(2))
print("seizures", len(sz), "eligible", int(sz.eligible_for_prediction.sum()))
dec = pd.read_csv(ROOT/"data/interim/manifests/decision_manifest.csv",
                  usecols=["recording_id","subject","decision_end_sample","decision_time_seconds","label","target_seizure_id","split"])
print("decisions", len(dec), "pos", int(dec.label.sum()))

CH=1280; NC=540
def load(rid):
    f = FD/f"{rid}_features.npy"; a = FD/f"{rid}_availability.npy"
    if not f.exists(): return None,None
    return np.load(f, mmap_mode="r"), np.load(a)

def grab(bank, end_sample):
    stop = int(end_sample)//CH; start = stop-NC
    if start < 0 or stop > bank.shape[0]: return None
    return np.asarray(bank[start:stop], dtype=np.float32)

# ---- ictal-containing and pre-ictal windows from eligible seizures
elig = sz[sz.eligible_for_prediction].copy()
ict, pre, meta = [], [], []
for rid, g in elig.groupby("recording_id"):
    bank, av = load(rid)
    if bank is None: continue
    for _, r in g.iterrows():
        on = float(r.onset_seconds)
        wi = grab(bank, (on+300.0)*256.0)
        wp = grab(bank, (on-60.0)*256.0)
        if wi is None or wp is None: continue
        ict.append(wi); pre.append(wp)
        meta.append(dict(seizure_id=r.seizure_id, subject=r.subject, recording_id=rid,
                         onset=on, vigilance=r.vigilance, dur=r.duration_seconds, av=av.copy()))
print("ictal windows", len(ict), "preictal", len(pre), "subjects", len({m['subject'] for m in meta}))

# ---- far-interictal windows: >2h from ANY annotated seizure in that recording
onsets = {rid: g.onset_seconds.values for rid, g in sz.groupby("recording_id")}
neg = dec[dec.label==0]
rows=[]
for rid, g in neg.groupby("recording_id"):
    on = onsets.get(rid, None)
    t = g.decision_time_seconds.values
    if on is not None and len(on):
        d = np.abs(t[:,None]-on[None,:]).min(axis=1)
        keep = d > 7200.0
    else:
        keep = np.ones(len(t), bool)
    gg = g[keep]
    if len(gg)==0: continue
    take = gg.sample(n=min(2,len(gg)), random_state=1)
    rows.append(take)
neg_s = pd.concat(rows)
neg_s = neg_s.sample(n=min(1600,len(neg_s)), random_state=2)
inter, imeta = [], []
for rid, g in neg_s.groupby("recording_id"):
    bank, av = load(rid)
    if bank is None: continue
    for _, r in g.iterrows():
        w = grab(bank, r.decision_end_sample)
        if w is None: continue
        inter.append(w); imeta.append(dict(subject=r.subject, recording_id=rid,
                                           t=r.decision_time_seconds, av=av.copy()))
print("interictal windows", len(inter), "subjects", len({m['subject'] for m in imeta}))
with open("/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred/outputs/analysis/refute6/cache.pkl","wb") as fh:
    pickle.dump(dict(ict=np.stack(ict), pre=np.stack(pre), inter=np.stack(inter),
                     meta=meta, imeta=imeta), fh, protocol=4)
print("saved")
