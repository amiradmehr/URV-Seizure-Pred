import json, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path('/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred')
CV = ROOT/'outputs/cv'
OUT = ROOT/'outputs/analysis/vigilance'
STRIDE = 60.0

sz = pd.read_csv(ROOT/'data/interim/manifests/seizure_manifest.csv', dtype={'subject':str})
sz['seizure_id']=sz['seizure_id'].astype(str)
vig = dict(zip(sz.seizure_id, sz.vigilance.astype(str)))
onset = dict(zip(sz.seizure_id, sz.onset_seconds))
szsub = dict(zip(sz.seizure_id, sz.subject))

res = pd.read_csv(ROOT/'outputs/sweep/results.csv')
tags = res.dropna(subset=['sens_asleep']).tag.tolist()
print('configs with vigilance:', len(tags))

def threshold_for_budget(prob_neg, budget):
    negs = np.sort(prob_neg)[::-1]
    hours = len(negs)*STRIDE/3600.0
    allowed = int(budget*hours)
    return float(negs[min(allowed, len(negs)-1)])

rows_sz = []   # per (tag, seizure)
rows_rec = []  # per (tag, recording)
glob = []
for tag in tags:
    f = CV/tag/'out_of_fold_predictions.csv'
    d = pd.read_csv(f, usecols=['recording_id','subject','decision_time_seconds','label','target_seizure_id','probability','fold'])
    d['subject']=d['subject'].astype(str).str.zfill(3)
    neg = d[d.label==0]
    pos = d[d.label==1].copy()
    thr = threshold_for_budget(neg.probability.to_numpy(), 1.0)
    # per-recording negative stats
    g = neg.groupby('recording_id')['probability']
    rec_stats = pd.DataFrame({'n_neg':g.size(),'neg_mean':g.mean(),'neg_med':g.median(),
                              'neg_q99':g.quantile(0.99),'neg_max':g.max(),
                              'dur_s':neg.groupby('recording_id')['decision_time_seconds'].max()})
    rec_stats['tag']=tag; rec_stats=rec_stats.reset_index()
    rows_rec.append(rec_stats)
    # negatives per recording sorted for percentile lookup
    negsort = {r: np.sort(v.to_numpy()) for r,v in neg.groupby('recording_id')['probability']}
    negtime = {r: (v['decision_time_seconds'].to_numpy(), v['probability'].to_numpy())
               for r,v in neg.groupby('recording_id')[['decision_time_seconds','probability']]}
    globalsort = np.sort(neg.probability.to_numpy())
    for sid, sub in pos.groupby('target_seizure_id'):
        rid = sub.recording_id.iloc[0]
        peak = float(sub.probability.max()); mean_p = float(sub.probability.mean())
        ns = negsort.get(rid, np.array([]))
        pct_rec = float(np.searchsorted(ns, peak)/len(ns)) if len(ns)>=30 else np.nan
        # local control: negatives in same recording within +-3h of onset
        t0 = onset.get(sid, sub.decision_time_seconds.max())
        pct_loc = np.nan; n_loc = 0
        if rid in negtime:
            tt, pp = negtime[rid]
            m = np.abs(tt-t0) <= 3*3600
            n_loc = int(m.sum())
            if n_loc >= 30:
                pct_loc = float((pp[m] < peak).mean())
        pct_glob = float(np.searchsorted(globalsort, peak)/len(globalsort))
        rows_sz.append(dict(tag=tag, seizure_id=sid, recording_id=rid, subject=sub.subject.iloc[0],
                            fold=int(sub.fold.iloc[0]), vigilance=vig.get(sid,'un'),
                            n_pos=int(len(sub)), peak=peak, mean_p=mean_p,
                            caught=int(peak>=thr), pct_rec=pct_rec, pct_loc=pct_loc,
                            n_loc=n_loc, pct_glob=pct_glob,
                            rec_neg_mean=float(rec_stats.set_index('recording_id').neg_mean.get(rid,np.nan))))
    glob.append(dict(tag=tag, thr=thr, n_neg=len(neg), n_pos=len(pos)))
    print(tag, 'thr=%.4f'%thr, flush=True)

S = pd.DataFrame(rows_sz); R = pd.concat(rows_rec, ignore_index=True)
S.to_csv(OUT/'per_seizure_by_config.csv', index=False)
R.to_csv(OUT/'per_recording_by_config.csv', index=False)
pd.DataFrame(glob).to_csv(OUT/'thresholds.csv', index=False)
print(S.shape, R.shape)
