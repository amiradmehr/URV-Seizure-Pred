import pandas as pd, numpy as np, json, os, sys
from pathlib import Path
from scipy import stats

sweep = pd.read_csv('outputs/sweep/results.csv')
tags = [t for t in sweep.tag if '__' in t]   # the 42 real sweep configs
print(len(tags), 'configs')

sz = pd.read_csv('data/interim/manifests/seizure_manifest.csv', dtype={'subject':str})
vig = dict(zip(sz.seizure_id.astype(str), sz.vigilance.astype(str)))
subj_of = dict(zip(sz.seizure_id.astype(str), sz.subject.astype(str)))

caught_mat = {}
for t in tags:
    p = Path('outputs/cv')/t/'out_of_fold_predictions.csv'
    if not p.exists():
        print('MISSING', t); continue
    d = pd.read_csv(p, usecols=['label','probability','target_seizure_id'])
    neg = np.sort(d.loc[d.label==0,'probability'].to_numpy())[::-1]
    hours = (d.label==0).sum()*60/3600.0
    allowed = int(1.0*hours)
    thr = float(neg[min(allowed, len(neg)-1)])
    pos = d[d.label==1]
    c = pos.assign(a=pos.probability>=thr).groupby('target_seizure_id')['a'].any()
    caught_mat[t] = c
    print(t, 'thr=%.4f'%thr, 'seiz=%d'%len(c), 'caught=%d'%c.sum(), flush=True)

M = pd.DataFrame(caught_mat)   # seizures x configs
M.to_csv('outputs/analysis/caught_matrix.csv')
print('matrix', M.shape, 'any NaN', M.isna().sum().sum())
