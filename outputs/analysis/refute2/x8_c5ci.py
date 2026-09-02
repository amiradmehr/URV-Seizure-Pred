import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
ROOT=Path('/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'); OUT=ROOT/'outputs/analysis/vigilance'
S=pd.read_csv(OUT/'per_seizure_full.csv',dtype={'subject':str})
A=S[S.vigilance.isin(['asleep','awake'])].copy()
dec=pd.read_csv(ROOT/'data/interim/manifests/decision_manifest.csv',usecols=['recording_id','bte_side'])
A['bte_side']=A.recording_id.map(dec.groupby('recording_id').bte_side.first())
per=A.groupby(['seizure_id','subject','vigilance','bte_side']).caught.mean().reset_index()
def gaps(d):
    crude=d[d.vigilance=='asleep'].caught.mean()-d[d.vigilance=='awake'].caught.mean()
    w=d.bte_side.value_counts(normalize=True); m=d.groupby(['vigilance','bte_side']).caught.mean().unstack()
    for s in w.index:
        if s not in m.columns: m[s]=np.nan
    m=m[w.index]
    def wm(r):
        v=r.values.astype(float); ok=np.isfinite(v)
        return np.nansum(v[ok]*w.values[ok])/w.values[ok].sum()
    return crude, wm(m.loc['asleep'])-wm(m.loc['awake'])
c0,s0=gaps(per); rng=np.random.default_rng(1)
subs=per.subject.unique(); ix={s:per.index[per.subject==s].to_numpy() for s in subs}
dif=[]
for _ in range(4000):
    pk=rng.choice(subs,size=len(subs),replace=True)
    g=per.loc[np.concatenate([ix[s] for s in pk])]
    if g.vigilance.nunique()<2: continue
    try:
        c,s=gaps(g)
        if np.isfinite(c) and np.isfinite(s): dif.append(c-s)
    except Exception: pass
dif=np.array(dif)
print('crude %+.4f  standardised %+.4f  amount removed by montage = %+.4f'%(c0,s0,c0-s0))
print('patient-boot 95%% CI on the AMOUNT REMOVED: [%+.4f, %+.4f]  p=%.4f'%(*np.percentile(dif,[2.5,97.5]),2*min((dif<=0).mean(),(dif>=0).mean())))
print('  -> i.e. montage adjustment removes between %.0f%% and %.0f%% of a +0.0188 gap'%(100*np.percentile(dif,2.5)/c0,100*np.percentile(dif,97.5)/c0))
