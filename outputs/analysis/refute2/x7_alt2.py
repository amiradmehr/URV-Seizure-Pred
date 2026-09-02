import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
ROOT=Path('/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'); OUT=ROOT/'outputs/analysis/vigilance'
S=pd.read_csv(OUT/'per_seizure_full.csv',dtype={'subject':str})
A=S[S.vigilance.isin(['asleep','awake'])].copy()
dec=pd.read_csv(ROOT/'data/interim/manifests/decision_manifest.csv',usecols=['recording_id','bte_side'])
A['bte_side']=A.recording_id.map(dec.groupby('recording_id').bte_side.first())
sz=pd.read_csv(ROOT/'data/interim/manifests/seizure_manifest.csv',dtype={'subject':str}).set_index('seizure_id')
per=A.groupby(['seizure_id','subject','vigilance','bte_side','fold']).caught.mean().reset_index()
for c in ['lateralization','localization','event_type']:
    per[c]=per.seizure_id.map(sz[c].astype(str))
# patient-level versions of each
for c in ['lateralization','localization','event_type']:
    pm=per.groupby('subject')[c].agg(lambda s:s.mode().iloc[0]); per['pat_'+c]=per.subject.map(pm)
crude=per[per.vigilance=='asleep'].caught.mean()-per[per.vigilance=='awake'].caught.mean()
def std_gap(d,col):
    w=d[col].value_counts(normalize=True); m=d.groupby(['vigilance',col]).caught.mean().unstack()
    for s in w.index:
        if s not in m.columns: m[s]=np.nan
    m=m[w.index]
    def wm(r):
        v=r.values.astype(float); ok=np.isfinite(v)
        return np.nansum(v[ok]*w.values[ok])/w.values[ok].sum()
    return wm(m.loc['asleep'])-wm(m.loc['awake'])
print('crude gap %+.4f'%crude)
print('%-24s %-8s %-10s %s'%('stratifier','levels','std gap','% "explained"'))
for c in ['bte_side','fold','pat_lateralization','pat_localization','pat_event_type','lateralization','localization']:
    g=std_gap(per,c); print('%-24s %-8d %+8.4f   %.0f%%'%(c,per[c].nunique(),g,100*(1-g/crude)))
print('\nsens by fold:',per.groupby('fold').caught.mean().round(4).to_dict())
print('asleep share by fold:',per.groupby('fold').vigilance.apply(lambda s:(s=="asleep").mean()).round(3).to_dict())
print('\nbte_side counts by recording:',dec.groupby('recording_id').bte_side.first().value_counts().to_dict())
