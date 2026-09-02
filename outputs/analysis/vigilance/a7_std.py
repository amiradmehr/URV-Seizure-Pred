import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
ROOT=Path('/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'); OUT=ROOT/'outputs/analysis/vigilance'
S=pd.read_csv(OUT/'per_seizure_full.csv',dtype={'subject':str})
S=S[S.vigilance.isin(['asleep','awake'])].copy()
dec=pd.read_csv(ROOT/'data/interim/manifests/decision_manifest.csv',usecols=['recording_id','bte_side']).groupby('recording_id').bte_side.first()
S['bte_side']=S.recording_id.map(dec)
# direct standardisation to the pooled montage distribution
raw=[];std=[]
for tag,g in S.groupby('tag'):
    w=g.bte_side.value_counts(normalize=True)
    m=g.groupby(['vigilance','bte_side']).caught.mean().unstack()
    for s in w.index:
        if s not in m.columns: m[s]=np.nan
    m=m[w.index]
    a=np.nansum(m.loc['asleep'].values*w.values)/np.nansum(w.values*np.isfinite(m.loc['asleep'].values))
    b=np.nansum(m.loc['awake'].values*w.values)/np.nansum(w.values*np.isfinite(m.loc['awake'].values))
    raw.append(g[g.vigilance=='asleep'].caught.mean()-g[g.vigilance=='awake'].caught.mean()); std.append(a-b)
raw=np.array(raw);std=np.array(std)
print('crude asleep-awake gap            = %+.4f (t=%.2f, p=%.4g)'%(raw.mean(),*stats.ttest_1samp(raw,0)[::1]))
print('montage-standardised gap          = %+.4f (t=%.2f, p=%.4g)'%(std.mean(),*stats.ttest_1samp(std,0)[::1]))
print('share of gap explained by montage = %.0f%%'%(100*(1-std.mean()/raw.mean())))
# expected gap from montage composition alone
mm=S.groupby(['tag','bte_side']).caught.mean().unstack().mean()
wa=S[S.tag==S.tag.iloc[0]].groupby('vigilance').bte_side.value_counts(normalize=True).unstack()
exp=(wa*mm).sum(1)
print('\nmontage mix:'); print(wa.round(3))
print('sens by montage:', dict(mm.round(4)))
print('gap predicted by montage mix alone = %+.4f  (observed %+.4f -> %.0f%%)'%(exp['asleep']-exp['awake'],raw.mean(),100*(exp['asleep']-exp['awake'])/raw.mean()))
print('\n=== n_pos mechanical contribution ===')
n=S.groupby(['tag','vigilance']).n_pos.mean().unstack()
p=S.groupby(['tag','vigilance']).caught.mean().unstack()
# sens ~ 1-(1-q)^n ; extra chances effect
q=1-(1-p['awake'])**(1/n['awake'])
pred=1-(1-q)**n['asleep']
print('asleep n_pos=%.2f awake n_pos=%.2f  -> gap from extra decisions alone = %+.5f (%.0f%% of observed)'%(
    n['asleep'].mean(),n['awake'].mean(),(pred-p['awake']).mean(),100*(pred-p['awake']).mean()/raw.mean()))
