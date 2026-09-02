import numpy as np, pandas as pd
from scipy import stats
from pathlib import Path
ROOT=Path('/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'); V=ROOT/'outputs/analysis/vigilance'
S=pd.read_csv(V/'per_seizure_full.csv',dtype={'subject':str})
S=S[S.vigilance.isin(['asleep','awake'])]
R=pd.read_csv(V/'per_recording_by_config.csv')
nneg=R[R.tag==R.tag.iloc[0]].set_index('recording_id').n_neg
sz=S.groupby(['seizure_id','subject','vigilance','recording_id']).agg(
    caught=('caught','mean'),caught_rec=('caught_rec','mean'),caught_loc=('caught_loc','mean'),
    n_loc=('n_loc','mean')).reset_index()
sz['n_neg']=sz.recording_id.map(nneg)

print('=== Is the PER-RECORDING "1 FA/h" threshold actually 1 FA/h? ===')
def eff_fa(n):
    if not np.isfinite(n) or n<1: return np.nan
    n=int(n); need=int(np.ceil(59*n/60.0)); allowed=max(n-need,0)
    return 60.0*allowed/n
sz['eff_fa_rec']=sz.n_neg.map(eff_fa)
sz['eff_fa_loc']=sz.n_loc.map(eff_fa)
print(sz.groupby('vigilance')[['n_neg','eff_fa_rec','n_loc','eff_fa_loc']].agg(['mean','median']).round(3))
for c in ['eff_fa_rec','eff_fa_loc']:
    a=sz[sz.vigilance=='asleep'][c].dropna(); b=sz[sz.vigilance=='awake'][c].dropna()
    print('  %s: asleep=%.3f awake=%.3f  diff=%+.3f  MWU p=%.3g'%(c,a.mean(),b.mean(),a.mean()-b.mean(),stats.mannwhitneyu(a,b).pvalue))
print('  -> the AWAKE group is scored at a LOOSER effective alarm budget' if sz[sz.vigilance=='awake'].eff_fa_rec.mean()>sz[sz.vigilance=='asleep'].eff_fa_rec.mean() else '  -> asleep looser')

print('\n=== realized alarm-rate mismatch: correlation of eff_fa with caught_rec ===')
r=stats.spearmanr(sz.eff_fa_rec.fillna(0),sz.caught_rec.fillna(0)); print('spearman(eff_fa_rec, caught_rec)=%+.3f p=%.3g'%(r.statistic,r.pvalue))
# stratify the "reversal" by effective alarm budget
sz['fq']=pd.qcut(sz.eff_fa_loc.rank(method='first'),3,labels=['low','mid','high'])
g=sz.dropna(subset=['caught_loc']).groupby(['fq','vigilance'],observed=True).caught_loc.mean().unstack()
print('caught_loc by effective-FA tertile:'); print(g.round(4)); print('gap per tertile:',(g['asleep']-g['awake']).round(4).to_dict())

print('\n=== overall catch rates under each calibration (are they even comparable?) ===')
for c in ['caught','caught_rec','caught_loc']:
    print('  %-11s overall=%.4f'%(c,sz[c].mean()))

print('\n=== C3: per-recording threshold on <=1h recordings is degenerate ===')
r1=R[R.tag==R.tag.iloc[0]]
r1=r1.assign(eff=r1.n_neg.map(eff_fa))
print('recordings: n=%d ; effective FA/h under per-recording rule: mean=%.3f median=%.3f ; %% with eff_fa==0: %.1f%%'%(
    len(r1),r1.eff.mean(),r1.eff.median(),100*(r1.eff==0).mean()))
print('among the 275 seizure recordings: %% with eff_fa_rec==0: %.1f%%  (asleep %.1f%% awake %.1f%%)'%(
    100*(sz.eff_fa_rec==0).mean(),100*(sz[sz.vigilance=="asleep"].eff_fa_rec==0).mean(),100*(sz[sz.vigilance=="awake"].eff_fa_rec==0).mean()))
