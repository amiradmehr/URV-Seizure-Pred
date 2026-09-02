import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
ROOT=Path('/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'); OUT=ROOT/'outputs/analysis/vigilance'
S=pd.read_csv(OUT/'per_seizure_full.csv',dtype={'subject':str})
print('per_seizure_full cols:',list(S.columns))
print('rows',len(S),'tags',S.tag.nunique(),'seizures',S.seizure_id.nunique())
A=S[S.vigilance.isin(['asleep','awake'])].copy()
print('asleep/awake seizures:',A.groupby('vigilance').seizure_id.nunique().to_dict(),'total',A.seizure_id.nunique())

# ---- A. does the recomputed 'caught' reproduce the sweep results.csv? ----
res=pd.read_csv(ROOT/'outputs/sweep/results.csv').set_index('tag')
mine=A.groupby(['tag','vigilance']).caught.mean().unstack()
cmp=mine.join(res[['sens_asleep','sens_awake','n_asleep','n_awake','sens_at_1ph']])
cmp['d_asleep']=cmp['asleep']-cmp['sens_asleep']; cmp['d_awake']=cmp['awake']-cmp['sens_awake']
print('\n== reproduction vs outputs/sweep/results.csv ==')
print('max |asleep diff| %.5f   max |awake diff| %.5f'%(cmp.d_asleep.abs().max(),cmp.d_awake.abs().max()))
print('n_asleep in results.csv:',sorted(res.n_asleep.dropna().unique()),' n_awake:',sorted(res.n_awake.dropna().unique()))
print('my counts per tag:',A[A.tag==A.tag.iloc[0]].vigilance.value_counts().to_dict())
print(cmp[['asleep','sens_asleep','awake','sens_awake','d_asleep','d_awake']].head(6).round(4).to_string())

# ---- B. the published paired-across-configs test ----
p=A.pivot_table(index='tag',columns='vigilance',values='caught',aggfunc='mean')
t=stats.ttest_rel(p['asleep'],p['awake'])
print('\n== published test: paired t across 42 configs ==')
print('asleep=%.4f awake=%.4f diff=%+.4f t=%.3f p=%.6f  (%d/%d configs +)'%(
    p['asleep'].mean(),p['awake'].mean(),(p['asleep']-p['awake']).mean(),t.statistic,t.pvalue,(p['asleep']>p['awake']).sum(),len(p)))

# ---- C. seizure as unit (average caught over 42 configs) ----
per_sz=A.groupby(['seizure_id','subject','vigilance','recording_id']).caught.mean().reset_index()
a=per_sz[per_sz.vigilance=='asleep'].caught; b=per_sz[per_sz.vigilance=='awake'].caught
print('\n== seizure as unit (mean caught over 42 configs) ==')
print('asleep %.4f (n=%d) awake %.4f (n=%d) diff=%+.4f'%(a.mean(),len(a),b.mean(),len(b),a.mean()-b.mean()))
print('Welch t=%.3f p=%.4f ; MWU p=%.4f'%(*stats.ttest_ind(a,b,equal_var=False),stats.mannwhitneyu(a,b).pvalue))

# ---- D. PATIENT as unit: cluster bootstrap over patients ----
rng=np.random.default_rng(0)
subs=per_sz.subject.unique()
idx={s:per_sz.index[per_sz.subject==s].to_numpy() for s in subs}
def gap(df):
    aa=df[df.vigilance=='asleep'].caught; bb=df[df.vigilance=='awake'].caught
    if len(aa)==0 or len(bb)==0: return np.nan
    return aa.mean()-bb.mean()
obs=gap(per_sz); boots=[]
for i in range(4000):
    pick=rng.choice(subs,size=len(subs),replace=True)
    d=per_sz.loc[np.concatenate([idx[s] for s in pick])]
    boots.append(gap(d))
boots=np.array(boots); boots=boots[np.isfinite(boots)]
lo,hi=np.percentile(boots,[2.5,97.5])
pboot=2*min((boots<=0).mean(),(boots>=0).mean())
print('\n== PATIENT cluster bootstrap (4000 reps, resample patients) ==')
print('obs gap %+.4f  95%% CI [%+.4f, %+.4f]  boot p=%.4f  (boot SD %.4f)'%(obs,lo,hi,pboot,boots.std()))
