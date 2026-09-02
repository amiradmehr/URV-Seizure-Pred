import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
import numpy.linalg as la
ROOT=Path('/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'); OUT=ROOT/'outputs/analysis/vigilance'
S=pd.read_csv(OUT/'per_seizure_full.csv',dtype={'subject':str})
A=S[S.vigilance.isin(['asleep','awake'])].copy()
dec=pd.read_csv(ROOT/'data/interim/manifests/decision_manifest.csv',usecols=['recording_id','bte_side','decision_time_seconds'])
side=dec.groupby('recording_id').bte_side.first(); nrec=dec.groupby('recording_id').size()
A['bte_side']=A.recording_id.map(side); A['n_dec']=A.recording_id.map(nrec)
rng=np.random.default_rng(0)

print('=== ALT-EXPLANATION for CLAIM 4: is pct_rec biased by recording LENGTH? ===')
per=A.groupby(['seizure_id','subject','vigilance','recording_id','bte_side','n_dec'])[['pct_rec','pct_loc','caught']].mean().reset_index().dropna(subset=['pct_rec'])
print('  corr(log n_dec, pct_rec) = %.3f (Spearman p=%.3g)'%stats.spearmanr(np.log(per.n_dec),per.pct_rec)[:2])
print('  n_dec asleep median %.0f vs awake median %.0f'%(per[per.vigilance=='asleep'].n_dec.median(),per[per.vigilance=='awake'].n_dec.median()))
x=(per.vigilance=='asleep').astype(float).values; y=per.pct_rec.values; z=np.log(per.n_dec.values); z=(z-z.mean())/z.std()
b1=la.lstsq(np.c_[np.ones(len(x)),x],y,rcond=None)[0][1]
b2=la.lstsq(np.c_[np.ones(len(x)),x,z],y,rcond=None)[0][1]
print('  beta(asleep) on pct_rec: unadjusted %+.4f -> adjusted for log(recording length) %+.4f  (%.0f%% removed)'%(b1,b2,100*(1-b2/b1)))
# stratify by length tertile
per['lt']=pd.qcut(per.n_dec,3,labels=['short','mid','long'])
g=per.groupby(['lt','vigilance'],observed=True).pct_rec.agg(['mean','size']).unstack()
print(g.round(4).to_string())

print('\n=== ALT-EXPLANATION for CLAIM 5: does ANY patient-level 3-level variable "explain" the gap? ===')
per2=A.groupby(['seizure_id','subject','vigilance','bte_side']).caught.mean().reset_index()
def std_gap(d,col):
    w=d[col].value_counts(normalize=True)
    m=d.groupby(['vigilance',col]).caught.mean().unstack()
    for s in w.index:
        if s not in m.columns: m[s]=np.nan
    m=m[w.index]
    def wm(row):
        v=row.values.astype(float); ok=np.isfinite(v); 
        return np.nansum(v[ok]*w.values[ok])/w.values[ok].sum() if ok.any() else np.nan
    return wm(m.loc['asleep'])-wm(m.loc['awake'])
crude=per2[per2.vigilance=='asleep'].caught.mean()-per2[per2.vigilance=='awake'].caught.mean()
print('  crude gap %+.4f ; montage-standardised %+.4f -> %.0f%% "explained"'%(crude,std_gap(per2,'bte_side'),100*(1-std_gap(per2,'bte_side')/crude)))
subs=per2.subject.unique()
expl=[]
for i in range(500):
    lab=pd.Series(rng.integers(0,3,size=len(subs)),index=subs)   # RANDOM patient-level 3-level variable
    per2['rnd']=per2.subject.map(lab)
    s=std_gap(per2,'rnd')
    if np.isfinite(s): expl.append(100*(1-s/crude))
expl=np.array(expl)
print('  RANDOM patient-level 3-level variable: mean %%explained = %.0f%%, SD %.0f, 2.5-97.5 pct [%.0f%%, %.0f%%]'%(
    expl.mean(),expl.std(),*np.percentile(expl,[2.5,97.5])))
print('  P(random patient-level variable explains >=100%%) = %.3f'%(expl>=100).mean())
print('  P(random patient-level variable explains >=  90%%) = %.3f'%(expl>=90).mean())

print('\n=== is the montage-vigilance link a PATIENT-composition fact? ===')
key=per2.drop_duplicates('seizure_id')
pt=key.groupby('subject').agg(side=('bte_side','first'),f_asleep=('vigilance',lambda s:(s=='asleep').mean()),n=('vigilance','size'))
print(pt.groupby('side').f_asleep.agg(['count','mean']).round(3).to_string())
print('  Kruskal over patients (f_asleep by side) p=%.4f'%stats.kruskal(*[g.f_asleep.values for _,g in pt.groupby('side')]).pvalue)
