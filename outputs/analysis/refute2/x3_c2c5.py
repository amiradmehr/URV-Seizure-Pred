import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
ROOT=Path('/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'); OUT=ROOT/'outputs/analysis/vigilance'
S=pd.read_csv(OUT/'per_seizure_full.csv',dtype={'subject':str})
A=S[S.vigilance.isin(['asleep','awake'])].copy()
dec=pd.read_csv(ROOT/'data/interim/manifests/decision_manifest.csv',usecols=['recording_id','bte_side','subject'],dtype={'subject':str})
side=dec.groupby('recording_id').bte_side.first(); A['bte_side']=A.recording_id.map(side)
rng=np.random.default_rng(0)

print('===== CLAIM 2 =====')
sub=A.dropna(subset=['caught'])
w=sub.groupby(['tag','subject','vigilance']).caught.mean().reset_index().pivot_table(index=['tag','subject'],columns='vigilance',values='caught').dropna()
print('mixed patients:',w.reset_index().subject.nunique())
t=stats.ttest_rel(w.groupby('tag')['asleep'].mean(),w.groupby('tag')['awake'].mean())
print("analyst WITHIN-PATIENT (config-paired): asleep=%.4f awake=%.4f diff=%+.4f t=%+.2f p=%.4g"%(
    w['asleep'].mean(),w['awake'].mean(),(w['asleep']-w['awake']).mean(),t.statistic,t.pvalue))
# proper: patient is the unit, paired within patient, averaged over 42 configs
pp=sub.groupby(['subject','vigilance']).caught.mean().unstack().dropna()
dd=pp['asleep']-pp['awake']
tt=stats.ttest_rel(pp['asleep'],pp['awake']); ww=stats.wilcoxon(pp['asleep'],pp['awake'])
print("PROPER within-patient paired (n=%d patients as units): diff=%+.4f t=%+.2f p=%.4f  wilcoxon p=%.4f"%(len(pp),dd.mean(),tt.statistic,tt.pvalue,ww.pvalue))
print("   95%% CI on within-patient diff: [%+.4f, %+.4f]   SD=%.4f"%(dd.mean()-1.96*dd.std()/np.sqrt(len(dd)),dd.mean()+1.96*dd.std()/np.sqrt(len(dd)),dd.std()))
q=sub.groupby(['tag','subject','vigilance']).caught.mean().reset_index().pivot_table(index='tag',columns='vigilance',values='caught',aggfunc='mean')
t2=stats.ttest_rel(q['asleep'],q['awake'])
print("analyst PATIENT-AVERAGED 'unpaired' (actually ttest_rel over 42 TAGS): asleep=%.4f awake=%.4f diff=%+.4f t=%.2f p=%.4g"%(
    q['asleep'].mean(),q['awake'].mean(),(q['asleep']-q['awake']).mean(),t2.statistic,t2.pvalue))
# proper: patients as units, between-patient
pa=sub.groupby(['subject','vigilance']).caught.mean().unstack()
ga=pa['asleep'].dropna(); gb=pa['awake'].dropna()
print("PROPER patient-as-unit between groups: asleep-mean=%.4f (n=%d pts) awake-mean=%.4f (n=%d pts) diff=%+.4f Welch p=%.4f MWU p=%.4f"%(
    ga.mean(),len(ga),gb.mean(),len(gb),ga.mean()-gb.mean(),stats.ttest_ind(ga,gb,equal_var=False).pvalue,stats.mannwhitneyu(ga,gb).pvalue))
# fully independent version: asleep-only vs awake-only patients
pc=sub.groupby('subject').vigilance.agg(lambda s:'asleep-only' if (s=='asleep').all() else ('awake-only' if (s=='awake').all() else 'mixed'))
pm=sub.groupby('subject').caught.mean()
print('patient groups:',pc.value_counts().to_dict())
for g in ['asleep-only','awake-only','mixed']:
    print('   %-12s n=%d mean sens=%.4f'%(g,(pc==g).sum(),pm[pc[pc==g].index].mean()))
ao=pm[pc[pc=='asleep-only'].index]; wo=pm[pc[pc=='awake-only'].index]
print('   asleep-only vs awake-only patients (INDEPENDENT): diff=%+.4f Welch p=%.4f MWU p=%.4f'%(
    ao.mean()-wo.mean(),stats.ttest_ind(ao,wo,equal_var=False).pvalue,stats.mannwhitneyu(ao,wo).pvalue))

print('\n===== CLAIM 5 : montage =====')
key=A[A.tag==A.tag.iloc[0]][['seizure_id','subject','vigilance','bte_side','recording_id']].drop_duplicates('seizure_id')
print('cell counts (eligible seizures):'); print(pd.crosstab(key.vigilance,key.bte_side))
print('proportions:'); print(pd.crosstab(key.vigilance,key.bte_side,normalize='index').round(3))
ct=pd.crosstab(key.vigilance,key.bte_side)
print('seizure-level chi2 p=%.4g  (treats %d seizures as independent)'%(stats.chi2_contingency(ct).pvalue,len(key)))
# patient-level: is montage patient-level?
ps=dec.groupby('subject').bte_side.nunique()
print('patients with >1 bte_side: %d of %d'%((ps>1).sum(),len(ps)))
# cluster-robust chi2 via patient permutation of vigilance labels (within patient)
per_sz=A.groupby(['seizure_id','subject','vigilance','bte_side']).caught.mean().reset_index()
mm=per_sz.groupby('bte_side').caught.mean()
print('sens by montage (seizure-level, avg 42 cfg):',mm.round(4).to_dict())
print('n by montage:',per_sz.bte_side.value_counts().to_dict())
# direct standardisation + patient cluster bootstrap
def std_gap(d):
    w=d.bte_side.value_counts(normalize=True)
    m=d.groupby(['vigilance','bte_side']).caught.mean().unstack()
    for s in w.index:
        if s not in m.columns: m[s]=np.nan
    m=m[w.index]
    def wm(row):
        v=row.values.astype(float); ok=np.isfinite(v)
        return np.nansum(v[ok]*w.values[ok])/w.values[ok].sum()
    return wm(m.loc['asleep'])-wm(m.loc['awake'])
obs_std=std_gap(per_sz); obs_raw=per_sz[per_sz.vigilance=='asleep'].caught.mean()-per_sz[per_sz.vigilance=='awake'].caught.mean()
subs=per_sz.subject.unique(); idx={s:per_sz.index[per_sz.subject==s].to_numpy() for s in subs}
bs=[];br=[]
for _ in range(3000):
    pick=rng.choice(subs,size=len(subs),replace=True)
    g=per_sz.loc[np.concatenate([idx[s] for s in pick])]
    if g.vigilance.nunique()<2: continue
    try:
        bs.append(std_gap(g)); br.append(g[g.vigilance=='asleep'].caught.mean()-g[g.vigilance=='awake'].caught.mean())
    except Exception: pass
bs=np.array(bs);bs=bs[np.isfinite(bs)];br=np.array(br)
print('crude gap %+.4f ; montage-standardised gap %+.4f'%(obs_raw,obs_std))
print('  standardised gap patient-boot 95%% CI [%+.4f, %+.4f]  SD=%.4f  p=%.3f'%(*np.percentile(bs,[2.5,97.5]),bs.std(),2*min((bs<=0).mean(),(bs>=0).mean())))
pctexp=100*(1-bs/br[np.isfinite(bs)][:len(bs)]) if False else None
print('  "%% explained" = 100*(1-std/crude); boot 95%% CI on that ratio:')
r=100*(1-bs/br[:len(bs)]); r=r[np.isfinite(r)]
print('     %.0f%% observed; boot 2.5-97.5 pct: [%.0f%%, %.0f%%]'%(100*(1-obs_std/obs_raw),*np.percentile(r,[2.5,97.5])))
# how much does the n=2 asleep-bilateral cell drive it?
bil=key[(key.vigilance=='asleep')&(key.bte_side=='bilateral')]
print('  asleep+bilateral seizures: n=%d ids=%s'%(len(bil),list(bil.seizure_id)))
cs=per_sz[(per_sz.vigilance=='asleep')&(per_sz.bte_side=='bilateral')].caught
print('  their mean caught over 42 configs:',list(cs.round(4)))
d2=per_sz[~((per_sz.vigilance=='asleep')&(per_sz.bte_side=='bilateral'))]
print('  standardised gap EXCLUDING those %d seizures: %+.4f'%(len(cs),std_gap(d2)))
