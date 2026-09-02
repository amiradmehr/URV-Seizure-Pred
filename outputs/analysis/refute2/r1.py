import numpy as np, pandas as pd
from scipy import stats
from pathlib import Path
ROOT=Path('/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred')
V=ROOT/'outputs/analysis/vigilance'
S=pd.read_csv(V/'per_seizure_full.csv',dtype={'subject':str})
S=S[S.vigilance.isin(['asleep','awake'])].copy()
print('rows',len(S),'configs',S.tag.nunique(),'seizures',S.seizure_id.nunique())
print(S.drop_duplicates('seizure_id').vigilance.value_counts().to_dict())

# ---- C1: replicate config-paired t
p=S.pivot_table(index='tag',columns='vigilance',values='caught',aggfunc='mean')
d=p['asleep']-p['awake']; t=stats.ttest_rel(p['asleep'],p['awake'])
print('\n[C1] config-paired: asleep=%.4f awake=%.4f diff=%+.4f t=%.3f p=%.6f  (%d/%d +)'%(
    p['asleep'].mean(),p['awake'].mean(),d.mean(),t.statistic,t.pvalue,(d>0).sum(),len(d)))

# seizure as unit (mean caught over configs)
sz=S.groupby(['seizure_id','subject','vigilance','recording_id']).caught.mean().reset_index()
a=sz[sz.vigilance=='asleep'].caught; b=sz[sz.vigilance=='awake'].caught
print('[C1] seizure-unit: asleep=%.4f (n=%d) awake=%.4f (n=%d) diff=%+.4f Welch p=%.4f MWU p=%.4f'%(
    a.mean(),len(a),b.mean(),len(b),a.mean()-b.mean(),
    stats.ttest_ind(a,b,equal_var=False).pvalue, stats.mannwhitneyu(a,b).pvalue))

# ---- C1b: CLUSTER-aware (patient) tests: seizure is NOT the sampling unit either
# cluster bootstrap over patients on the seizure-level mean difference
rng=np.random.default_rng(7)
subs=sz.subject.unique(); by={s:g for s,g in sz.groupby('subject')}
obs=a.mean()-b.mean()
boot=[]
for i in range(4000):
    pick=rng.choice(subs,size=len(subs),replace=True)
    g=pd.concat([by[s] for s in pick])
    aa=g[g.vigilance=='asleep'].caught; bb=g[g.vigilance=='awake'].caught
    if len(aa)<2 or len(bb)<2: continue
    boot.append(aa.mean()-bb.mean())
boot=np.array(boot)
print('[C1b] patient cluster bootstrap on seizure-level diff: obs=%+.4f  95%%CI [%+.4f,%+.4f]  2-sided p=%.4f'%(
    obs,np.percentile(boot,2.5),np.percentile(boot,97.5), 2*min((boot<=0).mean(),(boot>=0).mean())))

# ---- C1c: permutation FREE over seizures, replicating a6
key=sz.set_index('seizure_id').vigilance.to_dict()
def stat_from_map(m):
    x=S.copy(); x['v']=x.seizure_id.map(m)
    q=x.pivot_table(index='tag',columns='v',values='caught',aggfunc='mean')
    return stats.ttest_rel(q['asleep'],q['awake']).statistic
ids=np.array(list(key.keys())); labs=np.array([key[i] for i in ids])
rng2=np.random.default_rng(0)
null=[]
for i in range(1500):
    pm=dict(zip(ids,rng2.permutation(labs)))
    null.append(stat_from_map(pm))
null=np.array(null)
print('[C1c] FREE perm null t: mean=%.2f sd=%.2f ; perm p(|t|>=%.2f)=%.4f'%(
    null.mean(),null.std(),t.statistic,(np.abs(null)>=abs(t.statistic)).mean()))

# ---- C1d: permutation of the SEIZURE-LEVEL diff, and CLUSTERED perm (swap whole patients' labels)
obs_d=obs
nulld=[]
for i in range(4000):
    pl=rng2.permutation(labs)
    nulld.append(sz.caught.values[pl=='asleep'].mean()-sz.caught.values[pl=='awake'].mean())
nulld=np.array(nulld)
print('[C1d] seizure-level free perm p=%.4f'%( (np.abs(nulld)>=abs(obs_d)).mean()))
