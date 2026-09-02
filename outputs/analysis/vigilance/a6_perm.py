import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
ROOT=Path('/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'); OUT=ROOT/'outputs/analysis/vigilance'
S=pd.read_csv(OUT/'per_seizure_full.csv',dtype={'subject':str})
S=S[S.vigilance.isin(['asleep','awake'])].copy()
dec=pd.read_csv(ROOT/'data/interim/manifests/decision_manifest.csv',
                usecols=['recording_id','bte_side'],dtype={'subject':str}).groupby('recording_id').bte_side.first()
S['bte_side']=S.recording_id.map(dec)
key=S[S.tag==S.tag.iloc[0]][['seizure_id','subject','vigilance','bte_side']].drop_duplicates('seizure_id').set_index('seizure_id')
print('n seizures',len(key))

def stat(labelmap):
    x=S.copy(); x['v']=x.seizure_id.map(labelmap)
    p=x.pivot_table(index='tag',columns='v',values='caught',aggfunc='mean')
    return stats.ttest_rel(p['asleep'],p['awake']).statistic, (p['asleep']-p['awake']).mean()

obs_t,obs_d=stat(key.vigilance.to_dict())
print('OBSERVED t=%.3f diff=%+.4f'%(obs_t,obs_d))
rng=np.random.default_rng(0); NP=2000
def perm(within):
    out=[]
    for i in range(NP):
        k=key.copy()
        if within:
            k['v']=k.groupby('subject').vigilance.transform(lambda s: rng.permutation(s.values))
        else:
            k['v']=rng.permutation(k.vigilance.values)
        t,_=stat(k.v.to_dict()); out.append(t)
    return np.array(out)
n1=perm(False); n2=perm(True)
print('perm p (labels shuffled FREELY, ignores patient)  = %.4f  [null t: %.2f +- %.2f]'%(((np.abs(n1)>=abs(obs_t)).mean()),n1.mean(),n1.std()))
print('perm p (labels shuffled WITHIN patient)           = %.4f  [null t: %.2f +- %.2f]'%(((np.abs(n2)>=abs(obs_t)).mean()),n2.mean(),n2.std()))

print('\n=== patient composition ===')
pc=key.groupby('subject').vigilance.agg(lambda s:'asleep-only' if (s=='asleep').all() else ('awake-only' if (s=='awake').all() else 'mixed'))
print(pc.value_counts())
S['pgroup']=S.subject.map(pc)
g=S.groupby(['tag','pgroup']).caught.mean().unstack()
print('mean sens @1FA/h by patient group (avg over 42 configs):'); print(g.mean().round(4))
bg=S.groupby(['tag','pgroup']).rec_neg_mean.mean().unstack()
print('mean NEGATIVE background prob by patient group:'); print(bg.mean().round(4))

print('\n=== bte_side (montage) ===')
b=S.groupby(['tag','bte_side']).caught.mean().unstack()
print('sens @1FA/h by montage:'); print(b.mean().round(4))
bb=S.groupby(['tag','bte_side']).rec_neg_mean.mean().unstack()
print('background prob by montage:'); print(bb.mean().round(4))
# asleep effect within montage strata
for side in ['left','right','bilateral']:
    f=S[S.bte_side==side]
    p=f.pivot_table(index='tag',columns='vigilance',values='caught',aggfunc='mean').dropna()
    if p.shape[1]<2 or len(p)<5: print(f'  {side}: too few'); continue
    t=stats.ttest_rel(p['asleep'],p['awake'])
    print(f'  {side:9s} n_asleep={ (f[f.tag==f.tag.iloc[0]].vigilance=="asleep").sum():3d} n_awake={(f[f.tag==f.tag.iloc[0]].vigilance=="awake").sum():3d}  '
          f'diff={(p["asleep"]-p["awake"]).mean():+.4f} p={t.pvalue:.4g}')
