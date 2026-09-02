import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
ROOT=Path('/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'); OUT=ROOT/'outputs/analysis/vigilance'
S=pd.read_csv(OUT/'per_seizure_full.csv',dtype={'subject':str})
A=S[S.vigilance.isin(['asleep','awake'])].copy()
dec=pd.read_csv(ROOT/'data/interim/manifests/decision_manifest.csv',usecols=['recording_id','bte_side']).groupby('recording_id').bte_side.first()
A['bte_side']=A.recording_id.map(dec)
rng=np.random.default_rng(0)

def cluster_test(per_sz, col, label, nboot=3000):
    d=per_sz.dropna(subset=[col])
    a=d[d.vigilance=='asleep'][col]; b=d[d.vigilance=='awake'][col]
    obs=a.mean()-b.mean()
    subs=d.subject.unique(); idx={s:d.index[d.subject==s].to_numpy() for s in subs}
    bs=[]
    for _ in range(nboot):
        pick=rng.choice(subs,size=len(subs),replace=True)
        g=d.loc[np.concatenate([idx[s] for s in pick])]
        aa=g[g.vigilance=='asleep'][col]; bb=g[g.vigilance=='awake'][col]
        if len(aa)and len(bb): bs.append(aa.mean()-bb.mean())
    bs=np.array(bs); lo,hi=np.percentile(bs,[2.5,97.5])
    pb=2*min((bs<=0).mean(),(bs>=0).mean())
    w=stats.ttest_ind(a,b,equal_var=False)
    print('%-34s asleep=%.4f(n=%d) awake=%.4f(n=%d) diff=%+.4f | Welch p=%.4f | patient-boot 95%%CI [%+.4f,%+.4f] p=%.4f'%(
        label,a.mean(),len(a),b.mean(),len(b),obs,w.pvalue,lo,hi,max(pb,1/nboot)))
    return obs

print('=== proper-unit versions of the analyst\'s config-paired tests ===')
for col,lab in [('caught','caught @1FA/h GLOBAL thr'),('caught_pat','caught PER-PATIENT thr'),
                ('caught_rec','caught PER-RECORDING thr'),('caught_loc','caught PER-REC +-3h thr'),
                ('pct_rec','peak pctile vs same-rec negs'),('pct_loc','peak pctile vs same-rec +-3h'),
                ('peak','raw peak preictal prob'),('n_pos','n positive decisions'),
                ('rec_neg_mean','recording mean NEG prob')]:
    per_sz=A.groupby(['seizure_id','subject','vigilance']) [col].mean().reset_index()
    cluster_test(per_sz,col,lab)

print('\n=== analyst\'s config-paired p-values for the SAME quantities (df=41) ===')
for col,lab in [('caught','caught GLOBAL'),('caught_pat','caught PER-PATIENT'),('caught_rec','caught PER-REC'),
                ('caught_loc','caught PER-REC+-3h'),('pct_rec','pct_rec'),('pct_loc','pct_loc'),('peak','peak'),
                ('n_pos','n_pos'),('rec_neg_mean','rec_neg_mean')]:
    p=A.pivot_table(index='tag',columns='vigilance',values=col,aggfunc='mean').dropna()
    t=stats.ttest_rel(p['asleep'],p['awake'])
    print('  %-22s diff=%+.4f t=%+.2f p=%.4g (%d/%d +)'%(lab,(p['asleep']-p['awake']).mean(),t.statistic,t.pvalue,(p['asleep']>p['awake']).sum(),len(p)))

print('\n=== analyst claim1: Welch with equal_var=True? ===')
ps=A.groupby(['seizure_id','subject','vigilance']).caught.mean().reset_index()
a=ps[ps.vigilance=='asleep'].caught;b=ps[ps.vigilance=='awake'].caught
print('  equal_var=False p=%.4f  equal_var=True p=%.4f  MWU p=%.4f'%(
    stats.ttest_ind(a,b,equal_var=False).pvalue,stats.ttest_ind(a,b).pvalue,stats.mannwhitneyu(a,b).pvalue))
