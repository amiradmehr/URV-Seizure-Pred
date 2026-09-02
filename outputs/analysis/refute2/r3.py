import numpy as np, pandas as pd
from scipy import stats
from pathlib import Path
ROOT=Path('/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'); V=ROOT/'outputs/analysis/vigilance'
S=pd.read_csv(V/'per_seizure_full.csv',dtype={'subject':str})
S=S[S.vigilance.isin(['asleep','awake'])].copy()

def both(col,label):
    p=S.pivot_table(index='tag',columns='vigilance',values=col,aggfunc='mean').dropna()
    d=p['asleep']-p['awake']; t=stats.ttest_rel(p['asleep'],p['awake'])
    sz=S.dropna(subset=[col]).groupby(['seizure_id','subject','vigilance'])[col].mean().reset_index()
    a=sz[sz.vigilance=='asleep'][col]; b=sz[sz.vigilance=='awake'][col]
    # cluster bootstrap over patients
    rng=np.random.default_rng(11); by={s:g for s,g in sz.groupby('subject')}; subs=sz.subject.unique()
    bo=[]
    for i in range(2000):
        g=pd.concat([by[s] for s in rng.choice(subs,len(subs),True)])
        x=g[g.vigilance=='asleep'][col]; y=g[g.vigilance=='awake'][col]
        if len(x)<2 or len(y)<2: continue
        bo.append(x.mean()-y.mean())
    bo=np.array(bo); pb=2*min((bo<=0).mean(),(bo>=0).mean())
    print('%-38s CONFIG-paired diff=%+.4f p=%.3g (%d/%d+) | SEIZURE-unit diff=%+.4f (n=%d/%d) MWU p=%.3f | patient-cluster-boot p=%.3f CI[%+.4f,%+.4f]'%(
        label,d.mean(),t.pvalue,(d>0).sum(),len(d),a.mean()-b.mean(),len(a),len(b),
        stats.mannwhitneyu(a,b).pvalue,pb,np.percentile(bo,2.5),np.percentile(bo,97.5)))

print('=== C3/C4: same quantity, invalid vs valid unit ===')
both('caught','sens GLOBAL thr')
both('caught_pat','sens PER-PATIENT thr')
both('caught_rec','sens PER-RECORDING thr')
both('caught_loc','sens PER-REC +-3h thr')
both('rec_neg_mean','recording mean NEG prob')
both('pct_rec','peak pct vs same-rec negs')
both('pct_loc','peak pct vs same-rec +-3h')
both('peak','raw peak preictal prob')
both('pct_glob','peak pct vs ALL negs')
both('n_pos','n positive decisions')

print('\n=== how many seizures survive each calibration (NaN drops) ===')
for c in ['caught','caught_pat','caught_rec','caught_loc']:
    g=S.dropna(subset=[c]).drop_duplicates('seizure_id')
    print('  %-12s n=%d  asleep=%d awake=%d'%(c,len(g),(g.vigilance=='asleep').sum(),(g.vigilance=='awake').sum()))

print('\n=== what does per-recording calibration DO? alarms budget per recording ===')
R=pd.read_csv(V/'per_recording_by_config.csv')
r1=R[R.tag==R.tag.iloc[0]]
print('recordings in OOF: %d ; n_neg per recording: median=%.0f  q10=%.0f q90=%.0f'%(len(r1),r1.n_neg.median(),r1.n_neg.quantile(.1),r1.n_neg.quantile(.9)))
print('recordings with <60 negatives (i.e. <1h, per-rec 1FA/h threshold = max):',(r1.n_neg<60).sum())
print('implied per-recording alarm allowance int(n_neg/60): median=%.0f, min=%d'%( (r1.n_neg//60).median(),(r1.n_neg//60).min()))
