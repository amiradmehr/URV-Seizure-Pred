import numpy as np, pandas as pd
from scipy import stats
from pathlib import Path
OUT=Path('/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred/outputs/analysis/vigilance')
S=pd.read_csv(OUT/'per_seizure_by_config.csv', dtype={'subject':str})
S=S[S.vigilance.isin(['asleep','awake'])].copy()
print('seizures per config:', S.groupby('tag').size().unique(), 'configs', S.tag.nunique())
print(S.groupby('vigilance').seizure_id.nunique())

def paired(col, label):
    p=S.pivot_table(index='tag', columns='vigilance', values=col, aggfunc='mean')
    d=p['asleep']-p['awake']
    t=stats.ttest_rel(p['asleep'],p['awake']); w=stats.wilcoxon(p['asleep'],p['awake'])
    print(f'{label:34s} asleep={p["asleep"].mean():.4f} awake={p["awake"].mean():.4f} '
          f'diff={d.mean():+.4f} t={t.statistic:+.2f} p={t.pvalue:.4g} wilcoxon p={w.pvalue:.4g} '
          f'({(d>0).sum()}/{len(d)} configs positive)')
    return p

print('\n=== 1. REPLICATION (seizure as unit, global 1 FA/h threshold) ===')
p_caught=paired('caught','seizure-level sens @1FA/h')
print('\n=== 2. THRESHOLD-FREE score comparisons ===')
paired('peak','peak preictal probability')
paired('pct_glob','peak percentile vs ALL negatives')
paired('pct_rec','peak percentile vs SAME-RECORDING negs')
paired('pct_loc','peak percentile vs SAME-REC +-3h negs')
print('\n=== 3. MECHANICAL: n positive decisions per seizure ===')
paired('n_pos','n positive decisions/seizure')
print('\n=== 4. BACKGROUND level of the seizure recording ===')
paired('rec_neg_mean','mean prob of NEGATIVES in that rec')
