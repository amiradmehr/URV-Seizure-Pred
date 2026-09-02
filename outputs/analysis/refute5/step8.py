import numpy as np, pandas as pd, sys
from sklearn.metrics import average_precision_score
ROOT='/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'
res=pd.read_csv(f'{ROOT}/outputs/sweep/results.csv').set_index('tag')
tag='spectral-gru__large__h45__sop10'
d=pd.read_csv(f'{ROOT}/outputs/cv/{tag}/out_of_fold_predictions.csv',usecols=['subject','label','probability'])
subs=d.subject.unique(); idxs=[np.where(d.subject.values==s)[0] for s in subs]
y=d.label.to_numpy(); s=d.probability.to_numpy()
for seed in (1,101):
    rng=np.random.default_rng(seed); o=[]
    for b in range(1500):
        sel=np.concatenate([idxs[p] for p in rng.integers(0,len(subs),len(subs))])
        yy=y[sel]
        if yy.sum(): o.append(average_precision_score(yy,s[sel])/yy.mean())
    o=np.array(o)
    print(f'seed={seed} B={len(o)} pt={res.loc[tag,"ap_over_chance"]:.4f} SD={o.std(ddof=1):.4f} '
          f'CI=[{np.percentile(o,2.5):.4f},{np.percentile(o,97.5):.4f}] P(<=1)={np.mean(o<=1):.4f}',flush=True)
