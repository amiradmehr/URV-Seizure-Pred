import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score
ROOT='/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'
res=pd.read_csv(f'{ROOT}/outputs/sweep/results.csv')
rng=np.random.default_rng(21)
tags=['spectral-gru__large__h45__sop10','spectral-meanpool__small__h5__sop10',
      'logistic-mean__small__h45__sop30','spectral-attention__large__h5__sop30']
sds={}
for tag in tags:
    d=pd.read_csv(f'{ROOT}/outputs/cv/{tag}/out_of_fold_predictions.csv',usecols=['subject','label','probability'])
    subs=d.subject.unique(); idxs=[np.where(d.subject.values==s)[0] for s in subs]
    y=d.label.to_numpy(); s=d.probability.to_numpy(); o=[]
    for b in range(2000):
        sel=np.concatenate([idxs[p] for p in rng.integers(0,len(subs),len(subs))])
        yy=y[sel]
        if yy.sum(): o.append(average_precision_score(yy,s[sel])/yy.mean())
    o=np.array(o); sds[tag]=o.std(ddof=1)
    print(f'{tag:44s} pt={res.loc[res.tag==tag,"ap_over_chance"].iloc[0]:.4f} SD={o.std(ddof=1):.4f} '
          f'95%CI=[{np.percentile(o,2.5):.3f},{np.percentile(o,97.5):.3f}] P(<=1)={np.mean(o<=1):.3f} B={len(o)}')
sd1=sds['spectral-gru__large__h45__sop10']
r42=res[~res.tag.isin(['spectral','linear'])]
print(f'\nAP/chance sweep: n=44 mean={res.ap_over_chance.mean():.4f} sd={res.ap_over_chance.std():.4f}')
print(f'                 n=42 mean={r42.ap_over_chance.mean():.4f} sd={r42.ap_over_chance.std():.4f}  (2 rows are legacy runs "spectral","linear" with NaN sop/history)')
z42=stats.norm.ppf(1-0.05/42)
print(f'z(Bonf42)={z42:.3f}')
print(f'  analyst bar using ACROSS-CONFIG sd 0.0644 : {res.ap_over_chance.mean()+z42*res.ap_over_chance.std():.3f}')
print(f'  correct bar using SINGLE-CONFIG clustered SD {sd1:.4f}: {1+z42*sd1:.3f}  (uncorrected 1-test bar {1+stats.norm.ppf(.95)*sd1:.3f})')
rho=1-(res.ap_over_chance.var()/sd1**2)
print(f'  implied mean pairwise corr between configs rho={rho:.3f} -> effective independent tests '
      f'{1+(42-1)*(1-rho):.1f} -> Bonferroni bar {1+stats.norm.ppf(1-0.05/(1+41*(1-rho)))*sd1:.3f}')
