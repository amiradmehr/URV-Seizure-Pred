import numpy as np, pandas as pd, os
from scipy import stats
from sklearn.metrics import average_precision_score
ROOT='/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'
rng=np.random.default_rng(7)
res=pd.read_csv(f'{ROOT}/outputs/sweep/results.csv')
C=pd.read_csv(f'{ROOT}/outputs/analysis/chance_seizure_sens.csv')
print(C.sort_values('obs',ascending=False).head(4)[['tag','sop','obs','chance_block']].to_string(index=False))

# AP/chance clustered-bootstrap SD on 4 configs spanning the sweep
for tag in ['spectral-gru__large__h45__sop10','spectral-meanpool__small__h5__sop10',
            'logistic-mean__small__h45__sop30','spectral-attention__large__h5__sop30']:
    d=pd.read_csv(f'{ROOT}/outputs/cv/{tag}/out_of_fold_predictions.csv',usecols=['subject','label','probability'])
    subs=d.subject.unique(); idxs=[d.index[d.subject.values==s].to_numpy() for s in subs]
    y=d.label.to_numpy(); s=d.probability.to_numpy(); o=[]
    for b in range(300):
        sel=np.concatenate([idxs[p] for p in rng.integers(0,len(subs),len(subs))])
        yy=y[sel]
        if yy.sum(): o.append(average_precision_score(yy,s[sel])/yy.mean())
    o=np.array(o)
    print(f"{tag:44s} AP/chance={res.loc[res.tag==tag,'ap_over_chance'].iloc[0]:.3f}  "
          f"clustered-boot SD={o.std(ddof=1):.4f}  95%CI=[{np.percentile(o,2.5):.3f},{np.percentile(o,97.5):.3f}]")

# achieved power of the CURRENT data to detect a true seizure-level sensitivity
za=stats.norm.ppf(0.975); n_eff=169.5; p0=0.055   # measured chance floor at SOP=10
print(f"\nPower of the EXISTING 286 seizures (n_eff={n_eff:.0f}) to reject "
      f"chance floor p0={p0:.3f}, alpha=.05 two-sided:")
for p1 in (0.08,0.10,0.15,0.20,0.30,0.40):
    se0=np.sqrt(p0*(1-p0)/n_eff); se1=np.sqrt(p1*(1-p1)/n_eff)
    pw=stats.norm.cdf((abs(p1-p0)-za*se0)/se1)
    print(f"   true sens {p1:.2f} -> power {pw:6.1%}")
# upper confidence bound on the true effect given a null result
best=C.loc[C.sop==10].assign(d=lambda x:x.obs-x.chance_block)
print(f"\nSOP=10: observed minus chance, across 21 configs: mean={best.d.mean():+.4f} "
      f"sd={best.d.std():.4f} max={best.d.max():+.4f}")
se=np.sqrt(0.055*0.945/n_eff)
print(f"   SE of a single sensitivity estimate at n_eff={n_eff:.0f} is {se:.4f};")
print(f"   one-sided 95% upper bound on the true lift for the BEST config "
      f"= {best.d.max():.4f} + 1.645*{se*np.sqrt(2):.4f} = {best.d.max()+1.645*se*np.sqrt(2):.4f}")
