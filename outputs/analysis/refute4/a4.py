import numpy as np, pandas as pd, pickle
from scipy import stats, optimize
from sklearn.metrics import average_precision_score
ROOT='/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'
res=pd.read_csv(f'{ROOT}/outputs/sweep/results.csv'); r42=res[res.sop_min.notna()]
C=pd.read_csv(f'{ROOT}/outputs/analysis/refute4/chance_matched.csv')
za=stats.norm.ppf(0.975); zb=stats.norm.ppf(0.80)
def n_two(p1,p2):
    pb=(p1+p2)/2
    return (za*np.sqrt(2*pb*(1-pb))+zb*np.sqrt(p1*(1-p1)+p2*(1-p2)))**2/(p2-p1)**2
def n_one(p0,p1): return (za*np.sqrt(p0*(1-p0))+zb*np.sqrt(p1*(1-p1)))**2/(p1-p0)**2
print('n_two(0.10,0.30) per arm = %.1f  -> raw per arm %.0f  -> TOTAL both arms %.0f'%(n_two(.1,.3),n_two(.1,.3)*1.687,2*n_two(.1,.3)*1.687))
print('n_one(0.10,0.30) = %.1f -> raw %.0f'%(n_one(.1,.3),n_one(.1,.3)*1.687))
print('n_two(0.055,0.0596) per arm = %.0f -> raw per arm %.0f -> TOTAL %.0f'%(n_two(.055,.055+.00455),n_two(.055,.055+.00455)*1.687,2*n_two(.055,.055+.00455)*1.687))
# power curve check
n_eff=169.5; p0=0.055
for p1 in (0.10,0.15,0.20,0.30):
    se0=np.sqrt(p0*(1-p0)/n_eff); se1=np.sqrt(p1*(1-p1)/n_eff)
    print('  power at true sens %.2f = %.1f%%'%(p1,100*stats.norm.cdf((abs(p1-p0)-za*se0)/se1)))
# per-config chance floors spread
for s in (10,30):
    g=C[C.sop==s]
    print('SOP=%d per-config chance floor: min %.4f max %.4f  (analyst treats it as the constant %.3f)'%(s,g.pooled.min(),g.pooled.max(),g.pooled.mean()))
    print('   n required (one-sample, 80%%) vs floor=min %.0f  vs floor=mean %.0f  vs floor=max %.0f   for true sens 0.30'%(
        n_one(g.pooled.min(),.30)*1.687,n_one(g.pooled.mean(),.30)*1.687,n_one(g.pooled.max(),.30)*1.687))
# CI on the mean lift (claim 6's delta)
det=pickle.load(open(f'{ROOT}/outputs/analysis/refute4/det.pkl','rb'))
tags=[t for t in r42.tag]
d0=pd.read_csv(f'{ROOT}/outputs/cv/{tags[0]}/out_of_fold_predictions.csv',usecols=['subject','label','target_seizure_id'])
meta=d0[d0.label==1].groupby('target_seizure_id').subject.first()
subj=meta; subs=subj.unique(); pos_by=[np.where(subj.values==s)[0] for s in subs]
rng=np.random.default_rng(9)
s10=[t for t in tags if t.endswith('sop10')]
Dm=pd.DataFrame({t:det[t]['obs'] for t in s10}).reindex(meta.index).to_numpy().astype(float)
floors=C.set_index('tag').loc[s10,'pooled'].to_numpy()
B=3000; bb=np.empty(B)
for b in range(B):
    sel=np.concatenate([pos_by[q] for q in rng.integers(0,len(subs),len(subs))])
    bb[b]=(Dm[sel].mean(0)-floors).mean()
obs=(Dm.mean(0)-floors).mean()
lo,hi=np.percentile(bb,[2.5,97.5])
print('\nMEAN SOP10 chance-corrected lift = %+.5f  clustered-bootstrap 95%% CI [%+.5f, %+.5f]  (contains 0: %s)'%(obs,lo,hi,lo<0<hi))
for dl in (hi, 0.0338, 0.05):
    if dl>0: print('   n required at delta=%+.4f : %.0f raw eligible seizures (one-sample vs 0.055)'%(dl,n_one(0.055,0.055+dl)*1.687))
# correlation among configs -> effective number of independent tests
Dall=pd.DataFrame({t:det[t]['obs'] for t in tags}).reindex(meta.index).astype(float)
cm=Dall.corr().to_numpy(); off=cm[np.triu_indices_from(cm,1)]
print('\nmean pairwise correlation of per-seizure detection across the 42 configs: %.3f (median %.3f)'%(off.mean(),np.median(off)))
ap=r42.ap_over_chance.to_numpy()
print('AP/chance over 42 (dedup, sweep only): mean %.4f sd %.4f max %.4f'%(ap.mean(),ap.std(ddof=1),ap.max()))
# claim 7 curve consistency
ICC=0.160;k=93
for m in (3.0753,6,10,20):
    print('claim7 curve: m=%.2f -> N=%.0f n_eff=%.1f'%(m,k*m,k*m/(1+(m-1)*ICC)))
print('asymptote k/ICC: ICC=0.160 ->%.0f ; ICC=0.175(42 cfgs) ->%.0f ; ICC=0.118(fold-resid) ->%.0f'%(93/0.160,93/0.1748,93/0.1176))
