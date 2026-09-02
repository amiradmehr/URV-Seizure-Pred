import numpy as np, pandas as pd, pickle
from scipy import stats
ROOT='/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'
Z=pickle.load(open(f'{ROOT}/outputs/analysis/refute5/cache.pkl','rb'))
D=Z['det'].astype(float); subj=Z['sz2subj'].reindex(D.index).to_numpy()
subs=np.unique(subj); k=len(subs); idx=[np.where(subj==s)[0] for s in subs]
res=pd.read_csv(f'{ROOT}/outputs/sweep/results.csv').set_index('tag')
rng=np.random.default_rng(77)
for sop in (10,30):
    cols=[t for t in D.columns if str(t).endswith(f'sop{sop}')]
    m=D[cols].mean().sort_values()
    lo_t,hi_t=m.index[0],m.index[-1]
    print(f'\nSOP={sop}: worst={lo_t} ({m.iloc[0]:.4f})  best={hi_t} ({m.iloc[-1]:.4f})  diff={m.iloc[-1]-m.iloc[0]:.4f}')
    d=(D[hi_t]-D[lo_t]).to_numpy()
    n10=int(((D[hi_t]==1)&(D[lo_t]==0)).sum()); n01=int(((D[hi_t]==0)&(D[lo_t]==1)).sum())
    print(f'   McNemar discordant b={n10} c={n01}  exact binomial p={stats.binomtest(n10,n10+n01,0.5).pvalue:.4g}')
    bb=np.empty(4000)
    for b in range(4000):
        sel=np.concatenate([idx[p] for p in rng.integers(0,k,k)]); bb[b]=d[sel].mean()
    print(f'   patient-clustered bootstrap: mean diff={d.mean():.4f} SE={bb.std(ddof=1):.4f} '
          f'95%CI=[{np.percentile(bb,2.5):.4f},{np.percentile(bb,97.5):.4f}] P(<=0)={np.mean(bb<=0):.4f}')
    # how many of the 210 pairs exceed the paired MDD 0.069?
    C=D[cols].to_numpy(); K=len(cols); big=0; tot=0
    for i in range(K):
        for j in range(i+1,K):
            tot+=1; big+= abs(C[:,i].mean()-C[:,j].mean())>0.0692
    print(f'   pairs with |diff| > paired MDD 0.0692: {big}/{tot}')
