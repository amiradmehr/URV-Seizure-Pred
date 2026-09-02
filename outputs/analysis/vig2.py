import pandas as pd, numpy as np
from scipy import stats

M = pd.read_csv('outputs/analysis/caught_matrix.csv', index_col=0)
sz = pd.read_csv('data/interim/manifests/seizure_manifest.csv', dtype={'subject':str})
sz = sz.set_index('seizure_id')
v = sz.loc[M.index,'vigilance'].astype(str).values
subj = sz.loc[M.index,'subject'].astype(str).values
print('seizures', M.shape[0], 'configs', M.shape[1])
print(pd.Series(v).value_counts().to_dict())
A = M.values.astype(float)   # seizures x configs

def stat(vv):
    a = A[vv=='asleep'].mean(0); w = A[vv=='awake'].mean(0)
    return (a-w).mean(), a.mean(), w.mean()

obs, oa, ow = stat(v)
print('OBSERVED mean-over-configs asleep %.4f awake %.4f diff %.4f'%(oa,ow,obs))

mask = np.isin(v,['asleep','awake'])
vv0 = v[mask]; A2 = A[mask]; s2 = subj[mask]
rng = np.random.default_rng(0)

# --- Test 1: seizure-level permutation (unrestricted) ---
def stat2(lab, Amat):
    return Amat[lab=='asleep'].mean(0).mean() - Amat[lab=='awake'].mean(0).mean()
o2 = stat2(vv0, A2)
null=[]
for _ in range(20000):
    null.append(stat2(rng.permutation(vv0), A2))
null=np.array(null)
p1 = (np.abs(null)>=abs(o2)).mean()
print('\nT1 seizure-level permutation: obs %.4f, null sd %.4f, two-sided p = %.4f'%(o2,null.std(),p1))

# --- Test 2: permutation restricted WITHIN patient (removes patient confounding) ---
null2=[]
idx_by_s = {s: np.where(s2==s)[0] for s in np.unique(s2)}
for _ in range(20000):
    lab = vv0.copy()
    for s,ix in idx_by_s.items():
        if len(ix)>1: lab[ix]=rng.permutation(lab[ix])
    null2.append(stat2(lab, A2))
null2=np.array(null2)
p2 = (np.abs(null2)>=abs(o2)).mean()
print('T2 within-patient permutation: null sd %.4f, two-sided p = %.4f'%(null2.std(),p2))

# --- Test 3: patient-level cluster bootstrap on the config-averaged per-seizure caught rate ---
r = A2.mean(1)   # per-seizure fraction of configs that caught it
pats = np.unique(s2)
boots=[]
for _ in range(20000):
    pick = rng.choice(pats, size=len(pats), replace=True)
    ix = np.concatenate([idx_by_s[p] for p in pick])
    l = vv0[ix]; rr = r[ix]
    if (l=='asleep').sum()==0 or (l=='awake').sum()==0: continue
    boots.append(rr[l=='asleep'].mean()-rr[l=='awake'].mean())
boots=np.array(boots)
print('T3 patient cluster bootstrap: obs %.4f  95%% CI [%.4f, %.4f]  frac<=0 %.4f'%(
    r[vv0=='asleep'].mean()-r[vv0=='awake'].mean(), np.percentile(boots,2.5), np.percentile(boots,97.5), (boots<=0).mean()))

# --- Test 4: per-config Fisher exact, distribution of p-values ---
ps=[]; sig=0
for j in range(A2.shape[1]):
    a=A2[vv0=='asleep',j]; w=A2[vv0=='awake',j]
    tab=[[int(a.sum()),int(len(a)-a.sum())],[int(w.sum()),int(len(w)-w.sum())]]
    _,pv=stats.fisher_exact(tab)
    ps.append(pv); sig += pv<0.05
ps=np.array(ps)
print('T4 per-config Fisher: median p %.3f, min p %.4f, n significant at .05: %d/42'%(np.median(ps),ps.min(),sig))

# --- Test 5: patient composition / clustering of vigilance ---
tab = pd.crosstab(s2, vv0)
both = ((tab>0).sum(1)==2).sum()
print('\nT5 patients with eligible seizures: %d; with BOTH asleep and awake: %d; asleep-only %d; awake-only %d'%(
    len(tab), both, ((tab['asleep']>0)&(tab['awake']==0)).sum(), ((tab['awake']>0)&(tab['asleep']==0)).sum()))
# per-patient mean caught rate
dfp = pd.DataFrame({'s':s2,'v':vv0,'r':r})
g = dfp.groupby(['s','v']).r.mean().unstack()
gg = g.dropna()
print('   paired within-patient (n=%d patients with both): asleep %.4f awake %.4f, wilcoxon p=%.4f'%(
    len(gg), gg['asleep'].mean(), gg['awake'].mean(),
    stats.wilcoxon(gg['asleep'],gg['awake']).pvalue if len(gg)>5 else float('nan')))
