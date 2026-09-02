import numpy as np, pandas as pd
from scipy import stats
from pathlib import Path
ROOT=Path('/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'); V=ROOT/'outputs/analysis/vigilance'
print('=== C6: sleep proxy file ===')
N=pd.read_csv(V/'negatives_with_proxy.csv')
print(N.columns.tolist(), len(N), 'recordings', N.recording_id.nunique())
print(N[['probability','proxy']].describe().round(4))
# how many configs are in this file?
print('any tag column?', 'tag' in N.columns)
g=N.dropna(subset=['proxy'])
print('pooled spearman rho(proxy, probability) = %.4f (n=%d)'%(stats.spearmanr(g.proxy,g.probability).statistic,len(g)))
w=[]
for r,d in g.groupby('recording_id'):
    if len(d)>=50 and d.proxy.nunique()>5:
        w.append((r,len(d),stats.spearmanr(d.proxy,d.probability).statistic))
W=pd.DataFrame(w,columns=['rec','n','rho'])
print('within-recording rho: n_rec=%d median=%.4f mean=%.4f  IQR=[%.3f,%.3f]  frac>0=%.2f'%(
    len(W),W.rho.median(),W.rho.mean(),W.rho.quantile(.25),W.rho.quantile(.75),(W.rho>0).mean()))
t=stats.ttest_1samp(W.rho.dropna(),0); print('one-sample t on within-rec rho: t=%.2f p=%.3g'%(t.statistic,t.pvalue))
print('wilcoxon p=%.3g'%stats.wilcoxon(W.rho.dropna()).pvalue)
# weight by n
print('n-weighted mean within-rec rho = %.4f'%(np.average(W.rho,weights=W.n)))
