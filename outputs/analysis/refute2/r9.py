import numpy as np, pandas as pd
from scipy import stats
from pathlib import Path
ROOT=Path('/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'); V=ROOT/'outputs/analysis/vigilance'
S=pd.read_csv(V/'per_seizure_full.csv',dtype={'subject':str}); S=S[S.vigilance.isin(['asleep','awake'])]
sz=S.groupby(['seizure_id','subject','vigilance']).caught.mean().reset_index()
a=sz[sz.vigilance=='asleep'].caught; b=sz[sz.vigilance=='awake'].caught
print('C1 Welch (equal_var=False) p=%.4f ; pooled t-test (equal_var=True) p=%.4f  <- claim says "Welch p=0.1294"'%(
    stats.ttest_ind(a,b,equal_var=False).pvalue, stats.ttest_ind(a,b,equal_var=True).pvalue))
pc=sz.groupby('subject').vigilance.agg(lambda s:'asleep-only' if (s=='asleep').all() else ('awake-only' if (s=='awake').all() else 'mixed'))
S['pg']=S.subject.map(pc)
print('\nC2 mean sens @1FA/h by patient group (config-avg):')
print(S.groupby(['tag','pg']).caught.mean().unstack().mean().round(4).to_dict())
print('n seizures per group:', sz.assign(pg=sz.subject.map(pc)).pg.value_counts().to_dict())
# Is the between-vs-within DIFFERENCE itself significant?
mixed=pc[pc=='mixed'].index
M=sz[sz.subject.isin(mixed)]
pw=M.groupby(['subject','vigilance']).caught.mean().unstack(); dd=(pw['asleep']-pw['awake']).dropna()
pp=sz.groupby(['subject','vigilance']).caught.mean().reset_index()
A=pp[pp.vigilance=='asleep'].caught; B=pp[pp.vigilance=='awake'].caught
between=A.mean()-B.mean(); within=dd.mean()
se=np.sqrt(stats.sem(dd)**2 + stats.sem(A)**2 + stats.sem(B)**2)
print('\nC2 between=%+.4f within=%+.4f  difference=%+.4f  approx SE=%.4f  z=%.2f p=%.3f'%(
    between,within,between-within,se,(between-within)/se,2*(1-stats.norm.cdf(abs(between-within)/se))))
# C4: is pct_rec gap arithmetically entailed by background gap?
r=S.groupby(['seizure_id','vigilance']).agg(peak=('peak','mean'),bg=('rec_neg_mean','mean'),pct=('pct_rec','mean')).reset_index()
print('\nC4 partial: corr(pct_rec, rec_neg_mean)=%.3f ; corr(pct_rec, peak)=%.3f'%(
    r[['pct','bg']].corr().iloc[0,1], r[['pct','peak']].corr().iloc[0,1]))
import numpy.linalg as la
rr=r.dropna()
x=(rr.vigilance=='asleep').astype(float).values
z=(rr.bg-rr.bg.mean())/rr.bg.std()
X1=np.c_[np.ones(len(x)),x]; X2=np.c_[np.ones(len(x)),x,z]
b1=la.lstsq(X1,rr.pct.values,rcond=None)[0][1]; b2=la.lstsq(X2,rr.pct.values,rcond=None)[0][1]
print('C4 beta(asleep -> pct_rec) unadj=%+.4f  adjusted for recording background=%+.4f  (%.0f%% of the "reversal" is the background term)'%(
    b1,b2,100*(1-b2/b1)))
