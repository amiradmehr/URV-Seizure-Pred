import numpy as np, pandas as pd
from scipy import stats
from pathlib import Path
ROOT=Path('/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'); V=ROOT/'outputs/analysis/vigilance'
S=pd.read_csv(V/'per_seizure_full.csv',dtype={'subject':str})
S=S[S.vigilance.isin(['asleep','awake'])].copy()
sz=S.groupby(['seizure_id','subject','vigilance','recording_id']).caught.mean().reset_index()

pc=sz.groupby('subject').vigilance.agg(lambda s:'asleep-only' if (s=='asleep').all() else ('awake-only' if (s=='awake').all() else 'mixed'))
print('patient groups:',pc.value_counts().to_dict(),'total patients',len(pc))
mixed=pc[pc=='mixed'].index
M=sz[sz.subject.isin(mixed)]
print('\nMIXED subset: %d patients, %d seizures (asleep %d, awake %d)'%(
    M.subject.nunique(),len(M),(M.vigilance=='asleep').sum(),(M.vigilance=='awake').sum()))

# a3-style: per patient mean caught per vigilance, then config-paired  (their number)
Sm=S[S.subject.isin(mixed)]
w=Sm.groupby(['tag','subject','vigilance']).caught.mean().reset_index().pivot_table(index=['tag','subject'],columns='vigilance',values='caught').dropna()
aa=w.groupby('tag')['asleep'].mean(); bb=w.groupby('tag')['awake'].mean()
tt=stats.ttest_rel(aa,bb)
print('[C2 replicate] within-patient config-paired: asleep=%.4f awake=%.4f diff=%+.4f t=%.2f p=%.4f (%d/42 +)'%(
    aa.mean(),bb.mean(),(aa-bb).mean(),tt.statistic,tt.pvalue,((aa-bb)>0).sum()))

# PROPER within-patient test: patient as unit, paired difference of patient means
pw=M.groupby(['subject','vigilance']).caught.mean().unstack()
dd=(pw['asleep']-pw['awake']).dropna()
tp=stats.ttest_rel(pw['asleep'],pw['awake'])
ci=stats.t.interval(0.95,len(dd)-1,loc=dd.mean(),scale=stats.sem(dd))
print('[C2 proper] patient-paired (n=%d patients): diff=%+.4f  95%%CI [%+.4f,%+.4f]  t=%.2f p=%.4f  wilcoxon p=%.4f'%(
    len(dd),dd.mean(),ci[0],ci[1],tp.statistic,tp.pvalue,stats.wilcoxon(dd).pvalue if (dd!=0).any() else np.nan))
print('   patients with nonzero diff: %d/%d  (%d>0, %d<0, %d exactly 0)'%(
    (dd!=0).sum(),len(dd),(dd>0).sum(),(dd<0).sum(),(dd==0).sum()))

# POWER of the within-patient test to detect the between-patient effect size +0.0405 and the crude +0.0188
sd=dd.std(ddof=1)
for eff in [0.0188,0.0405,0.10]:
    from scipy.stats import nct
    ncp=eff/(sd/np.sqrt(len(dd)))
    crit=stats.t.ppf(0.975,len(dd)-1)
    pwr=1-nct.cdf(crit,len(dd)-1,ncp)+nct.cdf(-crit,len(dd)-1,ncp)
    print('   power to detect diff=%+.4f  (sd of patient diffs=%.4f) = %.2f'%(eff,sd,pwr))
# minimum detectable effect at 80% power
mde=sd/np.sqrt(len(dd))*(stats.t.ppf(0.975,len(dd)-1)+stats.t.ppf(0.80,len(dd)-1))
print('   minimum detectable effect @80%% power = %+.4f'%mde)

# --- patient-averaged unpaired over ALL patients, done with PATIENT as unit
pp=sz.groupby(['subject','vigilance']).caught.mean().reset_index()
A=pp[pp.vigilance=='asleep'].caught; B=pp[pp.vigilance=='awake'].caught
print('\n[C2] patient-averaged, PATIENT as unit: asleep=%.4f (n=%d patient-arms) awake=%.4f (n=%d) diff=%+.4f Welch p=%.4f MWU p=%.4f'%(
    A.mean(),len(A),B.mean(),len(B),A.mean()-B.mean(),
    stats.ttest_ind(A,B,equal_var=False).pvalue, stats.mannwhitneyu(A,B).pvalue))
# their version (config-paired over patient means) - the invalid unit
q=S.groupby(['tag','subject','vigilance']).caught.mean().reset_index().pivot_table(index='tag',columns='vigilance',values='caught',aggfunc='mean')
t2=stats.ttest_rel(q['asleep'],q['awake'])
print('[C2 replicate] their config-paired patient-averaged: asleep=%.4f awake=%.4f diff=%+.4f t=%.2f p=%.4g'%(
    q['asleep'].mean(),q['awake'].mean(),(q['asleep']-q['awake']).mean(),t2.statistic,t2.pvalue))

# --- the within-patient permutation: how many seizures actually get shuffled?
n_shuf=len(M); n_fixed=len(sz)-len(M)
print('\n[C2] within-patient permutation shuffles only %d/%d seizures (%.0f%%); %d seizures in single-vigilance patients keep their TRUE label'%(
    n_shuf,len(sz),100*n_shuf/len(sz),n_fixed))
