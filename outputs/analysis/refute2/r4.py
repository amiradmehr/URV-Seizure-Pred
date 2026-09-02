import numpy as np, pandas as pd
from scipy import stats
from pathlib import Path
ROOT=Path('/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'); V=ROOT/'outputs/analysis/vigilance'
S=pd.read_csv(V/'per_seizure_full.csv',dtype={'subject':str})
S=S[S.vigilance.isin(['asleep','awake'])].copy()
sz=pd.read_csv(ROOT/'data/interim/manifests/seizure_manifest.csv',dtype={'subject':str})
sz['recording_id']=('sub-'+sz.subject+'_ses-'+sz.session.astype(str).str.zfill(2)+'_task-'+sz.task+'_run-'+sz.run.astype(str).str.zfill(2))
E=sz[sz.eligible_for_prediction & sz.vigilance.isin(['asleep','awake'])]

print('=== C3 headline: recording-level background, their unit vs valid unit ===')
R=pd.read_csv(V/'per_recording_by_config.csv')
lab=E.groupby('recording_id').vigilance.agg(lambda x:'asleep' if (x=='asleep').any() else 'awake').rename('recvig')
R2=R.merge(lab,on='recording_id',how='inner')
p=R2.pivot_table(index='tag',columns='recvig',values='neg_mean',aggfunc='mean')
t=stats.ttest_rel(p['asleep'],p['awake'])
print('their config-paired: %.4f vs %.4f diff=%+.4f t=%.2f p=%.3g (%d/42+)'%(
    p['asleep'].mean(),p['awake'].mean(),(p['asleep']-p['awake']).mean(),t.statistic,t.pvalue,(p['asleep']>p['awake']).sum()))
rec=R2.groupby(['recording_id','recvig']).neg_mean.mean().reset_index()
rec['subject']=rec.recording_id.str.slice(4,7)
a=rec[rec.recvig=='asleep'].neg_mean; b=rec[rec.recvig=='awake'].neg_mean
print('RECORDING as unit: n_asleep_rec=%d n_awake_rec=%d diff=%+.4f Welch p=%.4f MWU p=%.4f'%(
    len(a),len(b),a.mean()-b.mean(),stats.ttest_ind(a,b,equal_var=False).pvalue,stats.mannwhitneyu(a,b).pvalue))
rng=np.random.default_rng(3); by={s:g for s,g in rec.groupby('subject')}; subs=rec.subject.unique(); bo=[]
for i in range(3000):
    g=pd.concat([by[s] for s in rng.choice(subs,len(subs),True)])
    x=g[g.recvig=='asleep'].neg_mean; y=g[g.recvig=='awake'].neg_mean
    if len(x)<2 or len(y)<2: continue
    bo.append(x.mean()-y.mean())
bo=np.array(bo); print('PATIENT cluster bootstrap: p=%.4f 95%%CI [%+.4f,%+.4f]'%(2*min((bo<=0).mean(),(bo>=0).mean()),np.percentile(bo,2.5),np.percentile(bo,97.5)))

print('\n=== C3 mediation replication ===')
import numpy.linalg as la
b1=[];b2=[]
for tag,g in S.groupby('tag'):
    g=g.dropna(subset=['rec_neg_mean'])
    x=(g.vigilance=='asleep').astype(float).values; y=g.caught.values
    X=np.c_[np.ones(len(x)),x]; b1.append(la.lstsq(X,y,rcond=None)[0][1])
    z=(g.rec_neg_mean.values-g.rec_neg_mean.mean())/g.rec_neg_mean.std()
    b2.append(la.lstsq(np.c_[np.ones(len(x)),x,z],y,rcond=None)[0][1])
b1=np.array(b1);b2=np.array(b2)
print('beta unadj=%+.4f (config-paired p=%.4g) ; adj=%+.4f (p=%.4g) ; removed=%.0f%%'%(
    b1.mean(),stats.ttest_1samp(b1,0).pvalue,b2.mean(),stats.ttest_1samp(b2,0).pvalue,100*(1-b2.mean()/b1.mean())))

print('\n=== C5 MONTAGE ===')
dec=pd.read_csv(ROOT/'data/interim/manifests/decision_manifest.csv',usecols=['recording_id','subject','bte_side'],dtype={'subject':str})
print('bte_side values overall:',dec.bte_side.value_counts().to_dict())
ps=dec.groupby('subject').bte_side.nunique()
print('patients total=%d ; with exactly 1 bte_side=%d ; with >1=%d'%(len(ps),(ps==1).sum(),(ps>1).sum()))
recside=dec.groupby('recording_id').bte_side.first()
key=S.drop_duplicates('seizure_id')[['seizure_id','subject','vigilance','recording_id']].copy()
key['bte']=key.recording_id.map(recside)
ct=pd.crosstab(key.vigilance,key.bte)
print('\nCOUNTS of the 275 analysed seizures:'); print(ct)
print('row proportions:'); print(pd.crosstab(key.vigilance,key.bte,normalize='index').round(3))
c2=stats.chi2_contingency(ct); print('chi2 p=%.4g  expected min cell=%.2f'%(c2.pvalue,c2.expected_freq.min()))
print('Fisher-like (MC) via permutation:')
lv=key.vigilance.values; bt=key.bte.values; rng2=np.random.default_rng(5)
obs=stats.chi2_contingency(pd.crosstab(pd.Series(lv),pd.Series(bt))).statistic
nl=[stats.chi2_contingency(pd.crosstab(pd.Series(rng2.permutation(lv)),pd.Series(bt))).statistic for _ in range(2000)]
print('  perm p=%.4f'%(np.mean(np.array(nl)>=obs)))
# patient-level permutation (bte is patient-level -> permute vigilance within patient)
pat=key.groupby('subject')
nl2=[]
for i in range(2000):
    k=key.copy(); k['v']=k.groupby('subject').vigilance.transform(lambda s: rng2.permutation(s.values))
    nl2.append(stats.chi2_contingency(pd.crosstab(k.v,k.bte)).statistic)
print('  perm p WITHIN patient (bte is patient-level) = %.4f'%(np.mean(np.array(nl2)>=obs)))
