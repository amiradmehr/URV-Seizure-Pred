import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
ROOT=Path('/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'); OUT=ROOT/'outputs/analysis/vigilance'
S=pd.read_csv(OUT/'per_seizure_full.csv',dtype={'subject':str})
A=S[S.vigilance.isin(['asleep','awake'])].copy()
dec=pd.read_csv(ROOT/'data/interim/manifests/decision_manifest.csv',usecols=['recording_id','bte_side','subject','decision_time_seconds'],dtype={'subject':str})
side=dec.groupby('recording_id').bte_side.first(); A['bte_side']=A.recording_id.map(side)
rng=np.random.default_rng(0)

# --- fast pivot: matrix of caught (seizures x 42 configs) ---
M=A.pivot_table(index='seizure_id',columns='tag',values='caught')
key=A.drop_duplicates('seizure_id').set_index('seizure_id').loc[M.index,['subject','vigilance','bte_side','recording_id']]
Mv=M.to_numpy(); isA=(key.vigilance=='asleep').to_numpy()
def tstat(mask):
    a=Mv[mask].mean(0); b=Mv[~mask].mean(0)
    return stats.ttest_rel(a,b).statistic
obs=tstat(isA); print('observed config-paired t = %.3f'%obs)
subs=key.subject.to_numpy()
order=np.argsort(subs,kind='stable'); groups=np.split(np.arange(len(subs))[order], np.cumsum(pd.Series(subs[order]).value_counts(sort=False).values)[:-1])
NP=3000
free=np.empty(NP); within=np.empty(NP)
for i in range(NP):
    free[i]=tstat(rng.permutation(isA))
    m=isA.copy()
    for g in groups: m[g]=rng.permutation(isA[g])
    within[i]=tstat(m)
print('FREE perm null t = %.2f +- %.2f ; p(|t|>=obs) = %.4f'%(free.mean(),free.std(),(np.abs(free)>=abs(obs)).mean()))
print('WITHIN-patient perm null t = %.2f +- %.2f ; p(|t|>=obs) = %.4f ; p(t>=obs one-sided)=%.4f'%(
    within.mean(),within.std(),(np.abs(within)>=abs(obs)).mean(),(within>=obs).mean()))

print('\n--- montage -> sensitivity, with patient clustering ---')
per=A.groupby(['seizure_id','subject','vigilance','bte_side']).caught.mean().reset_index()
for x,y in [('right','left'),('right','bilateral'),('left','bilateral')]:
    a=per[per.bte_side==x].caught; b=per[per.bte_side==y].caught
    ssub=per[per.bte_side.isin([x,y])]
    ss=ssub.subject.unique(); ix={s:ssub.index[ssub.subject==s].to_numpy() for s in ss}
    bo=[]
    for _ in range(2000):
        pk=rng.choice(ss,size=len(ss),replace=True); g=ssub.loc[np.concatenate([ix[s] for s in pk])]
        aa=g[g.bte_side==x].caught; bb=g[g.bte_side==y].caught
        if len(aa) and len(bb): bo.append(aa.mean()-bb.mean())
    bo=np.array(bo)
    print('  %-10s vs %-10s diff=%+.4f (n=%d,%d) Welch p=%.4f | patient-boot p=%.4f CI[%+.4f,%+.4f]'%(
        x,y,a.mean()-b.mean(),len(a),len(b),stats.ttest_ind(a,b,equal_var=False).pvalue,
        2*min((bo<=0).mean(),(bo>=0).mean()),*np.percentile(bo,[2.5,97.5])))

print('\n===== CLAIM 7: recording duration =====')
sz=pd.read_csv(ROOT/'data/interim/manifests/seizure_manifest.csv',dtype={'subject':str})
sz['recording_id']=('sub-'+sz.subject+'_ses-'+sz.session.astype(str).str.zfill(2)+'_task-'+sz.task+'_run-'+sz.run.astype(str).str.zfill(2))
rec=dec.groupby('recording_id').decision_time_seconds.agg(lambda x:(x.max()-x.min())/3600+0.75)
E=sz[sz.eligible_for_prediction & sz.vigilance.isin(['asleep','awake'])].copy()
E['dur_h']=E.recording_id.map(rec)
a=E[E.vigilance=='asleep'].dur_h.dropna(); b=E[E.vigilance=='awake'].dur_h.dropna()
print('dur_h asleep mean %.2f med %.2f (n=%d) ; awake mean %.2f med %.2f (n=%d) ; MWU p=%.4f'%(
    a.mean(),a.median(),len(a),b.mean(),b.median(),len(b),stats.mannwhitneyu(a,b).pvalue))
Ed=E.dropna(subset=['dur_h']); ss=Ed.subject.unique(); ix={s:Ed.index[Ed.subject==s].to_numpy() for s in ss}
bo=[]
for _ in range(2000):
    pk=rng.choice(ss,size=len(ss),replace=True); g=Ed.loc[np.concatenate([ix[s] for s in pk])]
    aa=g[g.vigilance=='asleep'].dur_h; bb=g[g.vigilance=='awake'].dur_h
    if len(aa) and len(bb): bo.append(aa.mean()-bb.mean())
bo=np.array(bo); print('  patient-boot diff CI [%+.2f,%+.2f] p=%.4f'%(*np.percentile(bo,[2.5,97.5]),2*min((bo<=0).mean(),(bo>=0).mean())))
a2=E[E.vigilance=='asleep'].duration_seconds; b2=E[E.vigilance=='awake'].duration_seconds
print('seizure duration asleep %.1f awake %.1f MWU p=%.4f'%(a2.mean(),b2.mean(),stats.mannwhitneyu(a2,b2).pvalue))

print('\n===== CLAIM 8: eligibility x vigilance =====')
ct=pd.crosstab(sz.vigilance,sz.eligible_for_prediction); print(ct)
print('rates:',(ct[True]/(ct[True]+ct[False])).round(4).to_dict())
print('eligibility_evaluated values:',sz.eligibility_evaluated.value_counts().to_dict())
print('chi2 asleep vs awake p=%.4g'%stats.chi2_contingency(ct.loc[['asleep','awake']]).pvalue)
# patient-clustered
sub2=sz[sz.vigilance.isin(['asleep','awake'])].copy(); sub2['e']=sub2.eligible_for_prediction.astype(float)
ss=sub2.subject.unique(); ix={s:sub2.index[sub2.subject==s].to_numpy() for s in ss}
bo=[]
for _ in range(2000):
    pk=rng.choice(ss,size=len(ss),replace=True); g=sub2.loc[np.concatenate([ix[s] for s in pk])]
    aa=g[g.vigilance=='asleep'].e; bb=g[g.vigilance=='awake'].e
    if len(aa) and len(bb): bo.append(aa.mean()-bb.mean())
bo=np.array(bo); print('elig-rate diff asleep-awake = %+.4f patient-boot CI [%+.4f,%+.4f] p=%.4f'%(
    sub2[sub2.vigilance=='asleep'].e.mean()-sub2[sub2.vigilance=='awake'].e.mean(),*np.percentile(bo,[2.5,97.5]),2*min((bo<=0).mean(),(bo>=0).mean())))
print('asleep analysed 103 of %d eligible-asleep; of %d annotated asleep'%(int(ct.loc['asleep',True]),int(ct.loc['asleep'].sum())))
