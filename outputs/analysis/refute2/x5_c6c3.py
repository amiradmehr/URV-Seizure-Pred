import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
ROOT=Path('/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'); OUT=ROOT/'outputs/analysis/vigilance'
TAGS=['spectral-gru__large__h15__sop30','spectral-meanpool__large__h5__sop30',
      'spectral-attention__large__h5__sop30','logistic-mean__small__h5__sop30',
      'spectral-meanpool__small__h45__sop10','spectral-attention__large__h45__sop10']
sub=pd.read_csv(OUT/'negatives_with_proxy.csv')
print('=== CLAIM 6: within-recording rho distribution (not just the median) ===')
for tag in TAGS:
    dd=pd.read_csv(ROOT/f'outputs/cv/{tag}/out_of_fold_predictions.csv',
                   usecols=['recording_id','decision_time_seconds','label','probability'])
    m=sub.merge(dd[dd.label==0][['recording_id','decision_time_seconds','probability']],
                on=['recording_id','decision_time_seconds'],how='inner',suffixes=('','_t')).dropna(subset=['probability_t'])
    rho=stats.spearmanr(m.proxy,m.probability_t).statistic
    rr=np.array([stats.spearmanr(g.proxy,g.probability_t).statistic for _,g in m.groupby('recording_id') if len(g)>=50])
    rr=rr[np.isfinite(rr)]
    npos=(rr>0).sum(); sg=stats.binomtest(npos,len(rr),0.5).pvalue
    w=stats.wilcoxon(rr).pvalue
    print('%-38s pooled=%+.3f | within-rec median=%+.3f mean=%+.3f  %d/%d>0 sign p=%.3g wilcoxon p=%.3g  IQR[%+.3f,%+.3f]'%(
        tag,rho,np.median(rr),rr.mean(),npos,len(rr),sg,w,*np.percentile(rr,[25,75])))

print('\n=== CLAIM 3: recording-level background, analyst config-paired vs proper unit ===')
S=pd.read_csv(OUT/'per_seizure_full.csv',dtype={'subject':str})
sz=pd.read_csv(ROOT/'data/interim/manifests/seizure_manifest.csv',dtype={'subject':str})
sz['recording_id']=('sub-'+sz.subject+'_ses-'+sz.session.astype(str).str.zfill(2)+'_task-'+sz.task+'_run-'+sz.run.astype(str).str.zfill(2))
E=sz[sz.eligible_for_prediction & sz.vigilance.isin(['asleep','awake'])]
lab=E.groupby('recording_id').vigilance.agg(lambda x:'asleep' if (x=='asleep').any() else 'awake').rename('recvig')
R=pd.read_csv(OUT/'per_recording_by_config.csv').merge(lab,on='recording_id',how='inner')
p=R.pivot_table(index='tag',columns='recvig',values='neg_mean',aggfunc='mean')
t=stats.ttest_rel(p['asleep'],p['awake'])
print("analyst (config-paired over 42 tags): %.4f vs %.4f diff=%+.4f t=%+.2f p=%.4g (%d/42 +)"%(
    p['asleep'].mean(),p['awake'].mean(),(p['asleep']-p['awake']).mean(),t.statistic,t.pvalue,(p['asleep']>p['awake']).sum()))
per_rec=R.groupby(['recording_id','recvig']).neg_mean.mean().reset_index()
per_rec['subject']=per_rec.recording_id.str.slice(4,7)
a=per_rec[per_rec.recvig=='asleep'].neg_mean; b=per_rec[per_rec.recvig=='awake'].neg_mean
rng=np.random.default_rng(0); ss=per_rec.subject.unique(); ix={s:per_rec.index[per_rec.subject==s].to_numpy() for s in ss}
bo=[]
for _ in range(3000):
    pk=rng.choice(ss,size=len(ss),replace=True); g=per_rec.loc[np.concatenate([ix[s] for s in pk])]
    aa=g[g.recvig=='asleep'].neg_mean; bb=g[g.recvig=='awake'].neg_mean
    if len(aa) and len(bb): bo.append(aa.mean()-bb.mean())
bo=np.array(bo)
print("RECORDING as unit: %.4f (n=%d) vs %.4f (n=%d) diff=%+.4f Welch p=%.4f | patient-boot CI [%+.4f,%+.4f] p=%.4f"%(
    a.mean(),len(a),b.mean(),len(b),a.mean()-b.mean(),stats.ttest_ind(a,b,equal_var=False).pvalue,
    *np.percentile(bo,[2.5,97.5]),2*min((bo<=0).mean(),(bo>=0).mean())))

print('\n=== CLAIM 3: is the recalibrated gap SIGNIFICANTLY different from the crude gap? (paired on same seizures) ===')
A=S[S.vigilance.isin(['asleep','awake'])]
for col in ['caught_pat','caught_rec','caught_loc']:
    d=A.dropna(subset=['caught',col]).groupby(['seizure_id','subject','vigilance'])[['caught',col]].mean().reset_index()
    d['delta']=d['caught']-d[col]
    aa=d[d.vigilance=='asleep'].delta; bb=d[d.vigilance=='awake'].delta
    ss=d.subject.unique(); ix={s:d.index[d.subject==s].to_numpy() for s in ss}; bo=[]
    for _ in range(2000):
        pk=rng.choice(ss,size=len(ss),replace=True); g=d.loc[np.concatenate([ix[s] for s in pk])]
        x=g[g.vigilance=='asleep'].delta; y=g[g.vigilance=='awake'].delta
        if len(x) and len(y): bo.append(x.mean()-y.mean())
    bo=np.array(bo)
    print('  gap(global)-gap(%s) = %+.4f  patient-boot CI [%+.4f,%+.4f] p=%.4f'%(
        col,aa.mean()-bb.mean(),*np.percentile(bo,[2.5,97.5]),2*min((bo<=0).mean(),(bo>=0).mean())))
