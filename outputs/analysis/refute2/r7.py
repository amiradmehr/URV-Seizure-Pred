import numpy as np, pandas as pd
from scipy import stats
from pathlib import Path
ROOT=Path('/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'); V=ROOT/'outputs/analysis/vigilance'
N=pd.read_csv(V/'negatives_with_proxy.csv')
TAGS=['spectral-gru__large__h15__sop30','spectral-meanpool__large__h5__sop30',
      'spectral-attention__large__h5__sop30','logistic-mean__small__h5__sop30',
      'spectral-meanpool__small__h45__sop10','spectral-attention__large__h45__sop10']
print('=== C6 per-config within-recording rho (replicating a4) ===')
for tag in TAGS:
    dd=pd.read_csv(ROOT/f'outputs/cv/{tag}/out_of_fold_predictions.csv',
                   usecols=['recording_id','decision_time_seconds','label','probability'])
    m=N.merge(dd[dd.label==0][['recording_id','decision_time_seconds','probability']],
              on=['recording_id','decision_time_seconds'],how='left',suffixes=('','_t')).dropna(subset=['probability_t'])
    rr=[stats.spearmanr(g.proxy,g.probability_t).statistic for _,g in m.groupby('recording_id') if len(g)>=50]
    rr=np.array([x for x in rr if np.isfinite(x)])
    pooled=stats.spearmanr(m.proxy,m.probability_t).statistic
    # recording as unit: mean proxy vs mean prob
    rm=m.groupby('recording_id').agg(proxy=('proxy','mean'),prob=('probability_t','mean'),n=('proxy','size'))
    rb=stats.spearmanr(rm.proxy,rm.prob)
    print('%-40s pooled=%+.3f | within-rec median=%+.3f mean=%+.3f frac>0=%.2f wilcox p=%.3g | BETWEEN-rec rho=%+.3f p=%.2g (n=%d)'%(
        tag,pooled,np.median(rr),rr.mean(),(rr>0).mean(),stats.wilcoxon(rr).pvalue,rb.statistic,rb.pvalue,len(rm)))

print('\n=== C7: does recording DURATION explain the gap? ===')
S=pd.read_csv(V/'per_seizure_full.csv',dtype={'subject':str})
S=S[S.vigilance.isin(['asleep','awake'])]
d2=pd.read_csv(ROOT/'data/interim/manifests/decision_manifest.csv',usecols=['recording_id','decision_time_seconds'])
rd=d2.groupby('recording_id').decision_time_seconds.agg(lambda x:(x.max()-x.min())/3600+0.75).rename('dur_h')
sz=S.groupby(['seizure_id','subject','vigilance','recording_id']).caught.mean().reset_index().merge(rd,on='recording_id',how='left')
print(sz.groupby('vigilance').dur_h.agg(['count','mean','median']).round(2))
print('MWU p (analysed 275) = %.4g'%stats.mannwhitneyu(sz[sz.vigilance=='asleep'].dur_h.dropna(),sz[sz.vigilance=='awake'].dur_h.dropna()).pvalue)
print('corr(dur_h, caught) spearman = %.4f p=%.3g'%tuple(stats.spearmanr(sz.dur_h,sz.caught))[:2] if False else '')
r=stats.spearmanr(sz.dur_h.fillna(sz.dur_h.median()),sz.caught); print('spearman(dur_h, caught) = %+.4f p=%.3g'%(r.statistic,r.pvalue))
# standardise on duration tertiles
sz['dq']=pd.qcut(sz.dur_h,3,labels=['short','mid','long'])
print(sz.groupby('dq',observed=True).caught.agg(['count','mean']).round(4))
print(pd.crosstab(sz.vigilance,sz.dq,normalize='index').round(3))
w=sz.dq.value_counts(normalize=True); m=sz.groupby(['vigilance','dq'],observed=True).caught.mean().unstack()[w.index]
a=(m.loc['asleep']*w).sum(); b=(m.loc['awake']*w).sum()
crude=sz[sz.vigilance=='asleep'].caught.mean()-sz[sz.vigilance=='awake'].caught.mean()
print('crude=%+.4f  duration-standardised=%+.4f  -> %.0f%% "explained" by duration alone'%(crude,a-b,100*(1-(a-b)/crude)))

print('\n=== C8: alternative mechanism for differential eligibility - seizure clustering ===')
szm=pd.read_csv(ROOT/'data/interim/manifests/seizure_manifest.csv',dtype={'subject':str})
szm['recording_id']=('sub-'+szm.subject+'_ses-'+szm.session.astype(str).str.zfill(2)+'_task-'+szm.task+'_run-'+szm.run.astype(str).str.zfill(2))
szm=szm.sort_values(['recording_id','onset_seconds'])
szm['gap_prev_min']=szm.groupby('recording_id').onset_seconds.diff()/60.0
print(szm.groupby('vigilance').gap_prev_min.agg(['count','median']).round(1))
szm['is_first']=szm.gap_prev_min.isna()
szm['prev_within60']=(szm.gap_prev_min<=60)
print('fraction with a preceding seizure <=60 min earlier in same recording:')
print(szm.groupby('vigilance').prev_within60.mean().round(3))
print('\neligibility rate SPLIT by whether a prior seizure was within 60 min:')
print(pd.crosstab([szm.vigilance,szm.prev_within60],szm.eligible_for_prediction,normalize='index').round(3))
print('\nalso onset < 60 min into recording (cannot have 60 min history):')
szm['early']=szm.onset_seconds<3600
print(szm.groupby('vigilance').early.mean().round(3))
print('\neligibility among seizures that are NOT early and have NO prior sz within 60 min:')
cl=szm[(~szm.early)&(~szm.prev_within60)]
print(pd.crosstab(cl.vigilance,cl.eligible_for_prediction,normalize='index').round(3))
print(pd.crosstab(cl.vigilance,cl.eligible_for_prediction))
print('\n103 analysed vs 112 eligible asleep -> extra attrition beyond eligibility:')
S1=S.drop_duplicates('seizure_id')
print('analysed:',S1.vigilance.value_counts().to_dict(),'| eligible in manifest: asleep 112 awake 193')
