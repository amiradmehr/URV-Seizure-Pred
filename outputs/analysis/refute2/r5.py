import numpy as np, pandas as pd
from scipy import stats
from pathlib import Path
ROOT=Path('/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'); V=ROOT/'outputs/analysis/vigilance'
S=pd.read_csv(V/'per_seizure_full.csv',dtype={'subject':str})
S=S[S.vigilance.isin(['asleep','awake'])].copy()
dec=pd.read_csv(ROOT/'data/interim/manifests/decision_manifest.csv',usecols=['recording_id','bte_side'])
recside=dec.groupby('recording_id').bte_side.first()
S['bte']=S.recording_id.map(recside)
sz=S.groupby(['seizure_id','subject','vigilance','bte']).caught.mean().reset_index()

print('=== C5: sens @1FA/h by montage (seizure-level mean over 42 configs) ===')
print(sz.groupby('bte').caught.agg(['count','mean']).round(4))
print('NB bilateral = BOTH BTE channels present (more real signal) yet LOWEST sensitivity')
print('\ncell counts vigilance x montage:'); print(pd.crosstab(sz.vigilance,sz.bte))

print('\n=== C5: direct standardisation, replicate + patient-cluster bootstrap ===')
def std_gap(df):
    w=df.bte.value_counts(normalize=True)
    m=df.groupby(['vigilance','bte']).caught.mean().unstack()
    for s in w.index:
        if s not in m.columns: m[s]=np.nan
    m=m[w.index]
    def wm(row):
        v=row.values; ok=np.isfinite(v)
        return np.nansum(v[ok]*w.values[ok])/w.values[ok].sum()
    if 'asleep' not in m.index or 'awake' not in m.index: return np.nan,np.nan
    return wm(m.loc['asleep'])-wm(m.loc['awake']), df[df.vigilance=='asleep'].caught.mean()-df[df.vigilance=='awake'].caught.mean()
sg,cg=std_gap(sz)
print('crude=%+.4f  montage-standardised=%+.4f  -> %.0f%% "explained"'%(cg,sg,100*(1-sg/cg)))
rng=np.random.default_rng(21); by={s:g for s,g in sz.groupby('subject')}; subs=sz.subject.unique()
bs=[];bc=[]
for i in range(3000):
    g=pd.concat([by[s] for s in rng.choice(subs,len(subs),True)])
    a,c=std_gap(g)
    if np.isfinite(a): bs.append(a); bc.append(c)
bs=np.array(bs);bc=np.array(bc)
print('patient-cluster bootstrap of STANDARDISED gap: 95%%CI [%+.4f,%+.4f]  p=%.3f'%(np.percentile(bs,2.5),np.percentile(bs,97.5),2*min((bs<=0).mean(),(bs>=0).mean())))
print('bootstrap of %% explained: median=%.0f%%  95%%CI [%.0f%%, %.0f%%]'%(
    np.median(100*(1-bs/bc)),np.percentile(100*(1-bs/bc),2.5),np.percentile(100*(1-bs/bc),97.5)))
print('  -> fraction of bootstrap reps where "explained" is outside 0-100%%: %.2f'%(np.mean((100*(1-bs/bc)<0)|(100*(1-bs/bc)>100))))

print('\n=== C5: drop the n=2 bilateral-asleep cell (restrict to left+right only) ===')
sz2=sz[sz.bte.isin(['left','right'])]
s2,c2=std_gap(sz2); print('left/right only: crude=%+.4f standardised=%+.4f -> %.0f%% explained'%(c2,s2,100*(1-s2/c2)))

print('\n=== C5: is montage separable from patient? ===')
pat_bte=sz.groupby('subject').bte.nunique()
print('patients among the 275-seizure set: %d ; with >1 montage: %d'%(len(pat_bte),(pat_bte>1).sum()))

print('\n=== C7: n_pos and recording duration ===')
n=S.groupby(['tag','vigilance']).n_pos.mean().unstack()
print('n_pos asleep=%.3f awake=%.3f diff=%+.4f (%.2f%% relative)'%(n['asleep'].mean(),n['awake'].mean(),(n['asleep']-n['awake']).mean(),100*(n['asleep']-n['awake']).mean()/n['awake'].mean()))
szm=pd.read_csv(ROOT/'data/interim/manifests/seizure_manifest.csv',dtype={'subject':str})
szm['recording_id']=('sub-'+szm.subject+'_ses-'+szm.session.astype(str).str.zfill(2)+'_task-'+szm.task+'_run-'+szm.run.astype(str).str.zfill(2))
E=szm[szm.eligible_for_prediction & szm.vigilance.isin(['asleep','awake'])]
d2=pd.read_csv(ROOT/'data/interim/manifests/decision_manifest.csv',usecols=['recording_id','decision_time_seconds'])
rd=d2.groupby('recording_id').decision_time_seconds.agg(lambda x:(x.max()-x.min())/3600+0.75).rename('dur_h')
E=E.merge(rd,on='recording_id',how='left')
print(E.groupby('vigilance').dur_h.agg(['count','mean','median']).round(2))
print('MWU p=%.4g'%stats.mannwhitneyu(E[E.vigilance=='asleep'].dur_h.dropna(),E[E.vigilance=='awake'].dur_h.dropna()).pvalue)
print(E.groupby('vigilance').duration_seconds.agg(['mean','median']).round(2))
print('sz duration MWU p=%.4g'%stats.mannwhitneyu(E[E.vigilance=='asleep'].duration_seconds,E[E.vigilance=='awake'].duration_seconds).pvalue)
# does n_pos difference survive at seizure level with clustering?
nn=S.groupby(['seizure_id','subject','vigilance']).n_pos.mean().reset_index()
print('n_pos distribution asleep:',nn[nn.vigilance=='asleep'].n_pos.describe()[['min','25%','50%','75%','max']].round(2).to_dict())
print('n_pos distribution awake :',nn[nn.vigilance=='awake'].n_pos.describe()[['min','25%','50%','75%','max']].round(2).to_dict())
print('seizures with truncated n_pos (<20 avg): asleep %d/%d, awake %d/%d'%(
    (nn[nn.vigilance=="asleep"].n_pos<20).sum(),(nn.vigilance=="asleep").sum(),
    (nn[nn.vigilance=="awake"].n_pos<20).sum(),(nn.vigilance=="awake").sum()))
