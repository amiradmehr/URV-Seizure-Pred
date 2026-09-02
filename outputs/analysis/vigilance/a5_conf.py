import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
ROOT=Path('/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'); OUT=ROOT/'outputs/analysis/vigilance'
S=pd.read_csv(OUT/'per_seizure_full.csv',dtype={'subject':str})
sz=pd.read_csv(ROOT/'data/interim/manifests/seizure_manifest.csv',dtype={'subject':str})
sz['recording_id']=('sub-'+sz.subject+'_ses-'+sz.session.astype(str).str.zfill(2)+
                    '_task-'+sz.task+'_run-'+sz.run.astype(str).str.zfill(2))
dec=pd.read_csv(ROOT/'data/interim/manifests/decision_manifest.csv',
                usecols=['recording_id','bte_side','decision_time_seconds','subject'],dtype={'subject':str})
recinfo=dec.groupby('recording_id').agg(bte_side=('bte_side','first'),
                                        dur_h=('decision_time_seconds',lambda x:(x.max()-x.min())/3600+0.75),
                                        n_dec=('decision_time_seconds','size')).reset_index()
print('=== ELIGIBILITY by vigilance (annotated -> usable) ===')
t=pd.crosstab(sz.vigilance,sz.eligible_for_prediction)
t['rate']=t[True]/(t[True]+t[False]); print(t)

E=sz[sz.eligible_for_prediction & sz.vigilance.isin(['asleep','awake'])].merge(recinfo,on='recording_id',how='left')
print('\n=== RECORDING-LEVEL CONFOUNDERS of eligible seizures ===')
print(E.groupby('vigilance')[['dur_h','n_dec','duration_seconds']].agg(['mean','median','count']))
print('bte_side x vigilance:'); print(pd.crosstab(E.vigilance,E.bte_side,normalize='index').round(3))
for c in ['dur_h','duration_seconds']:
    a=E[E.vigilance=='asleep'][c].dropna(); b=E[E.vigilance=='awake'][c].dropna()
    print(f'  {c}: Mann-Whitney p={stats.mannwhitneyu(a,b).pvalue:.4g}')
ct=pd.crosstab(E.vigilance,E.bte_side)
print('  bte_side chi2 p=%.4g'%stats.chi2_contingency(ct).pvalue)

print('\n=== AMPLITUDE regime (mean log total power, 10 min pre-onset) ===')
N=pd.read_csv(OUT/'negatives_with_proxy.csv')
V=None
# reuse proxy computation for seizures via a4 output? recompute quickly from cache-free path
import json
FD=ROOT/'data/interim/chunk_features'
rows=[]
for _,s in E.iterrows():
    f=FD/f'{s.recording_id}_features.npy'; a=FD/f'{s.recording_id}_availability.npy'
    if not f.exists(): continue
    F=np.load(f,mmap_mode='r'); AV=np.load(a); ch=[c for c in range(3) if AV[9*c]]
    if not ch: continue
    c1=int(s.onset_seconds/5.0); c0=max(0,c1-120); c1=min(c1,F.shape[0])
    if c1<=c0: continue
    amp=float(np.asarray(F[c0:c1][:,[5+9*c for c in ch]]).mean())
    rows.append(dict(seizure_id=s.seizure_id,vigilance=s.vigilance,subject=s.subject,amp=amp,nch=len(ch)))
A=pd.DataFrame(rows)
print(A.groupby('vigilance').amp.agg(['count','mean','std']))
print('Mann-Whitney p=%.4g'%stats.mannwhitneyu(A[A.vigilance=='asleep'].amp,A[A.vigilance=='awake'].amp).pvalue)
wp=[g.groupby('vigilance').amp.mean() for _,g in A.groupby('subject') if g.vigilance.nunique()==2]
dd=[x['asleep']-x['awake'] for x in wp]
print('within-patient asleep-awake amp = %+.3f (n=%d) p=%.4g'%(np.mean(dd),len(dd),stats.ttest_1samp(dd,0).pvalue))

print('\n=== BACKGROUND base rate: between vs within patient ===')
R=pd.read_csv(OUT/'per_recording_by_config.csv')
lab=E.groupby('recording_id').vigilance.agg(lambda x:'asleep' if (x=='asleep').any() else 'awake').rename('recvig')
R=R.merge(lab,on='recording_id',how='inner')
p=R.pivot_table(index='tag',columns='recvig',values='neg_mean',aggfunc='mean')
t=stats.ttest_rel(p['asleep'],p['awake'])
print(f'mean NEGATIVE prob, recordings with an asleep sz vs awake-only: {p["asleep"].mean():.4f} vs {p["awake"].mean():.4f} '
      f'diff={(p["asleep"]-p["awake"]).mean():+.4f} t={t.statistic:+.2f} p={t.pvalue:.4g} ({(p["asleep"]>p["awake"]).sum()}/42 +)')

print('\n=== MEDIATION: caught ~ asleep, with and without recording base rate ===')
import numpy.linalg as la
b1=[];b2=[]
for tag,g in S[S.vigilance.isin(['asleep','awake'])].groupby('tag'):
    g=g.dropna(subset=['rec_neg_mean'])
    x=(g.vigilance=='asleep').astype(float).values; y=g.caught.values
    X=np.c_[np.ones(len(x)),x]; b1.append(la.lstsq(X,y,rcond=None)[0][1])
    z=(g.rec_neg_mean.values-g.rec_neg_mean.mean())/g.rec_neg_mean.std()
    X2=np.c_[np.ones(len(x)),x,z]; b2.append(la.lstsq(X2,y,rcond=None)[0][1])
b1=np.array(b1);b2=np.array(b2)
print('beta(asleep) unadjusted = %+.4f (p=%.4g)'%(b1.mean(),stats.ttest_1samp(b1,0).pvalue))
print('beta(asleep) adjusted for recording base rate = %+.4f (p=%.4g)'%(b2.mean(),stats.ttest_1samp(b2,0).pvalue))
print('proportion of effect removed = %.0f%%'%(100*(1-b2.mean()/b1.mean())))
