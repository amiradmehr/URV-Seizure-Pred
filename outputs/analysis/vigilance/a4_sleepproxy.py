import numpy as np, pandas as pd, json
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score
ROOT=Path('/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'); OUT=ROOT/'outputs/analysis/vigilance'
FD=ROOT/'data/interim/chunk_features'
TAGS=['spectral-gru__large__h15__sop30','spectral-meanpool__large__h5__sop30',
      'spectral-attention__large__h5__sop30','logistic-mean__small__h5__sop30',
      'spectral-meanpool__small__h45__sop10','spectral-attention__large__h45__sop10']
d=pd.read_csv(ROOT/f'outputs/cv/{TAGS[0]}/out_of_fold_predictions.csv',
              usecols=['recording_id','subject','decision_time_seconds','label','target_seizure_id','probability'])
d['subject']=d['subject'].astype(str).str.zfill(3)
recs=d.recording_id.unique()
rng=np.random.default_rng(0)
sz=pd.read_csv(ROOT/'data/interim/manifests/seizure_manifest.csv',dtype={'subject':str})
sz['recording_id']=('sub-'+sz.subject+'_ses-'+sz.session.astype(str).str.zfill(2)+
                    '_task-'+sz.task+'_run-'+sz.run.astype(str).str.zfill(2))
elig=sz[sz.eligible_for_prediction & sz.vigilance.isin(['asleep','awake'])]
sz_recs=set(elig.recording_id)&set(recs)
other=[r for r in recs if r not in sz_recs]
sample=sorted(sz_recs)+list(rng.choice(other,size=min(250,len(other)),replace=False))
print('recordings sampled:',len(sample),'(seizure-bearing:',len(sz_recs),')')

DELTA=[0,9,18]; BETA=[3,12,21]; TOT=[5,14,23]
cache={}
for r in sample:
    f=FD/f'{r}_features.npy'; a=FD/f'{r}_availability.npy'
    if not f.exists(): continue
    F=np.load(f,mmap_mode='r'); AV=np.load(a)
    ch=[c for c in range(3) if AV[9*c]]
    if not ch: continue
    dl=np.asarray(F[:,[DELTA[c] for c in ch]]).mean(1)
    bt=np.asarray(F[:,[BETA[c] for c in ch]]).mean(1)
    tp=np.asarray(F[:,[TOT[c] for c in ch]]).mean(1)
    cache[r]=(dl-bt, tp, len(ch))
print('feature files loaded:',len(cache))

def window_stats(r,t,minutes=10):
    if r not in cache: return np.nan,np.nan
    dv,tp,_=cache[r]; c1=int(t/5.0); c0=max(0,c1-int(minutes*60/5))
    if c1<=c0 or c0>=len(dv): return np.nan,np.nan
    c1=min(c1,len(dv))
    return float(np.nanmean(dv[c0:c1])), float(np.nanmean(tp[c0:c1]))

# --- A. validate the sleep proxy on labelled seizure onsets ---
rows=[]
for _,s in elig.iterrows():
    p,a=window_stats(s.recording_id,s.onset_seconds)
    rows.append(dict(seizure_id=s.seizure_id,vigilance=s.vigilance,proxy=p,amp=a,rec=s.recording_id,subject=s.subject))
V=pd.DataFrame(rows).dropna(subset=['proxy'])
print('\n=== A. sleep proxy (log delta - log beta, 10 min pre-onset) validated on vigilance labels ===')
print(V.groupby('vigilance').proxy.agg(['count','mean','std']))
y=(V.vigilance=='asleep').astype(int)
print('AUC(proxy -> asleep) = %.3f'%roc_auc_score(y,V.proxy))
# within-patient
wp=[]
for s,g in V.groupby('subject'):
    if g.vigilance.nunique()==2: wp.append(g.groupby('vigilance').proxy.mean().diff().iloc[-1])
print('within-patient mean(asleep-awake) proxy = %+.3f over %d patients, t-test p=%.4g'%(np.mean(wp),len(wp),stats.ttest_1samp(wp,0).pvalue))

# --- B. does the model score sleep-like BACKGROUND higher? (negatives only) ---
print('\n=== B. NEGATIVE decisions: model probability vs sleep proxy ===')
neg=d[(d.label==0)&d.recording_id.isin(cache)].copy()
sub=neg.sample(n=min(60000,len(neg)),random_state=0).copy()
pr=[];am=[]
for r,t in zip(sub.recording_id.values,sub.decision_time_seconds.values):
    p,a=window_stats(r,t); pr.append(p); am.append(a)
sub['proxy']=pr; sub['amp']=am; sub=sub.dropna(subset=['proxy'])
sub.to_csv(OUT/'negatives_with_proxy.csv',index=False)
print('negatives with proxy:',len(sub))
for tag in TAGS:
    dd=pd.read_csv(ROOT/f'outputs/cv/{tag}/out_of_fold_predictions.csv',
                   usecols=['recording_id','decision_time_seconds','label','probability'])
    m=sub.merge(dd[dd.label==0][['recording_id','decision_time_seconds','probability']],
                on=['recording_id','decision_time_seconds'],how='left',suffixes=('','_t'))
    pcol=m['probability_t'] if 'probability_t' in m else m['probability']
    ok=m.dropna(subset=['probability_t']) if 'probability_t' in m else m
    rho=stats.spearmanr(ok.proxy,ok.probability_t).statistic
    # within-recording spearman
    rr=[stats.spearmanr(g.proxy,g.probability_t).statistic for _,g in ok.groupby('recording_id') if len(g)>=50]
    rr=[x for x in rr if np.isfinite(x)]
    # tertile split within recording
    ok=ok.copy(); ok['q']=ok.groupby('recording_id').proxy.rank(pct=True)
    hi=ok[ok.q>=2/3].probability_t.mean(); lo=ok[ok.q<=1/3].probability_t.mean()
    print(f'{tag:38s} rho_pooled={rho:+.3f}  rho_within_rec(median over {len(rr)} recs)={np.median(rr):+.3f}  '
          f'mean prob sleep-like={hi:.4f} wake-like={lo:.4f} diff={hi-lo:+.4f}')
