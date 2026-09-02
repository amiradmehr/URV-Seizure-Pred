import pandas as pd, numpy as np, glob, os
from scipy import stats
files=sorted(glob.glob('outputs/cv/*/out_of_fold_predictions.csv'))
files=[f for f in files if '__sop30' not in f]  # keep sop10-labelled sets comparable? no -> keep all
files=sorted(glob.glob('outputs/cv/*/out_of_fold_predictions.csv'))
print('configs',len(files))
per={}
for f in files:
    d=pd.read_csv(f,usecols=['subject','label','target_seizure_id','probability'],dtype={'subject':str})
    d['p']=d.probability
    # within-subject percentile of each positive among that subject's negatives
    out=[]
    for s,g in d.groupby('subject'):
        neg=np.sort(g.loc[g.label==0,'p'].values)
        pos=g[g.label==1]
        if len(neg)==0 or len(pos)==0: continue
        pct=np.searchsorted(neg,pos.p.values,side='left')/len(neg)
        out.append(pd.DataFrame({'sid':pos.target_seizure_id.values,'subject':s,'pct':pct}))
    if not out: continue
    o=pd.concat(out).groupby(['sid','subject']).pct.mean()
    per[os.path.basename(os.path.dirname(f))]=o
M=pd.DataFrame(per)
print('seizures',len(M),'configs',M.shape[1])
M['mean_pct']=M.mean(axis=1)
M=M.reset_index()
sz=pd.read_csv('data/interim/manifests/seizure_manifest.csv')
sz['recording_id']=sz.seizure_id.str.replace(r'_seizure-\d+$','',regex=True)
M=M.merge(sz[['seizure_id','vigilance','recording_id','onset_seconds']],left_on='sid',right_on='seizure_id')
# has_prior: another seizure earlier in same recording
prior=[]
for _,r in M.iterrows():
    g=sz[sz.recording_id==r.recording_id]
    prior.append(bool((g.onset_seconds<r.onset_seconds).any()))
M['has_prior']=prior
M.to_csv('outputs/analysis/verify/oof_perseizure.csv',index=False)

def test(mask_a,mask_b,name,labels):
    a=M.loc[mask_a,'mean_pct']; b=M.loc[mask_b,'mean_pct']
    u,p=stats.mannwhitneyu(a,b)
    t,pt=stats.ttest_ind(a,b,equal_var=False)
    print(f'{name}: {labels[0]} n={len(a)} mean={a.mean():.4f} | {labels[1]} n={len(b)} mean={b.mean():.4f} | diff={a.mean()-b.mean():+.4f} MWU p={p:.3f} Welch p={pt:.3f}')
    # subject-clustered bootstrap
    rng=np.random.default_rng(0); subs=M.subject.unique(); bs=[]
    for _ in range(2000):
        pick=rng.choice(subs,len(subs),replace=True)
        rows=pd.concat([M[M.subject==s] for s in pick])
        aa=rows.loc[rows.index.isin(M.index[mask_a])] if False else None
        bs.append(np.nan)
    return
print()
print('=== SEIZURE-LEVEL (unit = seizure, not config) ===')
test(M.vigilance=='asleep', M.vigilance=='awake','vigilance',['asleep','awake'])
test(M.has_prior, ~M.has_prior,'prior-seizure',['has_prior','isolated'])
# subject-clustered permutation for vigilance
sub=M[M.vigilance.isin(['asleep','awake'])].copy()
rng=np.random.default_rng(0)
obs=sub.loc[sub.vigilance=='asleep','mean_pct'].mean()-sub.loc[sub.vigilance=='awake','mean_pct'].mean()
null=[]
for _ in range(5000):
    v=sub.groupby('subject').vigilance.transform(lambda x: rng.permutation(x.values))
    null.append(sub.loc[v=='asleep','mean_pct'].mean()-sub.loc[v=='awake','mean_pct'].mean())
null=np.array(null); p=2*min((null>=obs).mean(),(null<=obs).mean())
print(f'vigilance within-patient permutation: obs diff={obs:+.4f}  p={p:.4f}  null sd={null.std():.4f}')
