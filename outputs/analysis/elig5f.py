import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats
R='/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'
CF=Path(f'{R}/data/interim/chunk_features')
m=pd.read_csv(f'{R}/data/interim/manifests/seizure_manifest.csv')
d=pd.read_csv(f'{R}/outputs/analysis/elig_reasons.csv')
m=m.merge(d[['seizure_id','clust','short_start','nonbg','gap_prev_end_min']],on='seizure_id')
m['rec']=m.apply(lambda r: f"sub-{r['subject']:03d}_ses-{int(r['session']):02d}_task-{r['task']}_run-{int(r['run']):02d}",axis=1)
CH=5.0; rng=np.random.default_rng(0)
def aucs(A,B):
    n,p=A.shape; mm=B.shape[0]
    Z=np.vstack([A,B])
    Rk=stats.rankdata(Z,axis=0)
    return (Rk[:n].sum(axis=0)-n*(n+1)/2.0)/(n*mm)
rows=[]
for rec,g in m.groupby('rec'):
    fp=CF/f'{rec}_features.npy'
    if not fp.exists(): continue
    F=np.load(fp); av=np.load(CF/f'{rec}_availability.npy')
    cols=np.where(av)[0]
    if len(cols)==0: continue
    F=F[:,cols].astype(np.float64); nC=F.shape[0]
    ok=np.isfinite(F).all(axis=1)
    seiz=g[['onset_seconds','duration_seconds']].values.astype(float)
    t=np.arange(nC)*CH
    base=np.ones(nC,bool)
    for on,du in seiz: base &= ~((t>on-7200)&(t<on+du+7200))
    base &= ok
    bidx=np.where(base)[0]
    if len(bidx)<200: continue
    B=F[bidx]
    for _,s in g.iterrows():
        on=float(s['onset_seconds'])
        i0=max(int((on-600)//CH),0); i1=min(int(on//CH),nC)
        idx=np.arange(i0,i1); idx=idx[ok[idx]]
        if len(idx)<60: continue
        a=aucs(F[idx],B)
        if len(bidx)<len(idx)+100: continue
        st=int(rng.integers(0,len(bidx)-len(idx)))
        ns=bidx[st:st+len(idx)]; rest=np.setdiff1d(bidx,ns)
        an=aucs(F[ns],F[rest])
        rows.append(dict(seizure_id=s['seizure_id'],subject=s['subject'],rec=rec,
            elig=bool(s['eligible_for_prediction']),clust=bool(s['clust']),short=bool(s['short_start']),
            vigilance=s['vigilance'],gap=s['gap_prev_end_min'],
            sep=float(np.mean(np.abs(a-0.5))),sepmax=float(np.max(np.abs(a-0.5))),
            null=float(np.mean(np.abs(an-0.5))),n_pre=len(idx),n_base=len(bidx)))
r=pd.DataFrame(rows); r.to_csv(f'{R}/outputs/analysis/sep.csv',index=False)
print("seizures analysed:",len(r),"of 883; recordings:",r.rec.nunique())
print("\nmean |AUC-0.5| over available features, 10-min pre-onset vs far-interictal baseline (same recording):")
grp=[('ELIGIBLE (the 317)',r[r.elig]),('EXCLUDED clustered',r[(~r.elig)&r.clust]),('EXCLUDED short-start only',r[(~r.elig)&r.short&(~r.clust)])]
for lab,s in grp:
    if len(s)<3: continue
    print(f"  {lab:26s} n={len(s):3d} sep={s.sep.mean():.4f} med={s.sep.median():.4f} | null={s.null.mean():.4f} | sep-null={s.sep.mean()-s.null.mean():+.4f}")
e=r[r.elig]; c=r[(~r.elig)&r.clust]
u,p=stats.mannwhitneyu(c.sep,e.sep); print(f"\nMWU clustered-excluded vs eligible: p={p:.3g} A={u/(len(c)*len(e)):.3f}")
for th in [10,30,60]:
    c2=c[c.gap>th]
    if len(c2)>5:
        u,p2=stats.mannwhitneyu(c2.sep,e.sep)
        print(f"  gap>{th:2d}min subset n={len(c2):3d} sep={c2.sep.mean():.4f} p={p2:.3g} A={u/(len(c2)*len(e)):.3f}")
print("\npaired sep vs null: eligible p=%.3g diff=%+.4f | clustered p=%.3g diff=%+.4f"%(
 stats.ttest_rel(e.sep,e.null).pvalue,(e.sep-e.null).mean(),stats.ttest_rel(c.sep,c.null).pvalue,(c.sep-c.null).mean()))
