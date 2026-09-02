import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats
R='/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'
CF=Path(f'{R}/data/interim/chunk_features')
m=pd.read_csv(f'{R}/data/interim/manifests/seizure_manifest.csv')
d=pd.read_csv('/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred/outputs/analysis/elig_reasons.csv'); m=m.merge(d[['seizure_id','clust','short_start','nonbg','gap_prev_end_min']],on='seizure_id')
m['rec']=m.apply(lambda r: f"sub-{r['subject']:03d}_ses-{int(r['session']):02d}_task-{r['task']}_run-{int(r['run']):02d}",axis=1)
CH=1280/256.0  # 5 s per chunk
rng=np.random.default_rng(0)

def auc_mat(A,B):
    # column-wise AUC of A vs B
    n,p=A.shape; mB=B.shape[0]
    out=np.empty(p)
    for j in range(p):
        u=stats.mannwhitneyu(A[:,j],B[:,j],alternative='two-sided').statistic
        out[j]=u/(n*mB)
    return out

rows=[]
for rec,g in m.groupby('rec'):
    fp=CF/f'{rec}_features.npy'
    if not fp.exists(): continue
    F=np.load(fp); av=np.load(CF/f'{rec}_availability.npy')
    cols=np.where(av)[0]
    if len(cols)==0: continue
    F=F[:,cols]; nC=F.shape[0]
    ok=np.isfinite(F).all(axis=1)
    seiz=g[['onset_seconds','duration_seconds']].values
    # baseline mask: >=2h from any seizure interval
    base=np.ones(nC,bool)
    t=np.arange(nC)*CH
    for on,du in seiz:
        base &= ~((t> on-7200) & (t< on+du+7200))
    base &= ok
    B=F[base]
    if B.shape[0]<200: continue
    for _,s in g.iterrows():
        on=float(s['onset_seconds'])
        i0=int(np.floor((on-600)/CH)); i1=int(np.floor(on/CH))
        i0=max(i0,0)
        idx=np.arange(i0,min(i1,nC))
        idx=idx[ok[idx]]
        if len(idx)<60: continue
        A=F[idx]
        a=auc_mat(A,B)
        # null: random contiguous baseline block of same size
        bidx=np.where(base)[0]
        if len(bidx)<len(idx)+50: continue
        st=rng.integers(0,len(bidx)-len(idx))
        nullsel=bidx[st:st+len(idx)]
        rest=np.setdiff1d(bidx,nullsel)
        an=auc_mat(F[nullsel],F[rest])
        rows.append(dict(seizure_id=s['seizure_id'],subject=s['subject'],rec=rec,
            elig=bool(s['eligible_for_prediction']),clust=bool(s['clust']),short=bool(s['short_start']),
            vigilance=s['vigilance'], gap=s['gap_prev_end_min'],
            sep=np.mean(np.abs(a-0.5)), sepmax=np.max(np.abs(a-0.5)),
            null=np.mean(np.abs(an-0.5)), nullmax=np.max(np.abs(an-0.5)),
            n_pre=len(idx), n_base=B.shape[0]))
r=pd.DataFrame(rows); r.to_csv('/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred/outputs/analysis/sep.csv',index=False)
print("seizures analysed:",len(r),"recordings:",r.rec.nunique())
print("\nmean |AUC-0.5| of 10-min pre-onset window vs far-interictal baseline (27 feats, within recording):")
for lab,s in [('ELIGIBLE (317-set)',r[r.elig]),('EXCLUDED-clustered',r[(~r.elig)&r.clust]),('EXCLUDED-short_start',r[(~r.elig)&r.short&~r.clust])]:
    if len(s)==0: continue
    print(f"  {lab:22s} n={len(s):3d}  sep={s.sep.mean():.4f} (med {s.sep.median():.4f})  null={s.null.mean():.4f}  sep-null={s.sep.mean()-s.null.mean():+.4f}")
e=r[r.elig]; c=r[(~r.elig)&r.clust]
if len(c)>5:
    u,p=stats.mannwhitneyu(c.sep,e.sep); A=u/(len(c)*len(e))
    print(f"\nMWU excluded-clustered vs eligible sep: p={p:.3g}  A={A:.3f}")
    # postictal-free subset
    c2=c[c.gap>10]
    if len(c2)>5:
        u,p2=stats.mannwhitneyu(c2.sep,e.sep); A2=u/(len(c2)*len(e))
        print(f"  restricted to clustered with prior-seizure gap >10 min (postictal-free 10-min window): n={len(c2)} sep={c2.sep.mean():.4f} p={p2:.3g} A={A2:.3f}")
    c3=c[c.gap>60]
    if len(c3)>5:
        u,p3=stats.mannwhitneyu(c3.sep,e.sep); A3=u/(len(c3)*len(e))
        print(f"  restricted to gap >60 min: n={len(c3)} sep={c3.sep.mean():.4f} p={p3:.3g} A={A3:.3f}")
print("\nsep vs null paired (eligible only): t-test p=%.3g, mean diff=%+.4f"%(stats.ttest_rel(e.sep,e.null).pvalue, (e.sep-e.null).mean()))
print("sep vs null paired (excluded-clustered): p=%.3g, mean diff=%+.4f"%(stats.ttest_rel(c.sep,c.null).pvalue,(c.sep-c.null).mean()))
