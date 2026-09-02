import numpy as np, pandas as pd, pickle
from scipy import stats
ROOT='/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'
Z=pickle.load(open(f'{ROOT}/outputs/analysis/refute5/cache.pkl','rb'))
D=Z['det']; S=Z['szmax']; NU=Z['nulls']; subj=Z['sz2subj'].reindex(D.index); R=Z['R']
print('D',D.shape,'subjects',subj.nunique())
sizes=subj.value_counts().to_numpy().astype(float); N=sizes.sum(); k=len(sizes)
m_bar=N/k; m_A=(sizes**2).sum()/N
print(f'N={N:.0f} k={k} m_bar={m_bar:.3f} m_A(Kish)={m_A:.3f} max={sizes.max():.0f}')

# ---- 1. significance of observed-vs-matched-null lift, per config, patient-clustered ----
rng=np.random.default_rng(11)
subs=subj.unique(); pos_by=[np.where(subj.values==s)[0] for s in subs]
out=[]
for t in D.columns:
    obs=D[t].to_numpy().astype(float); nul=NU[t].to_numpy().astype(float)
    ok=np.isfinite(nul); d=obs-nul
    # patient-clustered bootstrap of the mean lift
    bb=np.empty(2000)
    for b in range(2000):
        sel=np.concatenate([pos_by[p] for p in rng.integers(0,k,k)])
        sel=sel[np.isfinite(nul[sel])]
        bb[b]=d[sel].mean()
    lo,hi=np.percentile(bb,[2.5,97.5])
    out.append(dict(tag=t,sop=R.set_index('tag').sop.get(t,np.nan),lift=d[ok].mean(),
                    se=bb.std(ddof=1),lo=lo,hi=hi,z=d[ok].mean()/bb.std(ddof=1)))
L=pd.DataFrame(out)
L.to_csv(f'{ROOT}/outputs/analysis/refute5/lift_ci.csv',index=False)
for sop,g in L.groupby(L.sop.fillna(-1)):
    print(f'SOP={sop}: mean lift={g.lift.mean():+.4f}  median clustered SE={g.se.median():.4f}  '
          f'configs with CI excluding 0: {(g.lo>0).sum()}/{len(g)}   max z={g.z.max():.2f}')
print(L.sort_values('lift',ascending=False).head(4).to_string(index=False))

# ---- 2. ICC checks ----
def icc_anova(y,g):
    df=pd.DataFrame({'y':np.asarray(y,float),'g':np.asarray(g)}); gb=df.groupby('g').y
    ni=gb.size().to_numpy().astype(float); mi=gb.mean().to_numpy(); n=ni.sum(); kk=len(ni); gm=df.y.mean()
    MSB=(ni*(mi-gm)**2).sum()/(kk-1); MSW=((df.y-df.g.map(gb.mean()))**2).sum()/(n-kk)
    m0=(n-(ni**2).sum()/n)/(kk-1)
    return (MSB-MSW)/(MSB+(m0-1)*MSW)
icc_det=np.array([icc_anova(D[t].astype(float),subj) for t in D.columns])
Sr=S.rank(pct=True); icc_scr=np.array([icc_anova(Sr[t],subj) for t in Sr.columns])
print(f'\nICC(detected) mean={np.nanmean(icc_det):.4f} median={np.nanmedian(icc_det):.4f} p10={np.nanpercentile(icc_det,10):.3f} p90={np.nanpercentile(icc_det,90):.3f}')
print(f'ICC(seizure-level score, rank-pct) mean={np.nanmean(icc_scr):.4f} median={np.nanmedian(icc_scr):.4f} p90={np.nanpercentile(icc_scr,90):.3f}')

# ---- 3. NULL simulation: is ANOVA-ICC / cluster-boot DE inflated at true ICC=0? ----
def cluster_boot_DE(v,reps=1500,rg=None):
    p=v.mean();
    if p<=0 or p>=1: return np.nan
    bb=np.empty(reps)
    for b in range(reps):
        sel=np.concatenate([pos_by[q] for q in rg.integers(0,k,k)])
        bb[b]=v[sel].mean()
    return bb.std(ddof=1)**2/(p*(1-p)/N)
rg=np.random.default_rng(3)
# (a) real data DE
real_de=[]; real_icc=[]
for t in D.columns:
    v=D[t].to_numpy().astype(float)
    de=cluster_boot_DE(v,1200,rg)
    if np.isfinite(de): real_de.append(de); real_icc.append(icc_anova(v,subj))
real_de=np.array(real_de)
print(f'\nREAL DE_emp: mean={real_de.mean():.3f} median={np.median(real_de):.3f} p90={np.percentile(real_de,90):.3f}')
# (b) permute detections across seizures -> destroys patient clustering, keeps p and cluster sizes
perm_de=[]; perm_icc=[]
for t in D.columns:
    v=D[t].to_numpy().astype(float).copy()
    for r_ in range(3):
        vp=rg.permutation(v)
        de=cluster_boot_DE(vp,800,rg)
        if np.isfinite(de): perm_de.append(de); perm_icc.append(icc_anova(vp,subj))
perm_de=np.array(perm_de); perm_icc=np.array(perm_icc)
print(f'PERMUTED (true ICC=0) DE_emp: mean={perm_de.mean():.3f} median={np.median(perm_de):.3f} p90={np.percentile(perm_de,90):.3f}')
print(f'PERMUTED ANOVA-ICC: mean={np.nanmean(perm_icc):+.4f} sd={np.nanstd(perm_icc):.4f} p95={np.nanpercentile(perm_icc,95):.4f}')
# permutation p-value for the real ICC
print(f'real mean ICC={np.nanmean(icc_det):.4f} vs permuted p95={np.nanpercentile(perm_icc,95):.4f}')
np.save(f'{ROOT}/outputs/analysis/refute5/perm_icc.npy',perm_icc)
np.save(f'{ROOT}/outputs/analysis/refute5/real_icc.npy',icc_det)
