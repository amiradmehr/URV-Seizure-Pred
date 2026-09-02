import numpy as np, pandas as pd, pickle
ROOT='/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'
Z=pickle.load(open(f'{ROOT}/outputs/analysis/refute5/cache.pkl','rb'))
D=Z['det'].astype(float); subj=Z['sz2subj'].reindex(D.index).to_numpy()
subs=np.unique(subj); k=len(subs); idx=[np.where(subj==s)[0] for s in subs]
V=D.to_numpy()   # 286 x 44

def icc_vec(Y,gid,ng):
    # Y: n x C ; gid: n ints in [0,ng)
    n,C=Y.shape
    cnt=np.bincount(gid,minlength=ng).astype(float)
    sums=np.zeros((ng,C)); np.add.at(sums,gid,Y)
    mi=sums/cnt[:,None]
    gm=Y.mean(0)
    MSB=((cnt[:,None]*(mi-gm)**2).sum(0))/(ng-1)
    resid=Y-mi[gid]
    MSW=(resid**2).sum(0)/(n-ng)
    m0=(n-(cnt**2).sum()/n)/(ng-1)
    return (MSB-MSW)/(MSB+(m0-1)*MSW)

gid0=np.searchsorted(subs,subj)
icc0=icc_vec(V,gid0,k)
print(f'point mean ICC across 44 configs = {np.nanmean(icc0):.4f} (analyst 0.160)')
rng=np.random.default_rng(31); B=1500; out=np.empty(B)
for b in range(B):
    pick=rng.integers(0,k,k)
    sel=np.concatenate([idx[p] for p in pick])
    g=np.concatenate([np.full(len(idx[p]),bi) for bi,p in enumerate(pick)])
    ic=icc_vec(V[sel],g,k)
    out[b]=np.nanmean(ic)
lo,hi=np.percentile(out,[2.5,97.5])
print(f'patient-bootstrap 95% CI on the MEAN ICC = [{lo:.4f},{hi:.4f}]  P(ICC<=0)={np.mean(out<=0):.3f}')
for nm,f in (('DE(Kish m_A=5.476)',lambda i:1+(5.4755-1)*i),('n_eff',lambda i:286/(1+(5.4755-1)*i))):
    print(f'  {nm}: point {f(np.nanmean(icc0)):.3f}  CI [{f(hi):.3f},{f(lo):.3f}]')
cl=[93/max(x,1e-6) for x in (hi,np.nanmean(icc0),lo)]
print(f'  ceiling k/ICC: point {93/np.nanmean(icc0):.0f}  CI [{cl[0]:.0f}, {"inf" if lo<=0 else f"{cl[2]:.0f}"}]')
