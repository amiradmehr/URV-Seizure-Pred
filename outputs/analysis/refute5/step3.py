import numpy as np, pandas as pd, pickle
from scipy import stats, optimize
ROOT='/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'
Z=pickle.load(open(f'{ROOT}/outputs/analysis/refute5/cache.pkl','rb'))
D=Z['det']; subj=Z['sz2subj'].reindex(D.index); R=Z['R'].set_index('tag')
sizes=subj.value_counts().to_numpy().astype(float); N=sizes.sum(); k=len(sizes)
m_bar=N/k; m_A=(sizes**2).sum()/N
za=stats.norm.ppf(.975); zb=stats.norm.ppf(.80)
def icc_anova(y,g):
    df=pd.DataFrame({'y':np.asarray(y,float),'g':np.asarray(g)}); gb=df.groupby('g').y
    ni=gb.size().to_numpy().astype(float); mi=gb.mean().to_numpy(); n=ni.sum(); kk=len(ni); gm=df.y.mean()
    MSB=(ni*(mi-gm)**2).sum()/(kk-1); MSW=((df.y-df.g.map(gb.mean()))**2).sum()/(n-kk)
    m0=(n-(ni**2).sum()/n)/(kk-1)
    return (MSB-MSW)/(MSB+(m0-1)*MSW)

# ---- CLAIM 3: DE for a PAIRED (same-seizure) contrast ----
sop10=[t for t in D.columns if t.endswith('sop10')]
print('sop10 configs:',len(sop10))
diffs=[]; iccs=[]
cols=D[sop10].to_numpy().astype(float)
for i in range(len(sop10)):
    for j in range(i+1,len(sop10)):
        dd=cols[:,i]-cols[:,j]
        if dd.std()==0: continue
        iccs.append(icc_anova(dd,subj))
iccs=np.array(iccs)
print(f'ICC of the PAIRED DIFFERENCE (config i - config j), {len(iccs)} pairs: '
      f'mean={np.nanmean(iccs):.4f} median={np.nanmedian(iccs):.4f} p90={np.nanpercentile(iccs,90):.3f}')
print(f'  => paired DE = 1+(m_A-1)*ICC_diff = {1+(m_A-1)*np.nanmean(iccs):.3f}  (analyst used DE=1.687 from the MARGINAL ICC)')
print(f'  marginal ICC 0.160 -> DE 1.687; paired ICC {np.nanmean(iccs):.3f} -> DE {1+(m_A-1)*np.nanmean(iccs):.3f}')

# empirical clustered bootstrap of the paired difference for a few pairs
rng=np.random.default_rng(5); subs=subj.unique(); pos_by=[np.where(subj.values==s)[0] for s in subs]
des=[]
for (i,j) in [(0,1),(2,3),(4,5),(6,7),(0,10),(3,12)]:
    dd=cols[:,i]-cols[:,j]; disc=(dd!=0).mean()
    if disc==0: continue
    bb=np.empty(2000)
    for b in range(2000):
        sel=np.concatenate([pos_by[q] for q in rng.integers(0,k,k)]); bb[b]=dd[sel].mean()
    var_naive=(dd.var(ddof=1))/N
    des.append(bb.var(ddof=1)/var_naive)
print(f'  empirical clustered DE of paired differences (6 pairs): {np.round(des,3)}  mean={np.mean(des):.3f}')

# ---- CLAIM 3: MDD reproduction ----
def n_two(p1,p2):
    pb=(p1+p2)/2
    return (za*np.sqrt(2*pb*(1-pb))+zb*np.sqrt(p1*(1-p1)+p2*(1-p2)))**2/(p2-p1)**2
def n_one(p0,p1):
    return (za*np.sqrt(p0*(1-p0))+zb*np.sqrt(p1*(1-p1)))**2/(p1-p0)**2
def solve(f,p0,n):
    g=lambda p2:f(p0,p2)-n
    return optimize.brentq(g,p0+1e-6,0.999) if g(0.999)<0 else np.nan
n_eff=169.5
print('\nMDD reproduction (n_eff=169.5):')
for p0 in (0.05,0.10,0.20):
    print(f'  p0={p0}: two-cohort MDD={solve(n_two,p0,n_eff):.3f} (+{solve(n_two,p0,n_eff)-p0:.3f})  '
          f'one-sample MDD={solve(n_one,p0,n_eff):.3f} (+{solve(n_one,p0,n_eff)-p0:.3f})')
def mcn(pi_d,n):
    g=lambda dl:((za*np.sqrt(pi_d)+zb*np.sqrt(pi_d-dl**2))**2/dl**2)-n
    return optimize.brentq(g,1e-5,pi_d-1e-6)
disc=[]
for i in range(len(sop10)):
    for j in range(i+1,len(sop10)): disc.append((cols[:,i]!=cols[:,j]).mean())
disc=np.array(disc); pdm=np.median(disc)
print(f'  discordance median={pdm:.4f}  McNemar MDD: n=286 -> {mcn(pdm,286):.4f}; '
      f'n_eff=169.5 -> {mcn(pdm,169.5):.4f}; n_eff with PAIRED DE ({1+(m_A-1)*np.nanmean(iccs):.2f}) -> '
      f'{mcn(pdm,286/(1+(m_A-1)*np.nanmean(iccs))):.4f}')
# actual observed config-vs-config differences in sens
res=pd.read_csv(f'{ROOT}/outputs/sweep/results.csv')
for sop in (10,30):
    g=res[res.sop_min==sop]
    print(f'  ACTUAL sens_at_1ph spread SOP={sop}: min={g.sens_at_1ph.min():.4f} max={g.sens_at_1ph.max():.4f} '
          f'range={g.sens_at_1ph.max()-g.sens_at_1ph.min():.4f}')
L=pd.read_csv(f'{ROOT}/outputs/analysis/refute5/lift_ci.csv')
for sop in (10,30):
    g=L[L.sop==sop]; print(f'  chance-corrected lift SOP={sop}: sd={g.lift.std():.4f} max={g.lift.max():.4f} range={g.lift.max()-g.lift.min():.4f}')

# ---- CLAIM 5/6: per-arm vs total ----
print('\nSample size: two-independent-cohort formula returns PER ARM')
ne=n_two(0.10,0.30); print(f'  0.10 vs 0.30: n_eff per arm={ne:.1f} -> raw per arm={ne*1.687:.0f} -> TOTAL raw={2*ne*1.687:.0f}')
ne1=n_one(0.10,0.30); print(f'  one-sample 0.30 vs fixed 0.10: n_eff={ne1:.1f} -> raw={ne1*1.687:.0f}')
d_obs=0.0046; p0=0.055; p1=p0+d_obs
ne2=n_two(p0,p1); print(f'  chasing observed lift {d_obs}: two-cohort n_eff per arm={ne2:.0f} -> raw per arm={ne2*1.687:.0f} -> TOTAL={2*ne2*1.687:.0f}')
ne3=n_one(p0,p1); print(f'                              one-sample n_eff={ne3:.0f} -> raw={ne3*1.687:.0f} (analyst reported 67,601)')
print(f'  patients at {N/k:.2f}/pt: two-cohort per-arm {ne2*1.687/(N/k):.0f}; TOTAL {2*ne2*1.687/(N/k):.0f}; one-sample {ne3*1.687/(N/k):.0f}')

# ---- CLAIM 7: ICC uncertainty -> ceiling uncertainty ----
rng2=np.random.default_rng(9)
tag_med=D.columns[np.argsort([D[t].mean() for t in D.columns])[len(D.columns)//2]]
ic=[]
for b in range(2000):
    pick=rng2.integers(0,k,k); sel=np.concatenate([pos_by[q] for q in pick])
    g=np.concatenate([np.full(len(pos_by[q]),bi) for bi,q in enumerate(pick)])
    v=D[tag_med].to_numpy().astype(float)[sel]
    try: ic.append(icc_anova(v,g))
    except Exception: pass
ic=np.array(ic); ic=ic[np.isfinite(ic)]
lo,hi=np.percentile(ic,[2.5,97.5])
print(f'\nCLAIM 7: patient-bootstrap 95% CI on ICC for the median config ({tag_med}) = [{lo:.3f},{hi:.3f}]')
print(f'  -> ceiling k/ICC ranges [{k/max(hi,1e-9):.0f}, {k/max(lo,1e-9):.0f}] (inf if ICC<=0); point at 0.160 = {k/0.160:.0f}')
# across-config spread of the ICC
ricc=np.load(f'{ROOT}/outputs/analysis/refute5/real_icc.npy'); ricc=ricc[np.isfinite(ricc)]
print(f'  across-config ICC p10..p90 = {np.percentile(ricc,10):.3f}..{np.percentile(ricc,90):.3f} -> ceiling {k/np.percentile(ricc,90):.0f} .. (neg ICC = no ceiling)')
print(f'  analyst n_eff at m_bar={m_bar:.2f}: {k*m_bar/(1+(m_bar-1)*0.160):.1f}   vs headline n_eff (Kish) {N/(1+(m_A-1)*0.160):.1f}')
