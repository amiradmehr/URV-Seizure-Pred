import numpy as np, pandas as pd, glob, os, json
from scipy import stats

ROOT='/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'
res=pd.read_csv(f'{ROOT}/outputs/sweep/results.csv')
tags=[t for t in res.tag if os.path.exists(f'{ROOT}/outputs/cv/{t}/out_of_fold_predictions.csv')]
print("configs with OOF:",len(tags))

det={}      # tag -> Series indexed by seizure id, bool detected at 1 FA/h
szscore={}  # tag -> Series seizure-level max prob
sz2subj=None
recon=[]
for t in tags:
    d=pd.read_csv(f'{ROOT}/outputs/cv/{t}/out_of_fold_predictions.csv',
                  usecols=['subject','label','target_seizure_id','probability','fold'])
    neg=d.loc[d.label==0,'probability'].to_numpy()
    pos=d[d.label==1]
    nneg=len(neg); allowed=int(np.floor(nneg/60.0))     # 1 FA per interictal hour, 60 s stride
    thr=np.sort(neg)[::-1][allowed]                      # (allowed+1)-th largest -> <=allowed FPs at >=thr
    a=pos.assign(al=pos.probability>=thr).groupby('target_seizure_id').al.any()
    det[t]=a
    szscore[t]=pos.groupby('target_seizure_id').probability.max()
    if sz2subj is None:
        sz2subj=pos.groupby('target_seizure_id').subject.first()
    recon.append((t, a.mean(), float(res.loc[res.tag==t,'sens_at_1ph'].iloc[0])))

rc=pd.DataFrame(recon,columns=['tag','mine','reported'])
print("reconstruction of sens_at_1ph  max abs err = %.4f  mean abs err = %.4f"%
      ((rc.mine-rc.reported).abs().max(),(rc.mine-rc.reported).abs().mean()))

D=pd.DataFrame(det)           # 286 x K detected
S=pd.DataFrame(szscore)
subj=sz2subj.reindex(D.index)
print("seizure x config matrix:",D.shape,"subjects:",subj.nunique())

# ---------- cluster sizes ----------
sizes=subj.value_counts().to_numpy().astype(float)
N=sizes.sum(); k=len(sizes)
m_bar=N/k
m_A=(sizes**2).sum()/N                      # Kish effective cluster size
m0=(N-(sizes**2).sum()/N)/(k-1)             # ANOVA m0
print(f"\nCLUSTERS: N={int(N)} seizures, k={k} patients, mean m={m_bar:.2f}, "
      f"Kish m_A={m_A:.2f}, ANOVA m0={m0:.2f}, max={int(sizes.max())}")

def icc_anova(y,g):
    df=pd.DataFrame({'y':np.asarray(y,float),'g':np.asarray(g)})
    gb=df.groupby('g').y
    ni=gb.size().to_numpy().astype(float); mi=gb.mean().to_numpy()
    N=ni.sum(); k=len(ni); gm=df.y.mean()
    if k<2 or N<=k: return np.nan
    MSB=(ni*(mi-gm)**2).sum()/(k-1)
    MSW=((df.y-df.g.map(gb.mean()))**2).sum()/(N-k)
    m0=(N-(ni**2).sum()/N)/(k-1)
    if MSB+(m0-1)*MSW==0: return np.nan
    return (MSB-MSW)/(MSB+(m0-1)*MSW)

icc_det=np.array([icc_anova(D[t].astype(float),subj) for t in D.columns])
# continuous seizure-level score, rank-normalised per config so scales are comparable
Sr=S.rank(pct=True)
icc_scr=np.array([icc_anova(Sr[t],subj) for t in Sr.columns])
print(f"ICC(detected@1FA/h) over {len(icc_det)} configs: mean={np.nanmean(icc_det):.4f} "
      f"median={np.nanmedian(icc_det):.4f} sd={np.nanstd(icc_det):.4f} "
      f"p10={np.nanpercentile(icc_det,10):.3f} p90={np.nanpercentile(icc_det,90):.3f}")
print(f"ICC(seizure-level score pctile): mean={np.nanmean(icc_scr):.4f} median={np.nanmedian(icc_scr):.4f}")

ICC=float(np.nanmean(icc_det)); ICC=max(ICC,0.0)
DE_kish=1+(m_A-1)*ICC
DE_mean=1+(m_bar-1)*ICC
n_eff_kish=N/DE_kish; n_eff_mean=N/DE_mean
print(f"\nDESIGN EFFECT: ICC={ICC:.4f} -> DE(Kish m_A={m_A:.2f})={DE_kish:.3f}  n_eff={n_eff_kish:.1f}")
print(f"               ICC={ICC:.4f} -> DE(mean m={m_bar:.2f})={DE_mean:.3f}  n_eff={n_eff_mean:.1f}")
ICC_hi=float(np.nanpercentile(icc_det,90))
print(f"  at ICC p90={ICC_hi:.3f}: DE={1+(m_A-1)*ICC_hi:.2f} n_eff={N/(1+(m_A-1)*ICC_hi):.1f}")

# ---------- empirical check: clustered bootstrap SD of sensitivity ----------
rng=np.random.default_rng(0)
subs=subj.unique()
by_sub={s:D.index[subj.values==s] for s in subs}
B=4000
tag0=rc.sort_values('mine').iloc[len(rc)//2].tag      # median config
v=D[tag0].to_numpy(); idx={i:j for j,i in enumerate(D.index)}
pos_by_sub=[np.array([idx[i] for i in by_sub[s]]) for s in subs]
boot=np.empty(B)
for b in range(B):
    pick=rng.integers(0,len(subs),len(subs))
    sel=np.concatenate([pos_by_sub[p] for p in pick])
    boot[b]=v[sel].mean()
p_hat=v.mean()
se_cluster=boot.std(ddof=1)
se_binom=np.sqrt(p_hat*(1-p_hat)/N)
print(f"\nEMPIRICAL SE of seizure-level sens (config {tag0}, p={p_hat:.4f}):")
print(f"  patient-cluster bootstrap SE = {se_cluster:.4f}")
print(f"  naive binomial SE (n=286)    = {se_binom:.4f}")
print(f"  variance inflation (DE_emp)  = {(se_cluster/se_binom)**2:.3f}  -> n_eff = {N/((se_cluster/se_binom)**2):.1f}")
DE_emp=(se_cluster/se_binom)**2

# average the empirical DE over several configs
des=[]
for t in list(D.columns):
    v=D[t].to_numpy(); p=v.mean()
    if p<=0 or p>=1: continue
    bb=np.empty(1500)
    for b in range(1500):
        pick=rng.integers(0,len(subs),len(subs))
        sel=np.concatenate([pos_by_sub[p2] for p2 in pick])
        bb[b]=v[sel].mean()
    des.append(bb.std(ddof=1)**2/(p*(1-p)/N))
des=np.array(des)
print(f"  DE_empirical across {len(des)} configs: mean={des.mean():.3f} median={np.median(des):.3f} p90={np.percentile(des,90):.3f}")
json.dump({'ICC':ICC,'m_A':m_A,'m_bar':m_bar,'k':int(k),'N':int(N),
           'DE_kish':DE_kish,'DE_emp_mean':float(des.mean()),'DE_emp_median':float(np.median(des))},
          open(f'{ROOT}/outputs/analysis/cluster_params.json','w'),indent=2)
