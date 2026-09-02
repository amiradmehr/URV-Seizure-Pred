import numpy as np, pandas as pd, os, json
from sklearn.metrics import average_precision_score
ROOT='/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'
res=pd.read_csv(f'{ROOT}/outputs/sweep/results.csv')
rng=np.random.default_rng(1)

# ---- A. what is CHANCE seizure-level sensitivity at 1 FA/h? ----
# a seizure fires if ANY of its ~n_dec pre-onset decisions crosses thr.
# alarms are temporally autocorrelated, so the effective number of independent
# trials is << n_dec. measure it on real negatives: slide blocks of n_dec
# consecutive decisions through interictal time and ask how often a block alarms.
rows=[]
for tag in res.tag:
    p=f'{ROOT}/outputs/cv/{tag}/out_of_fold_predictions.csv'
    if not os.path.exists(p): continue
    d=pd.read_csv(p,usecols=['recording_id','label','target_seizure_id','probability','decision_time_seconds'])
    neg=d.loc[d.label==0,'probability'].to_numpy()
    nneg=len(neg); thr=np.sort(neg)[::-1][int(np.floor(nneg/60.0))]
    ndec=int(round(d[d.label==1].groupby('target_seizure_id').size().mean()))
    obs=d[d.label==1].assign(a=d.probability>=thr).groupby('target_seizure_id').a.any().mean()
    # blocks of ndec consecutive negative decisions inside one recording
    n=d[d.label==0].sort_values(['recording_id','decision_time_seconds'])
    alarm=(n.probability.to_numpy()>=thr).astype(np.int8)
    rid=n.recording_id.to_numpy()
    hit=[];  # rolling any() within recording
    cs=np.concatenate([[0],np.cumsum(alarm)])
    same=np.ones(len(alarm),bool)
    starts=np.arange(0,len(alarm)-ndec+1)
    ok=rid[starts]==rid[starts+ndec-1]
    blk=(cs[starts+ndec]-cs[starts])>0
    chance=blk[ok].mean()
    indep=1-(1-1/60.0)**ndec
    rows.append(dict(tag=tag,sop=(lambda v: int(v) if pd.notna(v) else -1)(res.loc[res.tag==tag,"sop_min"].iloc[0]),ndec=ndec,
                     obs=obs,chance_block=chance,chance_indep=indep,n_blocks=int(ok.sum())))
C=pd.DataFrame(rows)
C.to_csv(f'{ROOT}/outputs/analysis/chance_seizure_sens.csv',index=False)
for sop,g in C.groupby('sop'):
    print(f"SOP={sop} min  n_dec/seizure={g.ndec.iloc[0]}  configs={len(g)}")
    print(f"   observed seizure sens@1FA/h : mean={g.obs.mean():.4f}  range {g.obs.min():.4f}-{g.obs.max():.4f}")
    print(f"   CHANCE (real alarm autocorr): mean={g.chance_block.mean():.4f}  range {g.chance_block.min():.4f}-{g.chance_block.max():.4f}")
    print(f"   chance if decisions indep   : {g.chance_indep.iloc[0]:.4f}")
    print(f"   observed - chance           : mean={ (g.obs-g.chance_block).mean():+.4f}  "
          f"n_above={(g.obs>g.chance_block).sum()}/{len(g)}")

# ---- B. measurement precision of AP/chance (patient-clustered bootstrap) ----
tag='spectral-gru__large__h45__sop10'
d=pd.read_csv(f'{ROOT}/outputs/cv/{tag}/out_of_fold_predictions.csv',
              usecols=['subject','label','probability'])
subs=d.subject.unique(); idxs=[d.index[d.subject.values==s].to_numpy() for s in subs]
y=d.label.to_numpy(); s=d.probability.to_numpy()
B=400; out=np.empty(B)
for b in range(B):
    pick=rng.integers(0,len(subs),len(subs))
    sel=np.concatenate([idxs[p] for p in pick])
    yy=y[sel]
    if yy.sum()==0: out[b]=np.nan; continue
    out[b]=average_precision_score(yy,s[sel])/yy.mean()
out=out[np.isfinite(out)]
print(f"\nAP/chance for {tag}: point={res.loc[res.tag==tag,'ap_over_chance'].iloc[0]:.4f}")
print(f"  patient-clustered bootstrap SD = {out.std(ddof=1):.4f}  (B={len(out)})")
print(f"  95% CI = [{np.percentile(out,2.5):.3f}, {np.percentile(out,97.5):.3f}]")
print(f"  across-42-config SD from sweep  = {res.ap_over_chance.std():.4f}")
print(f"  ratio (config spread / single-config sampling SD) = {res.ap_over_chance.std()/out.std(ddof=1):.3f}")
json.dump({'ap_boot_sd':float(out.std(ddof=1)),'ap_ci':[float(np.percentile(out,2.5)),float(np.percentile(out,97.5))],
           'sweep_sd':float(res.ap_over_chance.std())},open(f'{ROOT}/outputs/analysis/ap_precision.json','w'),indent=2)
