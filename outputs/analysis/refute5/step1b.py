import numpy as np, pandas as pd, os, pickle, time
ROOT='/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'
res=pd.read_csv(f'{ROOT}/outputs/sweep/results.csv')
tags=[t for t in res.tag if os.path.exists(f'{ROOT}/outputs/cv/{t}/out_of_fold_predictions.csv')]
det={}; szmax={}; nulls={}; rows=[]; sz2subj=None
t0=time.time()
for t in tags:
    d=pd.read_csv(f'{ROOT}/outputs/cv/{t}/out_of_fold_predictions.csv',
        usecols=['subject','recording_id','label','target_seizure_id','probability','decision_time_seconds'])
    neg=d[d.label==0]; pos=d[d.label==1]
    nn=len(neg); thr=np.sort(neg.probability.to_numpy())[::-1][int(np.floor(nn/60.0))]
    a=pos.assign(al=pos.probability.values>=thr).groupby('target_seizure_id').al.any()
    det[t]=a; szmax[t]=pos.groupby('target_seizure_id').probability.max()
    szn=pos.groupby('target_seizure_id').size()          # PER CONFIG
    rec=pos.groupby('target_seizure_id').recording_id.first()
    sub=pos.groupby('target_seizure_id').subject.first()
    if sz2subj is None: sz2subj=sub
    n=neg.sort_values(['recording_id','decision_time_seconds'])
    alarm=(n.probability.to_numpy()>=thr).astype(np.int8); rid=n.recording_id.to_numpy()
    cs=np.concatenate([[0],np.cumsum(alarm)])
    uniq,si=np.unique(rid,return_index=True); o=np.argsort(si); uniq=uniq[o]; si=si[o]
    ei=np.append(si[1:],len(rid)); span=dict(zip(uniq,zip(si,ei)))
    ndec_a=int(round(szn.mean()))
    st=np.arange(0,len(alarm)-ndec_a+1); ok=rid[st]==rid[st+ndec_a-1]
    pooled=((cs[st+ndec_a]-cs[st])>0)[ok].mean()
    vals=[]
    for sid in a.index:
        r=rec[sid]; k=int(szn[sid])
        if r not in span: vals.append(np.nan); continue
        s,e=span[r]
        if e-s<k: vals.append(np.nan); continue
        s2=np.arange(s,e-k+1); vals.append((((cs[s2+k]-cs[s2])>0)).mean())
    nulls[t]=pd.Series(vals,index=a.index)
    rows.append(dict(tag=t,ndec_mean=float(szn.mean()),ndec_a=ndec_a,obs=a.mean(),pooled=pooled,
                     matched=np.nanmean(vals),sop=res.loc[res.tag==t,'sop_min'].iloc[0]))
print('elapsed',time.time()-t0)
R=pd.DataFrame(rows); R.to_csv(f'{ROOT}/outputs/analysis/refute5/nulls.csv',index=False)
pickle.dump(dict(det=pd.DataFrame(det),szmax=pd.DataFrame(szmax),nulls=pd.DataFrame(nulls),
                 sz2subj=sz2subj,R=R),open(f'{ROOT}/outputs/analysis/refute5/cache.pkl','wb'))
for sop,g in R.groupby(R.sop.fillna(-1)):
    print(f"SOP={sop:5} n={len(g):2d} ndec_mean={g.ndec_mean.mean():.2f} obs={g.obs.mean():.4f} "
          f"pooled={g.pooled.mean():.4f} matched={g.matched.mean():.4f} "
          f"lift_pool={(g.obs-g.pooled).mean():+.4f} lift_match={(g.obs-g.matched).mean():+.4f} "
          f"above_pool={(g.obs>g.pooled).sum()}/{len(g)} above_match={(g.obs>g.matched).sum()}/{len(g)}")
print()
print(R.sort_values('obs',ascending=False).head(5).to_string(index=False))
