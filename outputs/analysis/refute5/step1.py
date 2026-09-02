import numpy as np, pandas as pd, os, pickle, time
ROOT='/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'
res=pd.read_csv(f'{ROOT}/outputs/sweep/results.csv')
tags=[t for t in res.tag if os.path.exists(f'{ROOT}/outputs/cv/{t}/out_of_fold_predictions.csv')]
print('tags with OOF:',len(tags))
det={}; szmax={}; sz2subj=None; sz2rec=None; szn={}
per_config_null={}   # tag -> Series per seizure: null prob from same recording
rows=[]
t0=time.time()
for t in tags:
    d=pd.read_csv(f'{ROOT}/outputs/cv/{t}/out_of_fold_predictions.csv',
        usecols=['subject','recording_id','label','target_seizure_id','probability','decision_time_seconds'])
    neg=d[d.label==0]; pos=d[d.label==1]
    nn=len(neg); thr=np.sort(neg.probability.to_numpy())[::-1][int(np.floor(nn/60.0))]
    a=pos.assign(al=pos.probability.values>=thr).groupby('target_seizure_id').al.any()
    det[t]=a
    szmax[t]=pos.groupby('target_seizure_id').probability.max()
    if sz2subj is None:
        sz2subj=pos.groupby('target_seizure_id').subject.first()
        sz2rec=pos.groupby('target_seizure_id').recording_id.first()
        szn_g=pos.groupby('target_seizure_id').size()
    # per-recording alarm-block null, matched to each seizure's own recording and own n_dec
    n=neg.sort_values(['recording_id','decision_time_seconds'])
    alarm=(n.probability.to_numpy()>=thr).astype(np.int8)
    rid=n.recording_id.to_numpy()
    cs=np.concatenate([[0],np.cumsum(alarm)])
    # index ranges per recording
    uniq,start_idx=np.unique(rid,return_index=True)
    order=np.argsort(start_idx); uniq=uniq[order]; start_idx=start_idx[order]
    end_idx=np.append(start_idx[1:],len(rid))
    recspan={u:(s,e) for u,s,e in zip(uniq,start_idx,end_idx)}
    # global pooled null with ndec = rounded mean (analyst's version)
    ndec_a=int(round(szn_g.mean()))
    starts=np.arange(0,len(alarm)-ndec_a+1)
    ok=rid[starts]==rid[starts+ndec_a-1]
    pooled=((cs[starts+ndec_a]-cs[starts])>0)[ok].mean()
    # seizure-matched null
    vals=[]
    for sid in a.index:
        rec=sz2rec[sid]; k=int(szn_g[sid])
        if rec not in recspan: vals.append(np.nan); continue
        s,e=recspan[rec]
        L=e-s
        if L<k: vals.append(np.nan); continue
        st=np.arange(s,e-k+1)
        vals.append((((cs[st+k]-cs[st])>0)).mean())
    per_config_null[t]=pd.Series(vals,index=a.index)
    # subject-matched null (pool all negatives of the seizure's subject, blocks within recording)
    rows.append(dict(tag=t,obs=a.mean(),pooled_null=pooled,
                     matched_null=np.nanmean(per_config_null[t].values),
                     matched_null_n=int(np.isfinite(per_config_null[t].values).sum()),
                     sop=res.loc[res.tag==t,'sop_min'].iloc[0]))
print('elapsed',time.time()-t0)
R=pd.DataFrame(rows)
pickle.dump(dict(det=pd.DataFrame(det),szmax=pd.DataFrame(szmax),sz2subj=sz2subj,sz2rec=sz2rec,
                 szn=szn_g,nulls=pd.DataFrame(per_config_null),R=R),
            open(f'{ROOT}/outputs/analysis/refute5/cache.pkl','wb'))
R.to_csv(f'{ROOT}/outputs/analysis/refute5/nulls.csv',index=False)
for sop,g in R.groupby(R.sop.fillna(-1)):
    print(f"SOP={sop} n={len(g)} obs={g.obs.mean():.4f} pooled_null={g.pooled_null.mean():.4f} "
          f"matched_null={g.matched_null.mean():.4f} lift_pooled={(g.obs-g.pooled_null).mean():+.4f} "
          f"lift_matched={(g.obs-g.matched_null).mean():+.4f} above_matched={(g.obs>g.matched_null).sum()}/{len(g)}")
