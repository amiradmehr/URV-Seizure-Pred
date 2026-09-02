import numpy as np, pandas as pd, os, pickle
ROOT='/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'
res=pd.read_csv(f'{ROOT}/outputs/sweep/results.csv')
tags=[t for t in res.tag if pd.notna(res.loc[res.tag==t,'sop_min'].iloc[0])]
rng=np.random.default_rng(11)
rows=[]; store={}
for t in tags:
    p=f'{ROOT}/outputs/cv/{t}/out_of_fold_predictions.csv'
    d=pd.read_csv(p,usecols=['subject','recording_id','label','target_seizure_id','probability','decision_time_seconds'])
    neg=d[d.label==0]; pos=d[d.label==1]
    nneg=len(neg); thr=np.sort(neg.probability.to_numpy())[::-1][int(np.floor(nneg/60.0))]
    fa_rate=(neg.probability.to_numpy()>=thr).mean()*60.0
    # observed, per-seizure ndec
    g=pos.assign(a=pos.probability>=thr).groupby('target_seizure_id')
    obs=g.a.any(); ndec_per=g.size()
    sz_subj=g.subject.first(); sz_rec=g.recording_id.first()
    # ---- POOLED block null (analyst's) ----
    n=neg.sort_values(['recording_id','decision_time_seconds'])
    alarm=(n.probability.to_numpy()>=thr).astype(np.int8); rid=n.recording_id.to_numpy()
    tt=n.decision_time_seconds.to_numpy()
    cs=np.concatenate([[0],np.cumsum(alarm)])
    ndec_mean=int(round(ndec_per.mean()))
    st=np.arange(0,len(alarm)-ndec_mean+1)
    ok=rid[st]==rid[st+ndec_mean-1]
    pooled=((cs[st+ndec_mean]-cs[st])>0)[ok].mean()
    # contiguity check: are the ndec negatives inside a block actually consecutive in time?
    contig=(tt[st+ndec_mean-1]-tt[st]==60.0*(ndec_mean-1))
    pooled_contig=((cs[st+ndec_mean]-cs[st])>0)[ok&contig].mean()
    frac_contig=(ok&contig).sum()/ok.sum()
    # ---- MATCHED null: for each seizure, blocks of ITS OWN ndec from the SAME SUBJECT ----
    by_sub={}
    for s,gg in n.groupby('subject',sort=False):
        by_sub[s]=((gg.probability.to_numpy()>=thr).astype(np.int8), gg.recording_id.to_numpy())
    matched=[]; matched_rec=[]
    by_rec={}
    for rr,gg in n.groupby('recording_id',sort=False):
        by_rec[rr]=(gg.probability.to_numpy()>=thr).astype(np.int8)
    for sid in obs.index:
        k=int(ndec_per.loc[sid]); s=sz_subj.loc[sid]; rr=sz_rec.loc[sid]
        if s in by_sub:
            a,rd=by_sub[s]
            if len(a)>=k:
                c=np.concatenate([[0],np.cumsum(a)]); stt=np.arange(0,len(a)-k+1)
                o=rd[stt]==rd[stt+k-1]
                if o.sum()>0: matched.append((((c[stt+k]-c[stt])>0)[o]).mean())
        if rr in by_rec:
            a=by_rec[rr]
            if len(a)>=k:
                c=np.concatenate([[0],np.cumsum(a)]); stt=np.arange(0,len(a)-k+1)
                matched_rec.append((((c[stt+k]-c[stt])>0)).mean())
    rows.append(dict(tag=t,sop=int(res.loc[res.tag==t,'sop_min'].iloc[0]),
        reported=float(res.loc[res.tag==t,'sens_at_1ph'].iloc[0]),obs=obs.mean(),
        fa_per_h=fa_rate,ndec_mean=ndec_per.mean(),ndec_min=ndec_per.min(),ndec_max=ndec_per.max(),
        pooled=pooled,pooled_contig=pooled_contig,frac_contig=frac_contig,
        matched_subj=np.mean(matched),n_matched=len(matched),
        matched_rec=np.mean(matched_rec),n_matched_rec=len(matched_rec)))
    store[t]=dict(obs=obs,subj=sz_subj,ndec=ndec_per)
R=pd.DataFrame(rows)
R.to_csv(f'{ROOT}/outputs/analysis/refute4/chance_matched.csv',index=False)
pickle.dump(store,open(f'{ROOT}/outputs/analysis/refute4/det.pkl','wb'))
print('recon max abs err %.6f'%(R.obs-R.reported).abs().max())
print('FA/h at threshold: mean %.4f min %.4f max %.4f'%(R.fa_per_h.mean(),R.fa_per_h.min(),R.fa_per_h.max()))
for s in (10,30):
    g=R[R.sop==s]
    print(f'--- SOP={s}  ndec mean {g.ndec_mean.mean():.2f} (min {g.ndec_min.min()} max {g.ndec_max.max()})')
    print(f'   obs         {g.obs.mean():.4f}')
    for c in ['pooled','pooled_contig','matched_subj','matched_rec']:
        d=g.obs-g[c]
        print(f'   {c:14s} {g[c].mean():.4f}  lift {d.mean():+.5f}  above {int((d>0).sum())}/{len(g)}  liftsd {d.std():.4f} liftmax {d.max():+.4f}')
    print(f'   frac fully-contiguous blocks {g.frac_contig.mean():.3f}')
