import numpy as np, pandas as pd, os
ROOT='/work/pi_phuc_umass_edu/aradmehr/Github/URV-Seizure-Pred'
res=pd.read_csv(f'{ROOT}/outputs/sweep/results.csv')
tags=[t for t in res.tag if str(t).endswith('sop10')][:8]
rows=[]
for t in tags:
    d=pd.read_csv(f'{ROOT}/outputs/cv/{t}/out_of_fold_predictions.csv',
        usecols=['recording_id','label','target_seizure_id','probability','decision_time_seconds'])
    pos=d[d.label==1]
    # approximate onset per seizure = last pre-onset decision time + stride
    onset=pos.groupby('target_seizure_id').agg(rec=('recording_id','first'),t=('decision_time_seconds','max'))
    onset['t']=onset['t']+60.0
    neg=d[d.label==0].sort_values(['recording_id','decision_time_seconds']).reset_index(drop=True)
    nn=len(d[d.label==0]); thr=np.sort(d.loc[d.label==0,'probability'].to_numpy())[::-1][int(np.floor(nn/60.0))]
    a=pos.assign(al=pos.probability.values>=thr).groupby('target_seizure_id').al.any()
    # mask negatives within +-BUF of any onset
    for BUF in (0.0, 3600.0, 7200.0):
        keep=np.ones(len(neg),bool)
        if BUF>0:
            for sid,r in onset.iterrows():
                m=(neg.recording_id.values==r['rec'])&(np.abs(neg.decision_time_seconds.values-r['t'])<=BUF)
                keep&=~m
        nk=neg[keep]
        alarm=(nk.probability.to_numpy()>=thr).astype(np.int8); rid=nk.recording_id.to_numpy()
        cs=np.concatenate([[0],np.cumsum(alarm)]); nd=10
        st=np.arange(0,len(alarm)-nd+1); ok=rid[st]==rid[st+nd-1]
        rows.append(dict(tag=t,buf=BUF,obs=a.mean(),null=((cs[st+nd]-cs[st])>0)[ok].mean(),
                         alarm_rate=alarm.mean(),n=int(ok.sum())))
X=pd.DataFrame(rows)
print(X.groupby('buf').agg(obs=('obs','mean'),null=('null','mean'),alarm_rate=('alarm_rate','mean')).assign(lift=lambda z:z.obs-z.null))
print()
print(X.pivot(index='tag',columns='buf',values='null').round(4).to_string())
